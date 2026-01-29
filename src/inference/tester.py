"""
PII Tester for NerGuard.

This module provides the PIITester class for complete PII detection inference,
including sliding window support for long documents and optional LLM routing.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn.functional as F

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_CONTEXT_LENGTH,
    OVERLAP,
    STRIDE,
)
from src.core.model_loader import load_model_and_tokenizer, load_labels, get_device
from src.core.metrics import compute_entropy_confidence
from src.inference.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class PIITester:
    """
    Complete PII detection tester with sliding window and LLM routing support.

    Example:
        >>> tester = PIITester(llm_routing=True)
        >>> results = tester.analyze_text("Dear John Smith, your SSN is 555-01-4433.")
        >>> for r in results:
        ...     if r["label"] != "O":
        ...         print(f"{r['token']}: {r['label']}")
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        label_path: Optional[str] = None,
        llm_routing: bool = False,
        llm_source: str = "openai",
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        """
        Initialize the PII Tester.

        Args:
            model_path: Path to the trained model
            label_path: Path to label mapping (optional, uses model config if None)
            llm_routing: Whether to enable LLM disambiguation
            llm_source: LLM backend ("openai" or "ollama")
            entropy_threshold: Entropy threshold for LLM routing
            confidence_threshold: Confidence threshold for LLM routing
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.device = get_device() if device is None else torch.device(device)
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
        self.llm_routing = llm_routing

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            device=str(self.device),
            eval_mode=True,
        )

        # Get label mappings
        if label_path:
            self.id2label = load_labels(label_path, as_id2label=True)
        else:
            self.id2label = self.model.config.id2label

        self.label2id = {v: k for k, v in self.id2label.items()}

        # Initialize LLM router if enabled
        self.router = None
        if llm_routing:
            try:
                self.router = LLMRouter(source=llm_source)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM router: {e}. Falling back to model-only.")
                self.llm_routing = False

        logger.info(f"PIITester initialized (LLM routing: {self.llm_routing})")

    def analyze_text(
        self,
        text: str,
        lang: str = "en",
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Analyze text for PII entities.

        Args:
            text: Input text to analyze
            lang: Language code
            verbose: Whether to print detailed output

        Returns:
            List of token analysis results with keys:
                - token: The token string
                - label: Final predicted label
                - confidence: Model confidence
                - entropy: Model entropy
                - source: "Model" or "LLM"
                - char_start: Character start position
                - char_end: Character end position
        """
        if verbose:
            print(f"\n[{lang.upper()}] Processing Text ({len(text)} chars)...")

        # Tokenize without special tokens for offset mapping
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        all_input_ids = encoding["input_ids"]
        all_offsets = encoding["offset_mapping"]
        total_tokens = len(all_input_ids)

        if verbose:
            print(f"   Total Tokens: {total_tokens}")

        # Initialize results buffer
        results: List[Optional[Dict[str, Any]]] = [None] * total_tokens
        max_model_content = MAX_CONTEXT_LENGTH - 2  # Account for [CLS] and [SEP]

        # Process in chunks if necessary
        if total_tokens <= max_model_content:
            # Single pass
            self._process_chunk(
                all_input_ids,
                all_offsets,
                text,
                lang,
                results,
                start_idx=0,
                verbose=verbose,
            )
        else:
            # Sliding window
            if verbose:
                print(f"   [INFO] Using sliding window (stride: {STRIDE})")

            for i in range(0, total_tokens, STRIDE):
                chunk_end = min(i + max_model_content, total_tokens)
                chunk_ids = all_input_ids[i:chunk_end]
                chunk_offsets = all_offsets[i:chunk_end]

                # Determine how many tokens to actually use from this chunk
                tokens_to_use = len(chunk_ids)
                if chunk_end < total_tokens:
                    tokens_to_use = min(len(chunk_ids), STRIDE)

                self._process_chunk(
                    chunk_ids[:tokens_to_use],
                    chunk_offsets[:tokens_to_use],
                    text,
                    lang,
                    results,
                    start_idx=i,
                    verbose=verbose,
                )

        # Filter out None results
        return [r for r in results if r is not None]

    def _process_chunk(
        self,
        input_ids: List[int],
        offsets: List[Tuple[int, int]],
        text: str,
        lang: str,
        results: List[Optional[Dict[str, Any]]],
        start_idx: int,
        verbose: bool,
    ) -> None:
        """Process a chunk of tokens."""
        # Add special tokens
        full_input_ids = (
            [self.tokenizer.cls_token_id]
            + input_ids
            + [self.tokenizer.sep_token_id]
        )

        # Get predictions
        tensor_input = torch.tensor([full_input_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(tensor_input).logits[0]

        # Remove special token predictions
        logits = logits[1:-1]

        # Compute metrics
        entropy, confidence, pred_ids = compute_entropy_confidence(logits)

        # Convert to lists
        entropy = entropy.cpu().tolist()
        confidence = confidence.cpu().tolist()
        pred_ids = pred_ids.cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Process each token
        for j, token in enumerate(tokens):
            global_idx = start_idx + j

            pred_id = pred_ids[j]
            model_label = self.id2label.get(pred_id, "O")
            conf = confidence[j]
            ent = entropy[j]
            char_start, char_end = offsets[j]

            final_label = model_label
            source = "Model"

            # Get previous label for BIO consistency
            prev_label = "O"
            if global_idx > 0 and results[global_idx - 1] is not None:
                prev_label = results[global_idx - 1]["label"]

            # LLM routing if enabled and uncertain
            # Uses selective entity routing: only route entities where LLM helps
            if self.llm_routing and self.router:
                should_route = LLMRouter.should_route(
                    current_pred=model_label,
                    entropy=ent,
                    confidence=conf,
                    entropy_threshold=self.entropy_threshold,
                    confidence_threshold=self.confidence_threshold,
                    use_selective_routing=True,
                )

                if should_route:
                    source = f"LLM ({self.router.source})"

                    llm_result = self.router.disambiguate(
                        target_token=token,
                        full_text=text,
                        char_start=char_start,
                        char_end=char_end,
                        current_pred=model_label,
                        prev_label=prev_label,
                        lang=lang,
                    )

                    if llm_result.get("is_pii"):
                        final_label = llm_result.get("corrected_label", model_label)
                    else:
                        final_label = "O"

            # Store result
            results[global_idx] = {
                "token": token,
                "label": final_label,
                "confidence": conf,
                "entropy": ent,
                "source": source,
                "char_start": char_start,
                "char_end": char_end,
            }

            if verbose:
                print(f"   {token:<15}\t{final_label:<15}\t{conf:.4f}\t{ent:.4f}\t{source}")

    def get_entities(
        self,
        text: str,
        lang: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        Extract PII entities from text.

        Args:
            text: Input text
            lang: Language code

        Returns:
            List of entities with keys:
                - text: Entity text
                - label: Entity type (without B-/I- prefix)
                - start: Character start position
                - end: Character end position
                - confidence: Average confidence
        """
        results = self.analyze_text(text, lang, verbose=False)

        entities = []
        current_entity = None

        for r in results:
            label = r["label"]

            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)

                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    "text": r["token"].replace("▁", " ").replace("##", "").strip(),
                    "label": entity_type,
                    "start": r["char_start"],
                    "end": r["char_end"],
                    "confidences": [r["confidence"]],
                }

            elif label.startswith("I-") and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    current_entity["text"] += r["token"].replace("▁", " ").replace("##", "")
                    current_entity["end"] = r["char_end"]
                    current_entity["confidences"].append(r["confidence"])

            else:
                # Outside or different entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)

        # Compute average confidence and clean up
        for e in entities:
            e["confidence"] = sum(e["confidences"]) / len(e["confidences"])
            del e["confidences"]
            e["text"] = e["text"].strip()

        return entities

    def redact_text(
        self,
        text: str,
        lang: str = "en",
        replacement: str = "█" * 8,
    ) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text
            lang: Language code
            replacement: Replacement string for PII

        Returns:
            Redacted text
        """
        entities = self.get_entities(text, lang)

        # Sort by position (reverse to maintain offsets)
        entities.sort(key=lambda e: e["start"], reverse=True)

        result = text
        for entity in entities:
            result = result[:entity["start"]] + replacement + result[entity["end"]:]

        return result


# For backwards compatibility
def analyze_sentence(text: str, model_path: str = DEFAULT_MODEL_PATH, llm_routing: bool = False):
    """
    Convenience function for quick analysis.

    Deprecated: Use PIITester class directly for better control.
    """
    tester = PIITester(model_path=model_path, llm_routing=llm_routing)
    return tester.analyze_text(text, verbose=True)
