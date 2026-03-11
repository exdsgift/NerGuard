#!/usr/bin/env python
"""
NerGuard Redact — detect PII with source tracking and produce redacted text.

Runs the full hybrid pipeline (base model + regex + optional LLM routing)
on a single text input and shows per-entity detection sources.

Usage:
    python -m src.scripts.redact --text "Dear John Smith, your SSN is 555-01-4433"
    python -m src.scripts.redact --file input.txt
    python -m src.scripts.redact --text "..." --llm --llm-source ollama
    python -m src.scripts.redact --text "..." --json
"""

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Suppress noisy warnings before importing heavy libs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_CONTEXT_LENGTH,
    STRIDE,
    REGEX_VALIDATABLE_ENTITIES,
    ENTITY_THRESHOLDS,
    ROUTABLE_ENTITIES,
    BLOCKED_ENTITIES,
    ROUTABLE_I_ENTITIES,
)
from src.core.model_loader import load_model_and_tokenizer, get_device
from src.core.metrics import compute_entropy_confidence
from src.inference.regex_validator import RegexValidator
from src.inference.entity_router import EntitySpecificRouter
from src.inference.span_assembler import assemble_entity_spans
from src.utils.colors import Colors


def _run_base_model(text, model, tokenizer, device, id2label):
    """Run base NER model with sliding window for long texts.

    If the text fits within MAX_CONTEXT_LENGTH tokens, a single forward pass is used.
    Otherwise, a sliding window (stride = STRIDE, overlap = OVERLAP) processes chunks
    and keeps the highest-confidence prediction per token position.
    """
    # Tokenize without truncation to get full token sequence
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    all_input_ids = encoding["input_ids"]
    all_offsets = encoding["offset_mapping"]
    total_tokens = len(all_input_ids)

    max_content = MAX_CONTEXT_LENGTH - 2  # reserve [CLS] and [SEP]

    if total_tokens <= max_content:
        # Single pass — add special tokens and run
        return _run_single_pass(all_input_ids, all_offsets, model, tokenizer, device, id2label)

    # Sliding window for long texts
    # Per-token buffers: keep prediction with highest confidence
    best_preds = ["O"] * total_tokens
    best_entropy = [0.0] * total_tokens
    best_conf = [0.0] * total_tokens

    for chunk_start in range(0, total_tokens, STRIDE):
        chunk_end = min(chunk_start + max_content, total_tokens)
        chunk_ids = all_input_ids[chunk_start:chunk_end]
        chunk_offsets = all_offsets[chunk_start:chunk_end]

        # Only use STRIDE tokens from each chunk (except the last)
        tokens_to_use = len(chunk_ids) if chunk_end >= total_tokens else min(len(chunk_ids), STRIDE)

        preds, ent_vals, conf_vals, _ = _run_single_pass(
            chunk_ids[:tokens_to_use],
            chunk_offsets[:tokens_to_use],
            model, tokenizer, device, id2label,
        )

        # Merge: keep highest-confidence prediction per token
        # Skip [CLS]/[SEP] — _run_single_pass returns them at index 0 and -1
        for j in range(1, len(preds) - 1):
            global_idx = chunk_start + (j - 1)  # j-1 because index 0 is [CLS]
            if global_idx < total_tokens and conf_vals[j] > best_conf[global_idx]:
                best_preds[global_idx] = preds[j]
                best_entropy[global_idx] = ent_vals[j]
                best_conf[global_idx] = conf_vals[j]

    # Build full output with [CLS] at start and [SEP] at end (matching single-pass format)
    subword_preds = ["O"] + best_preds + ["O"]
    entropy_vals = [0.0] + best_entropy + [0.0]
    conf_vals = [0.0] + best_conf + [0.0]
    offset_mapping = [(0, 0)] + all_offsets + [(0, 0)]

    return subword_preds, entropy_vals, conf_vals, offset_mapping


def _run_single_pass(input_ids, offsets, model, tokenizer, device, id2label):
    """Run a single forward pass with [CLS]/[SEP] wrapping."""
    full_ids = [tokenizer.cls_token_id] + list(input_ids) + [tokenizer.sep_token_id]
    full_offsets = [(0, 0)] + list(offsets) + [(0, 0)]

    input_tensor = torch.tensor([full_ids], device=device)
    attention_mask = torch.ones_like(input_tensor)

    with torch.no_grad():
        logits = model(input_tensor, attention_mask=attention_mask).logits[0]

    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
    conf, pred_ids_tensor = torch.max(probs, dim=-1)

    pred_ids = pred_ids_tensor.cpu().tolist()
    entropy_vals = entropy.cpu().tolist()
    conf_vals = conf.cpu().tolist()
    subword_preds = [id2label.get(pid, "O") for pid in pred_ids]

    return subword_preds, entropy_vals, conf_vals, full_offsets


def _determine_entity_source(token_sources: List[Optional[str]]) -> str:
    """Determine entity-level source from per-token sources."""
    sources = {s for s in token_sources if s is not None}
    if "llm routed" in sources:
        return "llm routed"
    if "regex override" in sources and "base model" not in sources:
        return "regex override"
    if sources == {"base model"}:
        return "base model"
    if "base + regex" in sources or ("base model" in sources and "regex override" in sources):
        return "base + regex"
    if sources:
        return next(iter(sources))
    return "base model"


def _assemble_entities(text, subword_preds, conf_vals, offset_mapping, sources):
    """Assemble BIO tokens into entities with text, confidence, and source."""
    entities = []
    current = None

    for i, label in enumerate(subword_preds):
        cs, ce = offset_mapping[i]
        if cs == ce == 0:  # special token
            continue

        if label.startswith("B-"):
            if current:
                entities.append(current)
            entity_type = label[2:]
            current = {
                "label": entity_type,
                "start": cs,
                "end": ce,
                "confidences": [conf_vals[i]],
                "token_sources": [sources[i]],
            }
        elif label.startswith("I-") and current and label[2:] == current["label"]:
            current["end"] = ce
            current["confidences"].append(conf_vals[i])
            current["token_sources"].append(sources[i])
        else:
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    # Finalize: extract text, compute avg confidence, determine source
    for e in entities:
        raw = text[e["start"]:e["end"]]
        # Strip leading/trailing whitespace and sentence-ending punctuation
        # that the tokenizer may attach to the last subword
        stripped = raw.strip().rstrip(".,;:!?)(").lstrip("(")
        if stripped != raw:
            offset = raw.index(stripped) if stripped else 0
            e["start"] += offset
            e["end"] = e["start"] + len(stripped)
        e["text"] = stripped
        e["confidence"] = sum(e["confidences"]) / len(e["confidences"])
        e["source"] = _determine_entity_source(e["token_sources"])
        del e["confidences"], e["token_sources"]

    return [e for e in entities if e["text"]]


def redact_pipeline(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    llm_routing: bool = False,
    llm_source: str = "openai",
    llm_model: str = "gpt-4o",
    model=None,
    tokenizer=None,
) -> Tuple[List[Dict], str]:
    """
    Run the full hybrid pipeline on text and return entities + redacted text.

    Args:
        model: Pre-loaded model instance (optional). If None, loads from model_path.
        tokenizer: Pre-loaded tokenizer instance (optional). If None, loads from model_path.

    Returns:
        Tuple of (entities, redacted_text) where entities have keys:
        text, label, start, end, confidence, source
    """
    device = get_device()
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_path, device=str(device), eval_mode=True)
    id2label = model.config.id2label
    label2id = {v: int(k) for k, v in id2label.items()}

    # Phase 1: Base model inference
    subword_preds, entropy_vals, conf_vals, offset_mapping = _run_base_model(
        text, model, tokenizer, device, id2label
    )
    n_tokens = len(subword_preds)

    # Initialize per-token source tracking
    sources: List[Optional[str]] = [
        "base model" if subword_preds[i] != "O" else None
        for i in range(n_tokens)
    ]

    # Phase 2: Regex pre-scan
    regex_validator = RegexValidator()
    pred_ids_array = np.array([label2id.get(p, label2id.get("O", 0)) for p in subword_preds])
    offset_array = np.array(offset_mapping)
    old_preds = pred_ids_array.copy()

    prescan_ids = regex_validator.correct_predictions(
        text=text,
        offset_mapping=offset_array,
        preds=pred_ids_array,
        id2label=id2label,
        label2id=label2id,
        correct_partial=True,
    )

    # Update sources for tokens changed by regex pre-scan
    for i in range(n_tokens):
        if prescan_ids[i] != old_preds[i]:
            sources[i] = "regex override"
    subword_preds = [id2label.get(int(pid), "O") for pid in prescan_ids]

    # Mark base-model entities that overlap with regex hints as "base + regex"
    regex_hints = regex_validator.find_regex_hints(text)
    for i in range(n_tokens):
        if sources[i] == "base model":
            cs, ce = offset_mapping[i]
            if cs == ce == 0:
                continue
            for hint_start, hint_end, _ in regex_hints:
                if cs < hint_end and ce > hint_start:
                    sources[i] = "base + regex"
                    break

    # Phase 3: LLM routing (optional)
    if llm_routing:
        entity_router = EntitySpecificRouter(
            entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            routable_entities=ROUTABLE_ENTITIES,
            blocked_entities=BLOCKED_ENTITIES,
            routable_i_entities=ROUTABLE_I_ENTITIES,
            entity_thresholds=ENTITY_THRESHOLDS,
        )

        entity_spans = assemble_entity_spans(
            pred_labels=subword_preds,
            entropy_flat=entropy_vals,
            conf_flat=conf_vals,
            offset_flat=offset_mapping,
            entity_router=entity_router,
        )

        from src.inference.llm_router import LLMRouter
        router = LLMRouter(source=llm_source, model=llm_model, ollama_model=llm_model)

        for span in entity_spans:
            if not span.is_uncertain:
                continue
            anchor_idx = span.indices[0]
            sw_start, sw_end = offset_mapping[anchor_idx]
            if sw_start == sw_end == 0:
                continue

            # Check if regex can skip LLM
            if regex_validator.can_skip_llm(
                entity_class=span.entity_class,
                text=text,
                char_start=span.char_start,
                char_end=span.char_end,
            ):
                for idx in span.indices:
                    if sources[idx] and sources[idx] != "regex override":
                        sources[idx] = "base + regex"
                continue

            prev_label = subword_preds[anchor_idx - 1] if anchor_idx > 0 else "O"
            try:
                res = router.disambiguate_span(
                    span_text=text[span.char_start:span.char_end],
                    token_count=len(span.indices),
                    full_text=text,
                    span_start=span.char_start,
                    span_end=span.char_end,
                    current_pred=span.entity_class,
                    prev_label=prev_label,
                )

                if res.get("is_pii"):
                    entity_out = res.get("corrected_label", f"B-{span.entity_class}")
                    entity_class = entity_out.replace("B-", "").replace("I-", "")
                    # Extend span to include continuation subwords that
                    # the model may have dropped (e.g. "Brin" + "kmann").
                    # Only extend into adjacent O tokens that are part of
                    # the same word (no gap AND no whitespace/punctuation).
                    extended_indices = list(span.indices)
                    last_idx = extended_indices[-1]
                    while last_idx + 1 < n_tokens:
                        next_idx = last_idx + 1
                        ns, ne = offset_mapping[next_idx]
                        if ns == ne == 0:
                            break
                        # Must be immediately adjacent (no character gap)
                        prev_end = offset_mapping[last_idx][1]
                        if ns != prev_end:
                            break
                        # Only extend into O tokens
                        if subword_preds[next_idx] != "O":
                            break
                        # Stop if the next token starts with whitespace or punctuation
                        next_char = text[ns] if ns < len(text) else " "
                        if not next_char.isalnum():
                            break
                        extended_indices.append(next_idx)
                        last_idx = next_idx
                    for k, idx in enumerate(extended_indices):
                        bio_prefix = "B-" if k == 0 else "I-"
                        subword_preds[idx] = f"{bio_prefix}{entity_class}"
                        sources[idx] = "llm routed"
                else:
                    for idx in span.indices:
                        subword_preds[idx] = "O"
                        sources[idx] = "llm routed"
            except Exception as _llm_exc:
                logger.warning("LLM routing failed for span, keeping model prediction: %s", _llm_exc)

    # Phase 4: Regex demotion
    pred_ids_array = np.array([label2id.get(p, label2id.get("O", 0)) for p in subword_preds])
    offset_array = np.array(offset_mapping)
    old_before_demotion = pred_ids_array.copy()

    demoted_ids = regex_validator.validate_predictions(
        text=text,
        offset_mapping=offset_array,
        preds=pred_ids_array,
        id2label=id2label,
        label2id=label2id,
        entities_to_validate=REGEX_VALIDATABLE_ENTITIES,
    )

    for i in range(n_tokens):
        if demoted_ids[i] != old_before_demotion[i]:
            sources[i] = None  # demoted to O

    # Phase 5: Regex post-processing (final promotions)
    old_preds_post = demoted_ids.copy()
    corrected_ids = regex_validator.correct_predictions(
        text=text,
        offset_mapping=offset_array,
        preds=demoted_ids,
        id2label=id2label,
        label2id=label2id,
        correct_partial=True,
    )

    for i in range(n_tokens):
        if corrected_ids[i] != old_preds_post[i]:
            sources[i] = "regex override"
    subword_preds = [id2label.get(int(pid), "O") for pid in corrected_ids]

    # Update sources for final O tokens
    for i in range(n_tokens):
        if subword_preds[i] == "O":
            sources[i] = None

    # Assemble entities
    entities = _assemble_entities(text, subword_preds, conf_vals, offset_mapping, sources)

    # Phase 6: Regex overlay for entity classes NOT in the model vocabulary
    # (e.g. IBAN). These can't go through _apply_bio_sequence because B-IBAN
    # doesn't exist in label2id, so we inject them directly as entities.
    model_classes = {v.replace("B-", "").replace("I-", "") for v in id2label.values() if v != "O"}
    for hint_start, hint_end, hint_class in regex_validator.find_regex_hints(text):
        if hint_class in model_classes:
            continue  # handled by the normal token-level pipeline
        hint_text = text[hint_start:hint_end].strip()
        if not hint_text:
            continue
        # Check if this hint overlaps with an existing entity — if so, replace it
        entities = [
            e for e in entities
            if not (e["start"] < hint_end and e["end"] > hint_start)
        ]
        entities.append({
            "label": hint_class,
            "text": hint_text,
            "start": hint_start + (len(text[hint_start:hint_end]) - len(text[hint_start:hint_end].lstrip())),
            "end": hint_start + len(text[hint_start:hint_end].rstrip()),
            "confidence": 1.0,
            "source": "regex override",
        })
    # Re-sort by position
    entities.sort(key=lambda e: e["start"])

    # Phase 7: IBAN-context deconfliction
    # Remove false positive entities (especially CC) whose spans fall inside
    # IBAN-shaped sequences. IBAN has country code + check digits prefix that
    # makes the structural pattern unambiguous even without checksum validation.
    from src.inference.regex_validator import _IBAN_STRUCTURAL_RE
    iban_spans = [(m.start(), m.end()) for m in _IBAN_STRUCTURAL_RE.finditer(text)]
    if iban_spans:
        def _inside_iban(entity):
            """Return True if entity is fully contained inside an IBAN span."""
            for ib_start, ib_end in iban_spans:
                if entity["start"] >= ib_start and entity["end"] <= ib_end:
                    return True
            return False

        cleaned = []
        for e in entities:
            if e["label"] == "IBAN":
                cleaned.append(e)
            elif _inside_iban(e) and e["label"] in ("CREDITCARDNUMBER", "SOCIALNUM", "IDCARDNUM", "TELEPHONENUM"):
                continue  # suppress false positive inside IBAN
            else:
                cleaned.append(e)
        entities = cleaned

    # Phase 8: Email integrity enforcement
    # The model sometimes detects SURNAME/GIVENNAME inside email addresses,
    # fragmenting them (e.g. "adv.sofia.marchetti-dabrowska@..." → SURNAME
    # inside the email span). Find all regex-validated EMAIL matches and
    # replace any overlapping entities with a single complete EMAIL entity.
    email_compiled = regex_validator._compiled.get("EMAIL")
    if email_compiled:
        email_config = regex_validator.patterns["EMAIL"]
        for m in email_compiled.finditer(text):
            if email_config.validator_fn and not email_config.validator_fn(m.group()):
                continue
            em_start, em_end = m.start(), m.end()
            # Check if any existing entity overlaps with this email match
            overlapping = [e for e in entities if e["start"] < em_end and e["end"] > em_start]
            # If there's already a single EMAIL entity covering the full match, skip
            if len(overlapping) == 1 and overlapping[0]["label"] == "EMAIL" and \
               overlapping[0]["start"] <= em_start and overlapping[0]["end"] >= em_end:
                continue
            # If any overlapping entity is NOT a complete EMAIL, replace them all
            if overlapping:
                entities = [e for e in entities if not (e["start"] < em_end and e["end"] > em_start)]
                entities.append({
                    "label": "EMAIL",
                    "text": text[em_start:em_end],
                    "start": em_start,
                    "end": em_end,
                    "confidence": 1.0,
                    "source": "regex override",
                })
        entities.sort(key=lambda e: e["start"])

    # Redact text — fixed-length replacement prevents entity value inference from length
    REDACT_LEN = 5
    redacted = text
    for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
        replacement = "\u2588" * REDACT_LEN
        redacted = redacted[:entity["start"]] + replacement + redacted[entity["end"]:]

    return entities, redacted


def _format_output(text: str, entities: List[Dict], redacted: str) -> None:
    """Print formatted output with colors."""
    print(f"\n{Colors.BOLD}Input:{Colors.ENDC}")
    print(f'  "{text}"')

    print(f"\n{Colors.BOLD}Detected PII:{Colors.ENDC}")
    if not entities:
        print(f"  {Colors.DIM}(none){Colors.ENDC}")
    else:
        # Calculate column widths for alignment
        max_label = max(len(e["label"]) for e in entities)
        max_text = max(len(e["text"]) for e in entities)

        for e in entities:
            label = f"{Colors.OKCYAN}{e['label']:<{max_label}}{Colors.ENDC}"
            value = f'"{Colors.BOLD}{e["text"]}{Colors.ENDC}"'
            value_padded = f'"{e["text"]}"'
            padding = max_text - len(e["text"])
            source = e["source"]
            conf = e["confidence"]
            print(
                f"  {label} \u2192 {value}{' ' * padding}"
                f"    {Colors.DIM}[{source:<16s} conf: {conf:.3f}]{Colors.ENDC}"
            )

    print(f"\n{Colors.BOLD}Redacted output:{Colors.ENDC}")
    print(f'  "{redacted}"')
    print()


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="NerGuard PII detection with source tracking and redaction"
    )
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File to analyze")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to NER model")
    parser.add_argument("--llm", action="store_true", help="Enable LLM routing")
    parser.add_argument("--llm-source", type=str, default="openai", choices=["openai", "ollama"])
    parser.add_argument("--llm-model", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Determine input text
    if args.text:
        text = args.text
    elif args.file:
        max_bytes = 10 * 1024 * 1024  # 10 MB
        if os.path.getsize(args.file) > max_bytes:
            sys.exit(f"error: file too large (max 10 MB): {args.file}")
        with open(args.file, "r") as f:
            text = f.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    # Run pipeline
    print(f"{Colors.DIM}Loading model...{Colors.ENDC}", file=sys.stderr)
    entities, redacted = redact_pipeline(
        text=text,
        model_path=args.model_path,
        llm_routing=args.llm,
        llm_source=args.llm_source,
        llm_model=args.llm_model,
    )

    # Output
    if args.json:
        output = {
            "input": text,
            "entities": entities,
            "redacted": redacted,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        _format_output(text, entities, redacted)


if __name__ == "__main__":
    main()
