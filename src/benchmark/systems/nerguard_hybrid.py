"""NerGuard Hybrid system wrapper — mDeBERTa + OpenAI LLM routing + regex validation."""

import logging
import os
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.benchmark.systems.base import DeferredPrediction, SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

MAX_LENGTH = 512


class NerGuardHybrid(SystemWrapper):
    def __init__(
        self,
        model_path: str = "./models/mdeberta-pii-safe/final",
        device: str = "auto",
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        span_prompt_version: str = "V14_SPAN",
    ):
        self.model_path = model_path
        self.device_str = device
        self.llm_source = llm_source
        self.llm_model = llm_model
        self.span_prompt_version = span_prompt_version
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.router = None
        self.entity_router = None
        self.regex_validator = None
        self._native_labels = None
        self._routing_meta = self._empty_routing_meta()

    def name(self) -> str:
        return f"NerGuard Hybrid ({self.llm_model})"

    def native_labels(self) -> Set[str]:
        if self._native_labels is not None:
            return self._native_labels
        return {
            "AGE", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATE",
            "DRIVERLICENSENUM", "EMAIL", "GENDER", "GIVENNAME", "IDCARDNUM",
            "PASSPORTNUM", "SEX", "SOCIALNUM", "STREET", "SURNAME",
            "TAXNUM", "TELEPHONENUM", "TIME", "TITLE", "ZIPCODE",
        }

    def setup(self) -> None:
        from src.inference.entity_router import EntitySpecificRouter
        from src.inference.llm_router import LLMRouter
        from src.inference.regex_validator import RegexValidator
        from src.tasks.pii.config import get_pii_route_config

        logger.info(f"Loading NerGuard Hybrid (LLM: {self.llm_source}/{self.llm_model})")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        self._native_labels = set()
        for label in self.id2label.values():
            if label != "O":
                self._native_labels.add(label.replace("B-", "").replace("I-", ""))

        # Initialize routing components
        self.router = LLMRouter(
            source=self.llm_source,
            model=self.llm_model,
            ollama_model=self.llm_model,
            span_prompt_version=self.span_prompt_version,
        )
        route_config = get_pii_route_config()
        self.entity_router = EntitySpecificRouter.from_config(route_config)
        self.regex_validator = RegexValidator()

        self._routing_meta = self._empty_routing_meta()
        logger.info(f"NerGuard Hybrid ready on {self.device}")

    def calibrate_thresholds(self, samples: list) -> None:
        """Calibrate routing thresholds on held-out samples."""
        from src.inference.entity_router import EntitySpecificRouter
        from src.optimization.calibrator import ThresholdCalibrator
        from src.tasks.pii.config import get_pii_route_config

        calibrator = ThresholdCalibrator()
        result, calibrated_config = calibrator.calibrate(
            model=self.model,
            tokenizer=self.tokenizer,
            id2label=self.id2label,
            device=self.device,
            samples=samples,
            base_config=get_pii_route_config(),
        )
        self.entity_router = EntitySpecificRouter.from_config(calibrated_config)
        logger.info(f"Calibrated router: {self.entity_router}")

    @staticmethod
    def _empty_routing_meta() -> Dict:
        return {
            "llm_calls": 0,
            "regex_skips": 0,
            "regex_promotions": 0,
        }

    def get_routing_metadata(self) -> Dict:
        """Return accumulated routing statistics and reset counters."""
        meta = self._routing_meta.copy()
        if self.entity_router:
            meta["entity_router_stats"] = self.entity_router.get_stats()
        self._routing_meta = self._empty_routing_meta()
        return meta

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.router = None
        self.entity_router = None
        self.regex_validator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NerGuard Hybrid teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        from src.inference.span_assembler import assemble_entity_spans

        t0 = time.time()

        # Subword tokenization
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[0]

        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
        conf, pred_ids_tensor = torch.max(probs, dim=-1)

        pred_ids = pred_ids_tensor.cpu().tolist()
        entropy_vals = entropy.cpu().tolist()
        conf_vals = conf.cpu().tolist()

        # Build initial subword predictions
        subword_preds = [self.id2label.get(pid, "O") for pid in pred_ids]

        # Span-level LLM routing (anchor propagation)
        # Groups B-X / I-X tokens into spans; routing decision based on B- anchor only
        spans = assemble_entity_spans(
            pred_labels=subword_preds,
            entropy_flat=entropy_vals,
            conf_flat=conf_vals,
            offset_flat=offset_mapping,
            entity_router=self.entity_router,
        )

        label2id = {v: int(k) for k, v in self.id2label.items()}
        llm_calls = 0

        for span in spans:
            if not span.is_uncertain:
                continue

            # Skip special tokens (offset 0,0)
            anchor_idx = span.indices[0]
            sw_start, sw_end = offset_mapping[anchor_idx]
            if sw_start == sw_end == 0:
                continue

            # Check regex skip: if regex confirms the entity, skip LLM
            if self.regex_validator.can_skip_llm(
                entity_class=span.entity_class,
                text=text,
                char_start=span.char_start,
                char_end=span.char_end,
            ):
                self._routing_meta["regex_skips"] += 1
                continue

            # Get previous label for context
            prev_label = subword_preds[anchor_idx - 1] if anchor_idx > 0 else "O"

            # Route entire span to LLM as a single call
            try:
                res = self.router.disambiguate_span(
                    span_text=text[span.char_start:span.char_end],
                    token_count=len(span.indices),
                    full_text=text,
                    span_start=span.char_start,
                    span_end=span.char_end,
                    current_pred=span.entity_class,
                    prev_label=prev_label,
                )
                llm_calls += 1

                # Apply correction to all tokens in the span
                if res.get("is_pii"):
                    entity_out = res.get("corrected_label", f"B-{span.entity_class}")
                    # Extract entity class from corrected label
                    entity_class = entity_out.replace("B-", "").replace("I-", "")
                    for k, idx in enumerate(span.indices):
                        bio_prefix = "B-" if k == 0 else "I-"
                        subword_preds[idx] = f"{bio_prefix}{entity_class}"
                else:
                    for idx in span.indices:
                        subword_preds[idx] = "O"
            except Exception:
                pass

        self._routing_meta["llm_calls"] += llm_calls

        # Apply regex post-processing (O → Entity promotions)
        pred_ids_array = np.array(
            [label2id.get(p, label2id.get("O", 0)) for p in subword_preds]
        )
        offset_array = np.array(offset_mapping)
        old_preds = pred_ids_array.copy()
        corrected_ids = self.regex_validator.correct_predictions(
            text=text,
            offset_mapping=offset_array,
            preds=pred_ids_array,
            id2label=self.id2label,
            label2id=label2id,
            correct_partial=True,
        )
        self._routing_meta["regex_promotions"] += int((corrected_ids != old_preds).sum())
        subword_preds = [self.id2label.get(int(pid), "O") for pid in corrected_ids]

        # Map subword predictions to word-level tokens
        word_labels = self._align_to_word_tokens(
            subword_preds, offset_mapping, tokens, token_spans
        )

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=word_labels, latency_ms=latency_ms)

    def predict_ner_only(
        self,
        sample_idx: int,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> DeferredPrediction:
        """Phase 1: NER inference only. Returns base predictions + pending LLM spans."""
        from src.inference.span_assembler import assemble_entity_spans

        t0 = time.time()

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[0]

        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
        conf, pred_ids_tensor = torch.max(probs, dim=-1)

        pred_ids = pred_ids_tensor.cpu().tolist()
        entropy_vals = entropy.cpu().tolist()
        conf_vals = conf.cpu().tolist()

        subword_preds = [self.id2label.get(pid, "O") for pid in pred_ids]

        spans = assemble_entity_spans(
            pred_labels=subword_preds,
            entropy_flat=entropy_vals,
            conf_flat=conf_vals,
            offset_flat=offset_mapping,
            entity_router=self.entity_router,
        )

        # Filter to spans that need LLM routing (uncertain + not regex-skippable)
        pending = []
        for span in spans:
            if not span.is_uncertain:
                continue
            anchor_idx = span.indices[0]
            sw_start, sw_end = offset_mapping[anchor_idx]
            if sw_start == sw_end == 0:
                continue
            if self.regex_validator.can_skip_llm(
                entity_class=span.entity_class,
                text=text,
                char_start=span.char_start,
                char_end=span.char_end,
            ):
                self._routing_meta["regex_skips"] += 1
                continue
            prev_label = subword_preds[anchor_idx - 1] if anchor_idx > 0 else "O"
            pending.append((span, prev_label))

        ner_latency_ms = (time.time() - t0) * 1000

        return DeferredPrediction(
            sample_idx=sample_idx,
            text=text,
            tokens=tokens,
            token_spans=token_spans,
            subword_preds=subword_preds,
            offset_mapping=offset_mapping,
            pending_spans=pending,
            ner_latency_ms=ner_latency_ms,
        )

    async def resolve_routing_batch(
        self,
        deferred_preds: List[DeferredPrediction],
        max_concurrent: int = 50,
    ) -> List[SystemPrediction]:
        """Phase 2+3: Batch LLM calls async, apply corrections, return final predictions."""
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)
        label2id = {v: int(k) for k, v in self.id2label.items()}

        # Collect all LLM tasks: (deferred_idx, span_idx_in_pending, span, prev_label)
        tasks = []
        for d_idx, d in enumerate(deferred_preds):
            for s_idx, (span, prev_label) in enumerate(d.pending_spans):
                tasks.append((d_idx, s_idx, span, prev_label, d.text))

        logger.info(f"Batch routing: {len(tasks)} LLM calls across {len(deferred_preds)} samples (concurrency={max_concurrent})")

        # Fire all LLM calls concurrently
        async def _route_one(span, prev_label, text):
            async with semaphore:
                return await self.router.disambiguate_span_async(
                    span_text=text[span.char_start:span.char_end],
                    token_count=len(span.indices),
                    full_text=text,
                    span_start=span.char_start,
                    span_end=span.char_end,
                    current_pred=span.entity_class,
                    prev_label=prev_label,
                )

        coroutines = [_route_one(span, prev, text) for _, _, span, prev, text in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Map results back to deferred predictions
        llm_results = {}  # (d_idx, s_idx) -> result
        for i, (d_idx, s_idx, _, _, _) in enumerate(tasks):
            res = results[i]
            if isinstance(res, Exception):
                logger.debug(f"LLM call failed: {res}")
                res = None
            llm_results[(d_idx, s_idx)] = res

        self._routing_meta["llm_calls"] += len(tasks)

        # Apply corrections and finalize
        final_preds = []
        for d_idx, d in enumerate(deferred_preds):
            subword_preds = list(d.subword_preds)  # copy

            for s_idx, (span, prev_label) in enumerate(d.pending_spans):
                res = llm_results.get((d_idx, s_idx))
                if res is None:
                    continue
                if res.get("is_pii"):
                    entity_out = res.get("corrected_label", f"B-{span.entity_class}")
                    entity_class = entity_out.replace("B-", "").replace("I-", "")
                    for k, idx in enumerate(span.indices):
                        bio_prefix = "B-" if k == 0 else "I-"
                        subword_preds[idx] = f"{bio_prefix}{entity_class}"
                else:
                    for idx in span.indices:
                        subword_preds[idx] = "O"

            # Regex post-processing
            pred_ids_array = np.array(
                [label2id.get(p, label2id.get("O", 0)) for p in subword_preds]
            )
            offset_array = np.array(d.offset_mapping)
            old_preds = pred_ids_array.copy()
            corrected_ids = self.regex_validator.correct_predictions(
                text=d.text,
                offset_mapping=offset_array,
                preds=pred_ids_array,
                id2label=self.id2label,
                label2id=label2id,
                correct_partial=True,
            )
            self._routing_meta["regex_promotions"] += int((corrected_ids != old_preds).sum())
            subword_preds = [self.id2label.get(int(pid), "O") for pid in corrected_ids]

            word_labels = self._align_to_word_tokens(
                subword_preds, d.offset_mapping, d.tokens, d.token_spans
            )

            final_preds.append(SystemPrediction(
                labels=word_labels,
                latency_ms=d.ner_latency_ms,
            ))

        return final_preds

    def _align_to_word_tokens(
        self,
        subword_preds: List[str],
        subword_offsets: List[Tuple[int, int]],
        word_tokens: List[str],
        word_spans: List[Tuple[int, int]],
    ) -> List[str]:
        """Map subword-level string predictions to word-level BIO labels."""
        word_labels = ["O"] * len(word_tokens)

        for word_idx, (w_start, w_end) in enumerate(word_spans):
            for sw_idx, (sw_start, sw_end) in enumerate(subword_offsets):
                if sw_start == sw_end == 0:
                    continue
                if sw_end <= w_start or sw_start >= w_end:
                    continue
                pred = subword_preds[sw_idx]
                if pred != "O":
                    word_labels[word_idx] = pred
                    break

        # BIO repair
        for i in range(len(word_labels)):
            if word_labels[i].startswith("I-"):
                entity_type = word_labels[i][2:]
                if i == 0:
                    word_labels[i] = f"B-{entity_type}"
                else:
                    prev = word_labels[i - 1]
                    prev_type = prev[2:] if prev.startswith(("B-", "I-")) else None
                    if prev_type != entity_type:
                        word_labels[i] = f"B-{entity_type}"

        return word_labels
