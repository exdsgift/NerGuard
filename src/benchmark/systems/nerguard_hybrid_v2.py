"""NerGuard Hybrid V2 — extends Hybrid with multi-layer regex + LLM pipeline.

Improvements over V1:
1. Regex pre-scan: promote high-precision regex matches (CC, SSN) BEFORE LLM routing
2. Regex demotion: demote entity predictions that don't match regex patterns (precision filter)
3. Regex-disagrees routing: force LLM routing when regex detects a pattern but model says O
4. Per-entity confidence gating: entity-specific LLM acceptance thresholds
5. Smarter O-span recovery: skip O-spans already handled by regex pre-scan
6. O-routing: uncertain O tokens (false negative candidates) are routed to LLM
7. Two-step PII gate: the LLM first decides if a span is PII (binary), then
   assigns a label from a constrained dictionary (dataset-aware).

The native_labels() stay at the model's 20 labels for fair evaluation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.benchmark.systems.base import DeferredPrediction, SystemPrediction
from src.benchmark.systems.nerguard_hybrid import NerGuardHybrid, MAX_LENGTH

logger = logging.getLogger(__name__)


class NerGuardHybridV2(NerGuardHybrid):
    """NerGuard Hybrid V2 with multi-layer regex, per-entity gating, and O-routing."""

    def __init__(
        self,
        model_path: str = "./models/mdeberta-pii-safe/final",
        device: str = "auto",
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        span_prompt_version: str = "V14_SPAN",
    ):
        super().__init__(model_path, device, llm_source, llm_model, span_prompt_version)
        self._target_labels: Optional[Set[str]] = None
        self._target_labels_str: Optional[str] = None
        self._o_prompt_template = None

    def name(self) -> str:
        return f"NerGuard Hybrid V2 ({self.llm_model})"

    def setup(self) -> None:
        super().setup()
        from src.inference.prompts import PROMPT_O_SPAN
        self._o_prompt_template = PROMPT_O_SPAN

    def set_dataset_labels(self, labels: Set[str]) -> None:
        """Set target dataset labels for the LLM prompt dictionary.

        Called by the runner before evaluation, passing the dataset's native labels.
        The LLM uses these as a constrained label dictionary when routing O spans.
        native_labels() is NOT expanded — evaluation stays on the model's 20 labels.
        """
        self._target_labels = labels
        self._target_labels_str = json.dumps(sorted(labels), indent=2)
        logger.info(f"V2: target labels set for LLM prompt ({len(labels)} types)")

    def predict_ner_only(
        self,
        sample_idx: int,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> DeferredPrediction:
        """Phase 1: NER + regex pre-scan + collect entity/O/regex-hint spans."""
        from src.inference.span_assembler import (
            assemble_entity_spans,
            assemble_uncertain_o_spans,
            EntitySpan,
        )

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

        # ── Change 1b: Regex PRE-SCAN ──────────────────────────────
        # Run regex correction BEFORE span assembly to promote CC/SSN tokens
        # that the model confidently predicted as O.
        label2id = {v: int(k) for k, v in self.id2label.items()}
        pred_ids_array = np.array(
            [label2id.get(p, label2id.get("O", 0)) for p in subword_preds]
        )
        offset_array = np.array(offset_mapping)
        old_preds_pre = pred_ids_array.copy()
        corrected_pre = self.regex_validator.correct_predictions(
            text=text,
            offset_mapping=offset_array,
            preds=pred_ids_array,
            id2label=self.id2label,
            label2id=label2id,
            correct_partial=True,
        )
        pre_scan_changes = int((corrected_pre != old_preds_pre).sum())
        self._routing_meta.setdefault("regex_prescan_promotions", 0)
        self._routing_meta["regex_prescan_promotions"] += pre_scan_changes

        # Update subword_preds from regex pre-scan corrections
        subword_preds = [self.id2label.get(int(pid), "O") for pid in corrected_pre]

        # ── Change 3: Collect regex hints ──────────────────────────
        regex_hints = self.regex_validator.find_regex_hints(text)

        # ── Standard entity spans (on pre-scan-corrected predictions) ──
        entity_spans = assemble_entity_spans(
            pred_labels=subword_preds,
            entropy_flat=entropy_vals,
            conf_flat=conf_vals,
            offset_flat=offset_mapping,
            entity_router=self.entity_router,
        )

        pending = []

        # Entity span routing (same as V1)
        for span in entity_spans:
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

        # ── Change 3: Regex-disagrees-with-model routing ───────────
        # For each regex hint, if model still predicts O for overlapping tokens,
        # create a synthetic span and force-route it to LLM.
        self._routing_meta.setdefault("regex_hint_routes", 0)
        for hint_start, hint_end, hint_entity in regex_hints:
            hint_tokens = []
            for idx, (sw_start, sw_end) in enumerate(offset_mapping):
                if sw_start == sw_end:
                    continue
                if sw_end <= hint_start or sw_start >= hint_end:
                    continue
                if subword_preds[idx] == "O":
                    hint_tokens.append(idx)

            if not hint_tokens:
                continue

            # Check these tokens aren't already in a pending span
            pending_indices = set()
            for sp, _ in pending:
                pending_indices.update(sp.indices)
            hint_tokens = [t for t in hint_tokens if t not in pending_indices]

            if not hint_tokens:
                continue

            span = EntitySpan(
                indices=hint_tokens,
                entity_class=hint_entity,
                is_uncertain=True,
                char_start=hint_start,
                char_end=hint_end,
            )
            prev_label = subword_preds[hint_tokens[0] - 1] if hint_tokens[0] > 0 else "O"
            pending.append((span, prev_label))
            self._routing_meta["regex_hint_routes"] += 1

        # ── Change 5 + O-span routing ─────────────────────────────
        from src.core.constants import DEFAULT_ENTROPY_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
        o_spans = assemble_uncertain_o_spans(
            pred_labels=subword_preds,
            entropy_flat=entropy_vals,
            conf_flat=conf_vals,
            offset_flat=offset_mapping,
            entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )

        # Skip O-spans that overlap with regex hints (already handled above)
        for span in o_spans:
            overlaps_regex = any(
                hint_start < span.char_end and hint_end > span.char_start
                for hint_start, hint_end, _ in regex_hints
            )
            if overlaps_regex:
                continue
            anchor_idx = span.indices[0]
            prev_label = subword_preds[anchor_idx - 1] if anchor_idx > 0 else "O"
            pending.append((span, prev_label))

        self._routing_meta.setdefault("o_spans_routed", 0)
        self._routing_meta["o_spans_routed"] += len(o_spans)

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
            conf_vals=conf_vals,
        )

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> "SystemPrediction":
        """Synchronous predict with all V2 features (used in --no-batch-llm mode).

        Runs V2's predict_ner_only (regex pre-scan, hints, span assembly), then
        routes each span synchronously, applies per-entity confidence gating,
        regex demotion, and regex post-processing.
        O-span routing is skipped here (async-only feature for OpenAI batch mode).
        """
        from src.core.constants import (
            ENTITY_CONFIDENCE_GATES,
            REGEX_VALIDATABLE_ENTITIES,
        )

        t0 = time.time()

        # Phase 1: V2 NER + regex pre-scan + span/hint/O-span assembly
        deferred = self.predict_ner_only(0, text, tokens, token_spans)
        subword_preds = list(deferred.subword_preds)
        label2id = {v: int(k) for k, v in self.id2label.items()}
        llm_calls = 0

        # Phase 2: sync routing — entity spans and regex-hint spans only
        for span, prev_label in deferred.pending_spans:
            if span.entity_class == "O":
                # O-span routing requires async; skip in sync mode
                continue
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

                if res.get("is_pii"):
                    entity_out = res.get("corrected_label", f"B-{span.entity_class}")
                    entity_class = entity_out.replace("B-", "").replace("I-", "")
                    for k, idx in enumerate(span.indices):
                        bio_prefix = "B-" if k == 0 else "I-"
                        subword_preds[idx] = f"{bio_prefix}{entity_class}"
                else:
                    # V2: per-entity confidence gating before accepting "not PII" verdict
                    entity_gate = ENTITY_CONFIDENCE_GATES.get(
                        span.entity_class,
                        ENTITY_CONFIDENCE_GATES.get("DEFAULT"),
                    )
                    anchor_idx = span.indices[0]
                    anchor_conf = (
                        deferred.conf_vals[anchor_idx]
                        if deferred.conf_vals and anchor_idx < len(deferred.conf_vals)
                        else 0.0
                    )
                    if entity_gate is not None and anchor_conf > entity_gate:
                        self._routing_meta["confidence_gates_applied"] += 1
                    else:
                        for idx in span.indices:
                            subword_preds[idx] = "O"
            except Exception:
                pass

        self._routing_meta["llm_calls"] += llm_calls

        # Phase 3: V2 regex demotion — demote entities that fail regex validation
        pred_ids_array = np.array(
            [label2id.get(p, label2id.get("O", 0)) for p in subword_preds]
        )
        offset_array = np.array(deferred.offset_mapping)
        old_before_demotion = pred_ids_array.copy()
        demoted_ids = self.regex_validator.validate_predictions(
            text=text,
            offset_mapping=offset_array,
            preds=pred_ids_array,
            id2label=self.id2label,
            label2id=label2id,
            entities_to_validate=REGEX_VALIDATABLE_ENTITIES,
        )
        self._routing_meta["regex_demotions"] += int(
            (demoted_ids != old_before_demotion).sum()
        )

        # Phase 4: regex post-processing (O → Entity promotions)
        old_preds = demoted_ids.copy()
        corrected_ids = self.regex_validator.correct_predictions(
            text=text,
            offset_mapping=offset_array,
            preds=demoted_ids,
            id2label=self.id2label,
            label2id=label2id,
            correct_partial=True,
        )
        self._routing_meta["regex_promotions"] += int(
            (corrected_ids != old_preds).sum()
        )
        subword_preds = [self.id2label.get(int(pid), "O") for pid in corrected_ids]

        word_labels = self._align_to_word_tokens(
            subword_preds, deferred.offset_mapping, tokens, token_spans
        )

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=word_labels, latency_ms=latency_ms)

    async def resolve_routing_batch(
        self,
        deferred_preds: List[DeferredPrediction],
        max_concurrent: int = 50,
    ) -> List[SystemPrediction]:
        """Phase 2+3: Batch route, apply corrections, regex demotion, post-processing."""
        from src.core.constants import (
            ENTITY_CONFIDENCE_GATES,
            REGEX_VALIDATABLE_ENTITIES,
        )

        semaphore = asyncio.Semaphore(max_concurrent)
        label2id = {v: int(k) for k, v in self.id2label.items()}

        tasks = []
        for d_idx, d in enumerate(deferred_preds):
            for s_idx, (span, prev_label) in enumerate(d.pending_spans):
                tasks.append((d_idx, s_idx, span, prev_label, d.text))

        n_entity = sum(1 for _, _, s, _, _ in tasks if s.entity_class != "O")
        n_o = sum(1 for _, _, s, _, _ in tasks if s.entity_class == "O")
        logger.info(
            f"Batch routing V2: {len(tasks)} LLM calls "
            f"({n_entity} entity + {n_o} O-span) across {len(deferred_preds)} samples"
        )

        async def _route_one(span, prev_label, text):
            async with semaphore:
                if span.entity_class == "O":
                    return await self._route_o_span_async(span, text)
                else:
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

        llm_results = {}
        for i, (d_idx, s_idx, _, _, _) in enumerate(tasks):
            res = results[i]
            if isinstance(res, Exception):
                logger.debug(f"LLM call failed: {res}")
                res = None
            llm_results[(d_idx, s_idx)] = res

        self._routing_meta["llm_calls"] += len(tasks)

        # Apply corrections
        self._routing_meta.setdefault("confidence_gates_applied", 0)
        self._routing_meta.setdefault("regex_demotions", 0)
        final_preds = []
        for d_idx, d in enumerate(deferred_preds):
            subword_preds = list(d.subword_preds)

            for s_idx, (span, prev_label) in enumerate(d.pending_spans):
                res = llm_results.get((d_idx, s_idx))
                if res is None:
                    continue

                if span.entity_class == "O":
                    self._apply_o_span_correction(
                        res, span, subword_preds, label2id
                    )
                else:
                    # ── Change 4: Per-entity confidence gating ─────
                    if res.get("is_pii"):
                        entity_out = res.get("corrected_label", f"B-{span.entity_class}")
                        entity_class = entity_out.replace("B-", "").replace("I-", "")
                        for k, idx in enumerate(span.indices):
                            bio_prefix = "B-" if k == 0 else "I-"
                            subword_preds[idx] = f"{bio_prefix}{entity_class}"
                    else:
                        # LLM says "not PII" — apply per-entity confidence gate
                        entity_gate = ENTITY_CONFIDENCE_GATES.get(
                            span.entity_class,
                            ENTITY_CONFIDENCE_GATES.get("DEFAULT"),
                        )
                        anchor_idx = span.indices[0]
                        anchor_conf = (
                            d.conf_vals[anchor_idx]
                            if d.conf_vals and anchor_idx < len(d.conf_vals)
                            else 0.0
                        )
                        if entity_gate is not None and anchor_conf > entity_gate:
                            # Gate: model confidence exceeds threshold, reject LLM's O verdict
                            self._routing_meta["confidence_gates_applied"] += 1
                        else:
                            # Accept LLM verdict — demote to O
                            for idx in span.indices:
                                subword_preds[idx] = "O"

            # ── Change 2: Regex demotion filter ────────────────────
            pred_ids_array = np.array(
                [label2id.get(p, label2id.get("O", 0)) for p in subword_preds]
            )
            offset_array = np.array(d.offset_mapping)
            old_before_demotion = pred_ids_array.copy()
            demoted_ids = self.regex_validator.validate_predictions(
                text=d.text,
                offset_mapping=offset_array,
                preds=pred_ids_array,
                id2label=self.id2label,
                label2id=label2id,
                entities_to_validate=REGEX_VALIDATABLE_ENTITIES,
            )
            self._routing_meta["regex_demotions"] += int(
                (demoted_ids != old_before_demotion).sum()
            )

            # ── Regex post-processing (O → Entity promotions) ─────
            old_preds = demoted_ids.copy()
            corrected_ids = self.regex_validator.correct_predictions(
                text=d.text,
                offset_mapping=offset_array,
                preds=demoted_ids,
                id2label=self.id2label,
                label2id=label2id,
                correct_partial=True,
            )
            self._routing_meta["regex_promotions"] += int(
                (corrected_ids != old_preds).sum()
            )

            # Rebuild subword preds, preserving extended labels from O-routing.
            # Extended labels (e.g. "B-first_name") come from LLM O-span routing
            # and aren't in the model vocabulary (label2id). These must be kept
            # as-is. All model-native labels use the corrected_ids values (which
            # incorporate regex demotion + post-processing).
            final_subword = []
            for idx, pid in enumerate(corrected_ids):
                current = subword_preds[idx]
                if current not in label2id and current != "O":
                    final_subword.append(current)
                else:
                    final_subword.append(self.id2label.get(int(pid), "O"))

            word_labels = self._align_to_word_tokens(
                final_subword, d.offset_mapping, d.tokens, d.token_spans
            )

            final_preds.append(SystemPrediction(
                labels=word_labels,
                latency_ms=d.ner_latency_ms,
            ))

        return final_preds

    async def _route_o_span_async(self, span, text: str) -> Dict:
        """Route an uncertain O span with the two-step PII gate prompt."""
        context = self.router._extract_context(text, span.char_start, span.char_end)
        clean_span = text[span.char_start:span.char_end].strip()

        from src.inference.prompts import ENTITY_CLASSES_STR
        target_str = self._target_labels_str or ENTITY_CLASSES_STR

        try:
            prompt = self._o_prompt_template.format(
                context=context,
                span_text=clean_span,
                token_count=len(span.indices),
                target_labels_str=target_str,
            )
        except KeyError as e:
            return {"is_pii": False, "corrected_label": "O", "reasoning": f"Prompt error: {e}"}

        try:
            raw = await self.router._call_openai_async(prompt)
            is_pii = raw.get("is_pii", False)
            entity_class = raw.get("entity_class", "O").strip()

            if not is_pii or entity_class == "O" or not entity_class:
                return {"is_pii": False, "corrected_label": "O", "reasoning": raw.get("reasoning", "")}
            return {
                "is_pii": True,
                "corrected_label": f"B-{entity_class}",
                "reasoning": raw.get("reasoning", ""),
            }
        except Exception as e:
            logger.debug(f"O-span LLM failed: {e}")
            return {"is_pii": False, "corrected_label": "O", "reasoning": str(e)}

    def _apply_o_span_correction(
        self,
        res: Dict,
        span,
        subword_preds: List[str],
        label2id: Dict,
    ) -> None:
        """Apply LLM correction for an O-span (potentially with extended labels)."""
        if not res.get("is_pii"):
            return

        entity_out = res.get("corrected_label", "O")
        entity_class = entity_out.replace("B-", "").replace("I-", "")

        if entity_class == "O":
            return

        self._routing_meta.setdefault("o_spans_promoted", 0)
        self._routing_meta["o_spans_promoted"] += 1

        for k, idx in enumerate(span.indices):
            bio_prefix = "B-" if k == 0 else "I-"
            subword_preds[idx] = f"{bio_prefix}{entity_class}"

    @staticmethod
    def _empty_routing_meta() -> Dict:
        return {
            "llm_calls": 0,
            "regex_skips": 0,
            "regex_promotions": 0,
            "regex_prescan_promotions": 0,
            "regex_demotions": 0,
            "regex_hint_routes": 0,
            "confidence_gates_applied": 0,
            "o_spans_routed": 0,
            "o_spans_promoted": 0,
        }
