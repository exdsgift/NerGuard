"""
Object-oriented core inference pipeline for NerGuard.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any

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
    ENTITY_CONFIDENCE_GATES,
)
from src.core.model_loader import load_model_and_tokenizer, get_device
from src.inference.regex_validator import RegexValidator
from src.inference.entity_router import EntitySpecificRouter
from src.inference.span_assembler import assemble_entity_spans, assemble_uncertain_o_spans, assemble_entities
from src.inference.llm_router import LLMRouter

logger = logging.getLogger(__name__)


class HybridPipeline:
    """Unified core pipeline with async batching support."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        llm_routing: bool = False,
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        device: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.device = get_device() if device is None else torch.device(device)
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, device=str(self.device), eval_mode=True)
        self.id2label = self.model.config.id2label
        self.label2id = {v: int(k) for k, v in self.id2label.items()}
        
        self.llm_routing = llm_routing
        self.regex_validator = RegexValidator()
        
        self.entity_router = None
        self.router = None
        if llm_routing:
            self.entity_router = EntitySpecificRouter(
                entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                routable_entities=ROUTABLE_ENTITIES,
                blocked_entities=BLOCKED_ENTITIES,
                routable_i_entities=ROUTABLE_I_ENTITIES,
                entity_thresholds=ENTITY_THRESHOLDS,
            )
            self.router = LLMRouter(source=llm_source, model=llm_model, ollama_model=llm_model, api_key=api_key)

    def _run_single_pass(self, input_ids, offsets):
        full_ids = [self.tokenizer.cls_token_id] + list(input_ids) + [self.tokenizer.sep_token_id]
        full_offsets = [(0, 0)] + list(offsets) + [(0, 0)]

        input_tensor = torch.tensor([full_ids], device=self.device)
        attention_mask = torch.ones_like(input_tensor)

        with torch.no_grad():
            logits = self.model(input_tensor, attention_mask=attention_mask).logits[0]

        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
        conf, pred_ids_tensor = torch.max(probs, dim=-1)

        return (
            [self.id2label.get(pid, "O") for pid in pred_ids_tensor.cpu().tolist()],
            entropy.cpu().tolist(),
            conf.cpu().tolist(),
            full_offsets
        )

    def _run_base_model(self, text: str):
        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
        all_input_ids = encoding["input_ids"]
        all_offsets = encoding["offset_mapping"]
        total_tokens = len(all_input_ids)
        max_content = MAX_CONTEXT_LENGTH - 2

        if total_tokens <= max_content:
            return self._run_single_pass(all_input_ids, all_offsets)

        best_preds = ["O"] * total_tokens
        best_entropy = [0.0] * total_tokens
        best_conf = [0.0] * total_tokens

        for chunk_start in range(0, total_tokens, STRIDE):
            chunk_end = min(chunk_start + max_content, total_tokens)
            chunk_ids = all_input_ids[chunk_start:chunk_end]
            chunk_offsets = all_offsets[chunk_start:chunk_end]

            tokens_to_use = len(chunk_ids) if chunk_end >= total_tokens else min(len(chunk_ids), STRIDE)
            preds, ent_vals, conf_vals, _ = self._run_single_pass(chunk_ids[:tokens_to_use], chunk_offsets[:tokens_to_use])

            for j in range(1, len(preds) - 1):
                global_idx = chunk_start + (j - 1)
                if global_idx < total_tokens and conf_vals[j] > best_conf[global_idx]:
                    best_preds[global_idx] = preds[j]
                    best_entropy[global_idx] = ent_vals[j]
                    best_conf[global_idx] = conf_vals[j]

        return (
            ["O"] + best_preds + ["O"],
            [0.0] + best_entropy + [0.0],
            [0.0] + best_conf + [0.0],
            [(0, 0)] + all_offsets + [(0, 0)]
        )

    def _phase1_and_2(self, text: str):
        subword_preds, entropy_vals, conf_vals, offset_mapping = self._run_base_model(text)
        n_tokens = len(subword_preds)
        sources = ["base model" if p != "O" else None for p in subword_preds]

        # Regex prescan
        pred_ids_array = np.array([self.label2id.get(p, self.label2id.get("O", 0)) for p in subword_preds])
        offset_array = np.array(offset_mapping)
        old_preds = pred_ids_array.copy()

        prescan_ids = self.regex_validator.correct_predictions(
            text=text, offset_mapping=offset_array, preds=pred_ids_array,
            id2label=self.id2label, label2id=self.label2id, correct_partial=True
        )

        for i in range(n_tokens):
            if prescan_ids[i] != old_preds[i]:
                sources[i] = "regex override"
        subword_preds = [self.id2label.get(int(pid), "O") for pid in prescan_ids]

        # Regex hints overlay
        regex_hints = self.regex_validator.find_regex_hints(text)
        for i in range(n_tokens):
            if sources[i] == "base model":
                cs, ce = offset_mapping[i]
                if cs == ce == 0: continue
                for hint_start, hint_end, _ in regex_hints:
                    if cs < hint_end and ce > hint_start:
                        sources[i] = "base + regex"
                        break

        pending_spans = []
        if self.llm_routing:
            entity_spans = assemble_entity_spans(subword_preds, entropy_vals, conf_vals, offset_mapping, self.entity_router)
            o_spans = assemble_uncertain_o_spans(
                subword_preds, entropy_vals, conf_vals, offset_mapping,
                DEFAULT_ENTROPY_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
            )

            for span in entity_spans:
                if not span.is_uncertain: continue
                anchor_idx = span.indices[0]
                if offset_mapping[anchor_idx][0] == offset_mapping[anchor_idx][1] == 0: continue
                if self.regex_validator.can_skip_llm(span.entity_class, text, span.char_start, span.char_end):
                    for idx in span.indices:
                        if sources[idx] and sources[idx] != "regex override":
                            sources[idx] = "base + regex"
                    continue
                pending_spans.append(span)

            for span in o_spans:
                overlaps = any(h_start < span.char_end and h_end > span.char_start for h_start, h_end, _ in regex_hints)
                if not overlaps:
                    pending_spans.append(span)

        return subword_preds, offset_mapping, sources, pending_spans, conf_vals

    def _apply_llm_corrections(self, text, subword_preds, offset_mapping, sources, conf_vals, span, res):
        if res.get("is_pii"):
            entity_out = res.get("corrected_label", f"B-{span.entity_class}")
            entity_class = entity_out.replace("B-", "").replace("I-", "")
            if entity_class == "O": return
            extended_indices = list(span.indices)
            last_idx = extended_indices[-1]
            n_tokens = len(subword_preds)
            while last_idx + 1 < n_tokens:
                next_idx = last_idx + 1
                ns, ne = offset_mapping[next_idx]
                if ns == ne == 0 or ns != offset_mapping[last_idx][1] or subword_preds[next_idx] != "O": break
                if not text[ns].isalnum(): break
                extended_indices.append(next_idx)
                last_idx = next_idx
            for k, idx in enumerate(extended_indices):
                bio_prefix = "B-" if k == 0 else "I-"
                subword_preds[idx] = f"{bio_prefix}{entity_class}"
                sources[idx] = "llm routed"
        else:
            if span.entity_class != "O":
                entity_gate = ENTITY_CONFIDENCE_GATES.get(span.entity_class, ENTITY_CONFIDENCE_GATES.get("DEFAULT"))
                anchor_conf = conf_vals[span.indices[0]]
                if entity_gate is not None and anchor_conf > entity_gate:
                    return # Gate active
                for idx in span.indices:
                    subword_preds[idx] = "O"
                    sources[idx] = "llm routed"

    def _phase4_to_8(self, text, subword_preds, offset_mapping, sources, conf_vals):
        n_tokens = len(subword_preds)
        pred_ids_array = np.array([self.label2id.get(p, self.label2id.get("O", 0)) for p in subword_preds])
        offset_array = np.array(offset_mapping)
        old_before_demotion = pred_ids_array.copy()

        demoted_ids = self.regex_validator.validate_predictions(
            text=text, offset_mapping=offset_array, preds=pred_ids_array,
            id2label=self.id2label, label2id=self.label2id, entities_to_validate=REGEX_VALIDATABLE_ENTITIES
        )
        for i in range(n_tokens):
            if demoted_ids[i] != old_before_demotion[i]: sources[i] = None

        corrected_ids = self.regex_validator.correct_predictions(
            text=text, offset_mapping=offset_array, preds=demoted_ids,
            id2label=self.id2label, label2id=self.label2id, correct_partial=True
        )
        for i in range(n_tokens):
            if corrected_ids[i] != demoted_ids[i]: sources[i] = "regex override"
        subword_preds = [self.id2label.get(int(pid), "O") for pid in corrected_ids]

        entities = assemble_entities(text, subword_preds, conf_vals, offset_mapping, sources)

        # Extra regex
        model_classes = {v.replace("B-", "").replace("I-", "") for v in self.id2label.values() if v != "O"}
        for h_start, h_end, h_class in self.regex_validator.find_regex_hints(text):
            if h_class in model_classes: continue
            h_text = text[h_start:h_end].strip()
            if not h_text: continue
            entities = [e for e in entities if not (e["start"] < h_end and e["end"] > h_start)]
            entities.append({
                "label": h_class, "text": h_text, "start": h_start + (len(text[h_start:h_end]) - len(text[h_start:h_end].lstrip())),
                "end": h_start + len(text[h_start:h_end].rstrip()), "confidence": 1.0, "source": "regex override"
            })
        entities.sort(key=lambda e: e["start"])

        # IBAN
        from src.inference.regex_validator import _IBAN_STRUCTURAL_RE
        iban_spans = [(m.start(), m.end()) for m in _IBAN_STRUCTURAL_RE.finditer(text)]
        if iban_spans:
            entities = [e for e in entities if e["label"] == "IBAN" or not any(e["start"] >= s and e["end"] <= end for s, end in iban_spans)]

        # Email
        email_compiled = self.regex_validator._compiled.get("EMAIL")
        if email_compiled:
            for m in email_compiled.finditer(text):
                em_start, em_end = m.start(), m.end()
                overlapping = [e for e in entities if e["start"] < em_end and e["end"] > em_start]
                if overlapping and not (len(overlapping) == 1 and overlapping[0]["label"] == "EMAIL"):
                    entities = [e for e in entities if not (e["start"] < em_end and e["end"] > em_start)]
                    entities.append({"label": "EMAIL", "text": text[em_start:em_end], "start": em_start, "end": em_end, "confidence": 1.0, "source": "regex override"})
            entities.sort(key=lambda e: e["start"])

        # Redact
        REDACT_LEN = 5
        redacted = text
        for entity in sorted(entities, key=lambda e: e["start"], reverse=True):
            redacted = redacted[:entity["start"]] + "\u2588" * REDACT_LEN + redacted[entity["end"]:]

        return entities, redacted, subword_preds, offset_mapping

    def process_text(self, text: str) -> Tuple[List[Dict], str]:
        """Synchronous full pipeline process."""
        subword_preds, offset_mapping, sources, pending_spans, conf_vals = self._phase1_and_2(text)
        
        if self.llm_routing and pending_spans:
            for span in pending_spans:
                anchor_idx = span.indices[0]
                prev_label = subword_preds[anchor_idx - 1] if anchor_idx > 0 else "O"
                try:
                    if span.entity_class == "O":
                        context = self.router._extract_context(text, span.char_start, span.char_end)
                        clean_span = text[span.char_start:span.char_end].strip()
                        from src.inference.prompts import PROMPT_O_SPAN, ENTITY_CLASSES_STR
                        prompt = PROMPT_O_SPAN.format(context=context, span_text=clean_span, token_count=len(span.indices), target_labels_str=ENTITY_CLASSES_STR)
                        raw = self.router.call([{"role": "system", "content": "You are a PII expert."}, {"role": "user", "content": prompt}])
                        res = {"is_pii": raw.get("is_pii", False), "corrected_label": f"B-{raw.get('entity_class', 'O').strip()}"}
                    else:
                        res = self.router.disambiguate_span(
                            span_text=text[span.char_start:span.char_end], token_count=len(span.indices),
                            full_text=text, span_start=span.char_start, span_end=span.char_end,
                            current_pred=span.entity_class, prev_label=prev_label
                        )
                    self._apply_llm_corrections(text, subword_preds, offset_mapping, sources, conf_vals, span, res)
                except Exception as e:
                    logger.warning(f"LLM routing failed: {e}")

        entities, redacted, _, _ = self._phase4_to_8(text, subword_preds, offset_mapping, sources, conf_vals)
        return entities, redacted

    async def process_batch_async(self, texts: List[str], max_concurrent: int = 50) -> List[Tuple[List[Dict], str]]:
        """Asynchronous batch processing with V2 async routing logic."""
        if not texts: return []
        
        batch_state = []
        for text in texts:
            batch_state.append(self._phase1_and_2(text))
            
        if not self.llm_routing:
            results = []
            for i, text in enumerate(texts):
                sub, off, src, _, conf = batch_state[i]
                ent, red, _, _ = self._phase4_to_8(text, sub, off, src, conf)
                results.append((ent, red))
            return results

        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []
        for d_idx, state in enumerate(batch_state):
            subword_preds, _, _, pending_spans, _ = state
            for s_idx, span in enumerate(pending_spans):
                anchor_idx = span.indices[0]
                prev_label = subword_preds[anchor_idx - 1] if anchor_idx > 0 else "O"
                tasks.append((d_idx, s_idx, span, prev_label, texts[d_idx]))

        async def _route_one(span, prev_label, text):
            async with semaphore:
                if span.entity_class == "O":
                    context = self.router._extract_context(text, span.char_start, span.char_end)
                    clean_span = text[span.char_start:span.char_end].strip()
                    from src.inference.prompts import PROMPT_O_SPAN, ENTITY_CLASSES_STR
                    prompt = PROMPT_O_SPAN.format(context=context, span_text=clean_span, token_count=len(span.indices), target_labels_str=ENTITY_CLASSES_STR)
                    try:
                        raw = await self.router.call_async([{"role": "system", "content": "You are a PII expert."}, {"role": "user", "content": prompt}])
                        return {"is_pii": raw.get("is_pii", False), "corrected_label": f"B-{raw.get('entity_class', 'O').strip()}"}
                    except Exception:
                        return {"is_pii": False, "corrected_label": "O"}
                else:
                    return await self.router.disambiguate_span_async(
                        span_text=text[span.char_start:span.char_end], token_count=len(span.indices),
                        full_text=text, span_start=span.char_start, span_end=span.char_end,
                        current_pred=span.entity_class, prev_label=prev_label
                    )

        coroutines = [_route_one(span, prev, text) for _, _, span, prev, text in tasks]
        llm_responses = await asyncio.gather(*coroutines, return_exceptions=True)

        llm_results = {}
        for i, (d_idx, s_idx, _, _, _) in enumerate(tasks):
            res = llm_responses[i]
            if isinstance(res, Exception): res = None
            llm_results[(d_idx, s_idx)] = res

        results = []
        for d_idx, text in enumerate(texts):
            sub, off, src, pending_spans, conf = batch_state[d_idx]
            for s_idx, span in enumerate(pending_spans):
                res = llm_results.get((d_idx, s_idx))
                if res:
                    self._apply_llm_corrections(text, sub, off, src, conf, span, res)
            
            ent, red, _, _ = self._phase4_to_8(text, sub, off, src, conf)
            results.append((ent, red))
            
        return results
