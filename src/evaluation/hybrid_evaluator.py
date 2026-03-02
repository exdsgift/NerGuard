import os
import logging
import json
import ast
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict


from src.inference.llm_router import LLMRouter
from src.inference.entity_router import EntitySpecificRouter
from src.inference.regex_validator import RegexValidator
from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    NVIDIA_TO_MODEL_MAP,
    EXCLUDED_NVIDIA_LABELS,
)
from src.core.label_mapper import LabelMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HybridEval")


@dataclass
class EntitySpan:
    """A contiguous span of B-X / I-X tokens of the same entity class."""
    indices: List[int]      # token indices in the flat sequence
    entity_class: str       # entity class without BIO prefix, e.g. "SURNAME"
    is_uncertain: bool      # True if the B- anchor token meets routing thresholds
    char_start: int         # char start of the first token
    char_end: int           # char end of the last token


def assemble_entity_spans(
    pred_labels: List[str],
    entropy_flat,
    conf_flat,
    offset_flat,
    labels_flat,
    entity_router,
) -> List[EntitySpan]:
    """
    Group consecutive B-X / I-X predicted tokens into EntitySpan objects.

    Rules:
    - B-X starts a new span; is_uncertain = entity_router.should_route(B-X, ...)
    - Following I-X tokens of the same class are appended to the span
    - A -100 token (excluded or special) breaks the span
    - O tokens and orphan I- tokens produce no span (skipped)

    The routing decision is anchored on the B- token: if the B- is confident,
    the entire span is skipped (anchor propagation). This eliminates the harmful
    per-token routing of continuation tokens (I-SURNAME, I-GIVENNAME, etc.).
    """
    spans: List[EntitySpan] = []
    n = len(pred_labels)
    i = 0
    while i < n:
        if labels_flat[i] == -100:
            i += 1
            continue

        label = pred_labels[i]

        if label.startswith("B-"):
            entity_class = label[2:]
            is_uncertain = entity_router.should_route(
                predicted_label=label,
                entropy=float(entropy_flat[i]),
                confidence=float(conf_flat[i]),
            )
            indices = [i]
            char_end = int(offset_flat[i][1])

            j = i + 1
            while j < n:
                if labels_flat[j] == -100:
                    break
                if pred_labels[j] != f"I-{entity_class}":
                    break
                indices.append(j)
                char_end = int(offset_flat[j][1])
                j += 1

            spans.append(EntitySpan(
                indices=indices,
                entity_class=entity_class,
                is_uncertain=is_uncertain,
                char_start=int(offset_flat[i][0]),
                char_end=char_end,
            ))
            i = j

        else:
            # O or orphan I- token: no span
            i += 1

    return spans


@dataclass
class EvalConfig:
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = "./plots/evaluation_comparison"
    max_samples: Optional[int] = 1000
    context_length: int = 512
    THRESHOLD_ENTROPY: float = DEFAULT_ENTROPY_THRESHOLD
    THRESHOLD_CONF: float = DEFAULT_CONFIDENCE_THRESHOLD
    use_selective_routing: bool = True  # Enable entity-specific routing
    block_continuation_tokens: bool = True  # Block I- tokens (LLM harms them)
    unbiased: bool = True  # Exclude ambiguous NVIDIA labels from metrics (intersection-only)
    use_regex: bool = True  # Enable regex post-processing and LLM routing guard
    two_pass: bool = False  # Route B- tokens first, then I- (ignored when use_span_routing=True)
    use_span_routing: bool = True  # Route entire B-/I- spans as one LLM call (anchor propagation)


@dataclass
class HybridResult:
    true_ids: List[int] = field(default_factory=list)
    baseline_preds: List[int] = field(default_factory=list)
    hybrid_preds: List[int] = field(default_factory=list)
    llm_calls: int = 0
    llm_corrections: int = 0
    llm_wrong_corrections: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    entropy_scores: List[float] = field(default_factory=list)
    llm_interventions: List[Dict] = field(default_factory=list)
    per_class_stats: Dict[str, Dict] = field(
        default_factory=lambda: defaultdict(
            lambda: {
                "total": 0,
                "baseline_correct": 0,
                "hybrid_correct": 0,
                "llm_called": 0,
                "llm_helped": 0,
                "llm_hurt": 0,
            }
        )
    )
    # Unbiased evaluation tracking
    n_excluded_span_tokens: int = 0   # tokens excluded due to EXCLUDED_NVIDIA_LABELS
    n_total_span_tokens: int = 0      # total tokens that were in any NVIDIA span
    # Per-sentence label sequences for seqeval span-level F1
    sentence_true: List[List[str]] = field(default_factory=list)
    sentence_base: List[List[str]] = field(default_factory=list)
    sentence_hyb: List[List[str]] = field(default_factory=list)
    # Regex stats
    regex_llm_skips: int = 0      # LLM calls skipped due to regex confirmation
    regex_promotions: int = 0     # O tokens promoted to entity by regex post-processing


class HybridEvaluator:
    def __init__(
        self, config: EvalConfig, llm_router: Optional[LLMRouter] = None
    ):
        self.config = config
        self.llm_router = llm_router
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize entity-specific router
        self.entity_router = EntitySpecificRouter(
            entropy_threshold=config.THRESHOLD_ENTROPY,
            confidence_threshold=config.THRESHOLD_CONF,
            enable_selective=config.use_selective_routing,
            block_continuation_tokens=config.block_continuation_tokens,
        )
        logger.info(f"Entity router: {self.entity_router}")

        # Initialize regex validator (optional post-processing + LLM guard)
        self.regex_validator = RegexValidator() if config.use_regex else None
        if self.regex_validator:
            logger.info(f"Regex validator enabled for: {list(self.regex_validator._compiled.keys())}")

        self._init_resources()
        os.makedirs(config.output_dir, exist_ok=True)

    def _init_resources(self):
        logger.info(f"Loading resources from: {self.config.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_path
            )
            self.model.to(self.device)
            self.model.eval()
            self.mapper = LabelMapper(self.model.config.id2label)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def load_and_preprocess(self):
        logger.info("Loading NVIDIA/Nemotron-PII dataset...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train")
        if self.config.max_samples:
            ds = ds.select(range(min(len(ds), self.config.max_samples)))
        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _tokenize_and_align(self, text: str, spans: List[Dict]) -> Dict:
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.context_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = tokenized["offset_mapping"][0].numpy()
        labels = [self.mapper.model_label2id.get("O", 0)] * len(offset_mapping)
        n_excluded = 0
        n_span = 0

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                labels[idx] = -100
                continue

            for span in spans:
                if end <= span["start"] or start >= span["end"]:
                    continue

                n_span += 1

                # Unbiased mode: exclude tokens whose NVIDIA label is ambiguous
                if self.config.unbiased and span["label"] in EXCLUDED_NVIDIA_LABELS:
                    labels[idx] = -100
                    n_excluded += 1
                    break

                is_start = start == span["start"]
                if not is_start and idx > 0 and labels[idx - 1] != -100:
                    prev_label = self.mapper.model_id2label.get(labels[idx - 1], "O")
                    curr_base = NVIDIA_TO_MODEL_MAP.get(span["label"], "O")
                    if curr_base not in prev_label:
                        is_start = True

                labels[idx] = self.mapper.get_token_label_id(span["label"], is_start)
                break

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "offset_mapping": offset_mapping,
            "labels": torch.tensor([labels]),
            "text": text,
            "n_excluded": n_excluded,
            "n_span": n_span,
        }

    @staticmethod
    def _parse_spans(spans: Any) -> List[Dict]:
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except (json.JSONDecodeError, ValueError):
                try:
                    spans = ast.literal_eval(spans)
                except (ValueError, SyntaxError):
                    return []
        if isinstance(spans, list):
            valid = [
                s
                for s in spans
                if isinstance(s, dict) and {"label", "start", "end"} <= s.keys()
            ]
            return sorted(valid, key=lambda x: x["start"])
        return []

    def evaluate_comparison(self, dataset):
        import time
        logger.info(f"Starting Comparative Inference on {len(dataset)} samples...")
        logger.info(
            f"Using thresholds - Confidence: {self.config.THRESHOLD_CONF}, Entropy: {self.config.THRESHOLD_ENTROPY}"
        )

        res = HybridResult()
        t_start = time.time()

        for i, example in tqdm(
            enumerate(dataset), total=len(dataset), desc="Processing"
        ):
            spans = self._parse_spans(example["spans"])
            full_text = example["text"]

            processed = self._tokenize_and_align(full_text, spans)
            input_ids = processed["input_ids"].to(self.device)
            mask = processed["attention_mask"].to(self.device)
            labels = processed["labels"].to(self.device)
            offsets = processed["offset_mapping"]
            res.n_excluded_span_tokens += processed["n_excluded"]
            res.n_total_span_tokens += processed["n_span"]

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=mask)
                logits = outputs.logits

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                confidence, preds = torch.max(probs, dim=-1)

            preds_flat = preds[0].cpu().numpy()
            conf_flat = confidence[0].cpu().numpy()
            entropy_flat = entropy[0].cpu().numpy()
            labels_flat = labels[0].cpu().numpy()

            hybrid_preds_flat = preds_flat.copy()

            if self.llm_router:
                # Phase 1: collect per-token stats (single pass — avoids double-counting
                # in two-pass routing mode where tokens are visited twice)
                for idx in range(len(preds_flat)):
                    if labels_flat[idx] == -100:
                        continue
                    true_label = self.mapper.model_id2label[labels_flat[idx]]
                    pred_id_val = int(preds_flat[idx])
                    res.per_class_stats[true_label]["total"] += 1
                    if pred_id_val == labels_flat[idx]:
                        res.per_class_stats[true_label]["baseline_correct"] += 1
                    res.confidence_scores.append(float(conf_flat[idx]))
                    res.entropy_scores.append(float(entropy_flat[idx]))

                # Phase 2: routing
                if self.config.use_span_routing:
                    # Span-level routing (default): group B-/I- tokens into entity spans
                    # and route each span as a single LLM call.
                    # Anchor propagation: if B- is confident, the whole span is skipped —
                    # this eliminates harmful I- routing (I-SURNAME: -16, I-GIVENNAME: -8).
                    pred_labels_list = [
                        self.mapper.model_id2label[int(p)] for p in preds_flat
                    ]
                    spans = assemble_entity_spans(
                        pred_labels_list, entropy_flat, conf_flat,
                        offsets, labels_flat, self.entity_router,
                    )

                    for span in spans:
                        # Only route entity spans (not O tokens)
                        if span.entity_class == "O":
                            continue

                        # Anchor propagation: B- was confident → skip the whole span
                        if not span.is_uncertain:
                            continue

                        anchor_idx = span.indices[0]
                        if labels_flat[anchor_idx] == -100:
                            continue

                        true_label_anchor = self.mapper.model_id2label[labels_flat[anchor_idx]]
                        if true_label_anchor == "O":
                            continue  # Ground truth is O: don't route

                        span_text = full_text[span.char_start:span.char_end]
                        if not span_text.strip():
                            continue

                        # Regex guard on the full span
                        if self.regex_validator and self.regex_validator.can_skip_llm(
                            span.entity_class, full_text, span.char_start, span.char_end
                        ):
                            res.regex_llm_skips += 1
                            continue

                        prev_label = (
                            self.mapper.model_id2label[hybrid_preds_flat[anchor_idx - 1]]
                            if anchor_idx > 0 else "O"
                        )

                        res.llm_calls += 1

                        llm_out = self.llm_router.disambiguate_span(
                            span_text=span_text,
                            token_count=len(span.indices),
                            full_text=full_text,
                            span_start=span.char_start,
                            span_end=span.char_end,
                            current_pred=span.entity_class,
                            prev_label=prev_label,
                        )

                        is_pii = llm_out.get("is_pii", False)
                        corrected_label_b = llm_out.get("corrected_label", f"B-{span.entity_class}")
                        entity_out = corrected_label_b.replace("B-", "").replace("I-", "") if is_pii else None

                        # Apply to every token in the span
                        for k, idx in enumerate(span.indices):
                            if labels_flat[idx] == -100:
                                continue

                            pred_id = int(preds_flat[idx])
                            tok_true = self.mapper.model_id2label[labels_flat[idx]]

                            if entity_out:
                                bio_prefix = "B-" if k == 0 else "I-"
                                new_label = f"{bio_prefix}{entity_out}"
                                new_id = self.mapper.model_label2id.get(
                                    new_label, self.mapper.model_label2id.get("O", 0)
                                )
                            else:
                                new_label = "O"
                                new_id = self.mapper.model_label2id.get("O", 0)

                            if tok_true != "O":
                                res.per_class_stats[tok_true]["llm_called"] += 1

                            res.llm_interventions.append({
                                "token": full_text[offsets[idx][0]:offsets[idx][1]],
                                "true_label": tok_true,
                                "baseline_pred": self.mapper.model_id2label[pred_id],
                                "llm_pred": new_label,
                                "confidence": float(conf_flat[idx]),
                                "entropy": float(entropy_flat[idx]),
                                "was_correct_before": pred_id == labels_flat[idx],
                                "is_correct_after": new_id == labels_flat[idx],
                            })

                            if new_id != pred_id:
                                hybrid_preds_flat[idx] = new_id
                                if new_id == labels_flat[idx] and pred_id != labels_flat[idx]:
                                    res.llm_corrections += 1
                                    res.per_class_stats[tok_true]["llm_helped"] += 1
                                elif new_id != labels_flat[idx] and pred_id == labels_flat[idx]:
                                    res.llm_wrong_corrections += 1
                                    res.per_class_stats[tok_true]["llm_hurt"] += 1

                else:
                    # Token-by-token routing (legacy, use --no-span-routing to enable)
                    # two_pass=True: B- tokens first, then I- (chaining effect).
                    route_passes = ["B-", "I-"] if self.config.two_pass else ["all"]
                    for pass_filter in route_passes:
                        for idx, (token_id, pred_id, conf, ent) in enumerate(
                            zip(input_ids[0], preds_flat, conf_flat, entropy_flat)
                        ):
                            if labels_flat[idx] == -100:
                                continue

                            true_label = self.mapper.model_id2label[labels_flat[idx]]
                            pred_label = self.mapper.model_id2label[pred_id]

                            if pass_filter == "B-" and pred_label.startswith("I-"):
                                continue
                            if pass_filter == "I-" and not pred_label.startswith("I-"):
                                continue

                            should_route = self.entity_router.should_route(
                                predicted_label=pred_label,
                                entropy=float(ent),
                                confidence=float(conf),
                            )

                            if should_route and pred_label != "O" and true_label != "O":
                                start_char, end_char = offsets[idx]
                                if start_char == end_char:
                                    continue

                                token_text = full_text[start_char:end_char]
                                current_pred_label = self.mapper.model_id2label[pred_id]
                                entity_class = current_pred_label.replace("B-", "").replace("I-", "")
                                prev_label = (
                                    self.mapper.model_id2label[hybrid_preds_flat[idx - 1]]
                                    if idx > 0 else "O"
                                )

                                if self.regex_validator and self.regex_validator.can_skip_llm(
                                    entity_class, full_text, int(start_char), int(end_char)
                                ):
                                    res.regex_llm_skips += 1
                                    continue

                                res.llm_calls += 1
                                res.per_class_stats[true_label]["llm_called"] += 1

                                llm_out = self.llm_router.disambiguate(
                                    target_token=token_text,
                                    full_text=full_text,
                                    char_start=start_char,
                                    char_end=end_char,
                                    current_pred=current_pred_label,
                                    prev_label=prev_label,
                                )

                                corrected_label = llm_out.get("corrected_label", "O")
                                corrected_id = self.mapper.model_label2id.get(
                                    corrected_label, self.mapper.model_label2id.get("O")
                                )

                                res.llm_interventions.append({
                                    "token": token_text,
                                    "true_label": true_label,
                                    "baseline_pred": pred_label,
                                    "llm_pred": corrected_label,
                                    "confidence": float(conf),
                                    "entropy": float(ent),
                                    "was_correct_before": pred_id == labels_flat[idx],
                                    "is_correct_after": corrected_id == labels_flat[idx],
                                })

                                if corrected_id != pred_id:
                                    hybrid_preds_flat[idx] = corrected_id
                                    if corrected_id == labels_flat[idx] and pred_id != labels_flat[idx]:
                                        res.llm_corrections += 1
                                        res.per_class_stats[true_label]["llm_helped"] += 1
                                    elif corrected_id != labels_flat[idx] and pred_id == labels_flat[idx]:
                                        res.llm_wrong_corrections += 1
                                        res.per_class_stats[true_label]["llm_hurt"] += 1

                # Phase 3: hybrid_correct — collected after ALL routing passes complete
                for idx in range(len(labels_flat)):
                    if labels_flat[idx] == -100:
                        continue
                    true_label = self.mapper.model_id2label[labels_flat[idx]]
                    if hybrid_preds_flat[idx] == labels_flat[idx]:
                        res.per_class_stats[true_label]["hybrid_correct"] += 1

            # Regex post-processing: promote O-predicted tokens where regex finds PII
            if self.regex_validator:
                id2label_map = self.model.config.id2label
                label2id_map = self.mapper.model_label2id
                before_count = int((hybrid_preds_flat != self.mapper.model_label2id.get("O", 0)).sum())
                hybrid_preds_flat = self.regex_validator.correct_predictions(
                    full_text, offsets, hybrid_preds_flat, id2label_map, label2id_map
                )
                after_count = int((hybrid_preds_flat != self.mapper.model_label2id.get("O", 0)).sum())
                res.regex_promotions += max(0, after_count - before_count)

            active_mask = labels_flat != -100
            res.true_ids.extend(labels_flat[active_mask])
            res.baseline_preds.extend(preds_flat[active_mask])
            res.hybrid_preds.extend(hybrid_preds_flat[active_mask])

            # Collect per-sentence sequences for seqeval span-level F1
            id2label_map = self.model.config.id2label
            sent_true = [id2label_map[j] for j in labels_flat[active_mask]]
            sent_base = [id2label_map[j] for j in preds_flat[active_mask]]
            sent_hyb = [id2label_map[j] for j in hybrid_preds_flat[active_mask]]
            if sent_true:
                res.sentence_true.append(sent_true)
                res.sentence_base.append(sent_base)
                res.sentence_hyb.append(sent_hyb)

        elapsed = time.time() - t_start
        logger.info(
            f"Evaluation complete. LLM calls: {res.llm_calls}, Helpful: {res.llm_corrections}, Harmful: {res.llm_wrong_corrections}"
        )
        self._generate_comparison_report(res, elapsed_seconds=elapsed)

    def _generate_comparison_report(self, res: HybridResult, elapsed_seconds: float = 0.0):
        logger.info("Generating comprehensive comparison report...")

        id2label = self.model.config.id2label
        y_true = [id2label[i] for i in res.true_ids]
        y_base = [id2label[i] for i in res.baseline_preds]
        y_hyb = [id2label[i] for i in res.hybrid_preds]

        labels = sorted(list({l for l in y_true if l != "O"}))

        base_report = classification_report(
            y_true, y_base, labels=labels, zero_division=0
        )
        hyb_report = classification_report(
            y_true, y_hyb, labels=labels, zero_division=0
        )

        # Coverage stats (unbiased mode)
        coverage_lines = []
        if self.config.unbiased and res.n_total_span_tokens > 0:
            included = res.n_total_span_tokens - res.n_excluded_span_tokens
            coverage_pct = included / res.n_total_span_tokens * 100
            coverage_lines = [
                f"Evaluation mode: UNBIASED (intersection-only)",
                f"Span tokens total:    {res.n_total_span_tokens}",
                f"Span tokens included: {included} ({coverage_pct:.1f}%)",
                f"Span tokens excluded: {res.n_excluded_span_tokens} ({100 - coverage_pct:.1f}%)",
                f"Excluded NVIDIA labels: {sorted(EXCLUDED_NVIDIA_LABELS)}",
            ]
        elif not self.config.unbiased:
            coverage_lines = ["Evaluation mode: STANDARD (all NVIDIA labels included)"]

        # Seqeval span-level F1
        span_f1_lines = []
        try:
            from seqeval.metrics import f1_score as seq_f1
            if res.sentence_true:
                base_span_f1 = seq_f1(res.sentence_true, res.sentence_base, zero_division=0)
                hyb_span_f1 = seq_f1(res.sentence_true, res.sentence_hyb, zero_division=0)
                span_f1_lines = [
                    f"Span-level F1 (seqeval strict):",
                    f"  Baseline: {base_span_f1:.4f}",
                    f"  Hybrid:   {hyb_span_f1:.4f}",
                    f"  Delta:    {hyb_span_f1 - base_span_f1:+.4f}",
                ]
        except ImportError:
            span_f1_lines = ["seqeval not available — span-level F1 skipped"]

        # Text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BASELINE MODEL (DeBERTa Only)")
        report_lines.append("=" * 80)
        if coverage_lines:
            report_lines.extend(coverage_lines)
            report_lines.append("")
        report_lines.append(base_report)

        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"HYBRID MODEL (DeBERTa + LLM Router)")
        report_lines.append(
            f"Thresholds: Conf < {self.config.THRESHOLD_CONF}, Entropy > {self.config.THRESHOLD_ENTROPY}"
        )
        report_lines.append(f"Selective Routing: {self.config.use_selective_routing}")
        report_lines.append(f"Span Routing:      {self.config.use_span_routing}")
        report_lines.append(f"Two-Pass Routing:  {self.config.two_pass} (ignored when span routing enabled)")
        report_lines.append(f"Regex Validator:   {self.config.use_regex}")

        # Entity router stats
        router_stats = self.entity_router.get_stats()
        report_lines.append(f"Tokens Checked: {router_stats['total_checked']}")
        report_lines.append(f"Routed to LLM: {router_stats['routed']} ({router_stats['routing_rate']})")
        report_lines.append(f"Blocked by Entity Type: {router_stats['blocked_by_entity']}")
        report_lines.append(f"Blocked by I- Continuation: {router_stats.get('blocked_by_continuation', 0)}")
        report_lines.append(f"Blocked by Threshold: {router_stats['blocked_by_threshold']}")
        if router_stats['routed_by_entity']:
            report_lines.append(f"Routed by Entity: {router_stats['routed_by_entity']}")

        report_lines.append(f"\nLLM Calls: {res.llm_calls}")
        report_lines.append(f"Helpful Corrections: {res.llm_corrections}")
        report_lines.append(f"Harmful Corrections: {res.llm_wrong_corrections}")
        report_lines.append(
            f"Net Improvement: {res.llm_corrections - res.llm_wrong_corrections}"
        )
        if self.config.use_regex:
            report_lines.append(f"Regex LLM Skips: {res.regex_llm_skips}")
            report_lines.append(f"Regex O→Entity Promotions: {res.regex_promotions}")
        elapsed_min = elapsed_seconds / 60
        report_lines.append(f"Elapsed Time: {elapsed_seconds:.1f}s ({elapsed_min:.2f} min)")
        if span_f1_lines:
            report_lines.append("")
            report_lines.extend(span_f1_lines)
        report_lines.append("=" * 80)
        report_lines.append(hyb_report)

        # Per-class analysis
        report_lines.append("\n" + "=" * 80)
        report_lines.append("PER-CLASS LLM IMPACT ANALYSIS")
        report_lines.append("=" * 80)

        for label in sorted(res.per_class_stats.keys()):
            stats = res.per_class_stats[label]
            if stats["total"] == 0:
                continue

            baseline_acc = stats["baseline_correct"] / stats["total"] * 100
            hybrid_acc = stats["hybrid_correct"] / stats["total"] * 100
            improvement = hybrid_acc - baseline_acc

            report_lines.append(f"\n{label}:")
            report_lines.append(f"  Total instances: {stats['total']}")
            report_lines.append(f"  Baseline accuracy: {baseline_acc:.1f}%")
            report_lines.append(f"  Hybrid accuracy: {hybrid_acc:.1f}%")
            report_lines.append(f"  Improvement: {improvement:+.1f}%")
            report_lines.append(f"  LLM interventions: {stats['llm_called']}")
            report_lines.append(
                f"  Helped: {stats['llm_helped']}, Hurt: {stats['llm_hurt']}"
            )

        report_text = "\n".join(report_lines)

        # Log and save
        for line in report_lines:
            logger.info(line)

        with open(
            os.path.join(self.config.output_dir, "comparison_report.txt"), "w"
        ) as f:
            f.write(report_text)

        # Save structured JSON summary for easy cross-model comparison
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        acc_base = accuracy_score(y_true, y_base)
        acc_hyb = accuracy_score(y_true, y_hyb)
        p_base, r_base, f1_base, _ = precision_recall_fscore_support(y_true, y_base, average="macro", zero_division=0)
        p_hyb, r_hyb, f1_hyb, _ = precision_recall_fscore_support(y_true, y_hyb, average="macro", zero_division=0)

        # Seqeval span-level F1 for JSON summary
        span_f1_summary: Dict[str, Any] = {}
        try:
            from seqeval.metrics import f1_score as seq_f1
            if res.sentence_true:
                span_f1_summary = {
                    "baseline_span_f1": round(seq_f1(res.sentence_true, res.sentence_base, zero_division=0), 4),
                    "hybrid_span_f1": round(seq_f1(res.sentence_true, res.sentence_hyb, zero_division=0), 4),
                }
                span_f1_summary["delta_span_f1"] = round(
                    span_f1_summary["hybrid_span_f1"] - span_f1_summary["baseline_span_f1"], 4
                )
        except ImportError:
            pass

        summary = {
            "samples": self.config.max_samples,
            "evaluation_mode": "unbiased" if self.config.unbiased else "standard",
            "span_routing": self.config.use_span_routing,
            "two_pass_routing": self.config.two_pass,
            "regex_enabled": self.config.use_regex,
            "coverage": {
                "total_span_tokens": res.n_total_span_tokens,
                "included_span_tokens": res.n_total_span_tokens - res.n_excluded_span_tokens,
                "excluded_span_tokens": res.n_excluded_span_tokens,
                "coverage_pct": round(
                    (res.n_total_span_tokens - res.n_excluded_span_tokens) / res.n_total_span_tokens * 100, 1
                ) if res.n_total_span_tokens > 0 else 100.0,
            },
            "baseline": {
                "accuracy": round(acc_base, 4),
                "macro_precision": round(p_base, 4),
                "macro_recall": round(r_base, 4),
                "macro_f1": round(f1_base, 4),
            },
            "hybrid": {
                "accuracy": round(acc_hyb, 4),
                "macro_precision": round(p_hyb, 4),
                "macro_recall": round(r_hyb, 4),
                "macro_f1": round(f1_hyb, 4),
            },
            "delta": {
                "accuracy": round(acc_hyb - acc_base, 4),
                "macro_f1": round(f1_hyb - f1_base, 4),
            },
            **span_f1_summary,
            "routing": {
                "llm_calls": res.llm_calls,
                "llm_helped": res.llm_corrections,
                "llm_hurt": res.llm_wrong_corrections,
                "net_corrections": res.llm_corrections - res.llm_wrong_corrections,
                "regex_llm_skips": res.regex_llm_skips,
                "regex_promotions": res.regex_promotions,
            },
            "elapsed_seconds": round(elapsed_seconds, 1),
            "elapsed_minutes": round(elapsed_seconds / 60, 2),
        }
        json_path = os.path.join(self.config.output_dir, "metrics_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"All reports saved to {self.config.output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid NerGuard Evaluation")
    parser.add_argument(
        "--selective-routing",
        action="store_true",
        default=True,
        help="Enable entity-specific selective routing (default: True)",
    )
    parser.add_argument(
        "--no-selective-routing",
        action="store_true",
        help="Disable selective routing (route all uncertain predictions)",
    )
    parser.add_argument(
        "--allow-continuation-tokens",
        action="store_true",
        help="Allow I- continuation tokens to be routed (default: blocked due to LLM harm)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots/evaluation_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2.5:3b",
        help="Ollama model name to use as LLM router (default: qwen2.5:3b)",
    )
    parser.add_argument(
        "--no-unbiased",
        action="store_true",
        help="Disable intersection-only evaluation (use full NVIDIA_TO_MODEL_MAP including ambiguous labels)",
    )
    parser.add_argument(
        "--no-regex",
        action="store_true",
        help="Disable regex post-processing and LLM routing guard",
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Enable two-pass routing: B- first, then I- (ignored when span routing enabled)",
    )
    parser.add_argument(
        "--no-span-routing",
        action="store_true",
        help="Disable span-level routing and fall back to token-by-token routing",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from dotenv import load_dotenv
    load_dotenv()

    use_selective = not args.no_selective_routing
    block_continuation = not args.allow_continuation_tokens

    config = EvalConfig(
        model_path="./models/mdeberta-pii-safe/final",
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        use_selective_routing=use_selective,
        block_continuation_tokens=block_continuation,
        unbiased=not args.no_unbiased,
        use_regex=not args.no_regex,
        two_pass=args.two_pass,
        use_span_routing=not args.no_span_routing,
    )

    logger.info(f"Selective routing: {use_selective}")
    logger.info(f"Block I- tokens: {block_continuation}")

    router = None
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Initializing OpenAI Router...")
        router = LLMRouter(
            source="openai", model="gpt-4o-mini", enable_cache=True
        )
    else:
        logger.warning(f"No OPENAI_API_KEY found. Using Ollama ({args.ollama_model})...")
        try:
            router = LLMRouter(
                source="ollama", ollama_model=args.ollama_model, enable_cache=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama router: {e}")

    if router is None:
        logger.warning("Running in baseline-only mode (no LLM router available)")

    evaluator = HybridEvaluator(config, llm_router=router)
    dataset = evaluator.load_and_preprocess()
    evaluator.evaluate_comparison(dataset)
