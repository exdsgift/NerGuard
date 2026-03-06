"""Metrics computation for the NER PII benchmark.

Computes all required metrics: token-level, entity-level (seqeval),
latency stats, confusion matrix, error analysis.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.benchmark.label_protocol import LabelOverlapReport

logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """A single false positive or false negative case."""
    token: str
    context: str  # Surrounding text
    predicted: str
    expected: str
    sample_id: str


@dataclass
class BenchmarkMetrics:
    """All metrics for a single system × dataset evaluation."""

    # Core token-level
    precision_macro: float = 0.0
    precision_micro: float = 0.0
    precision_weighted: float = 0.0
    recall_macro: float = 0.0
    recall_micro: float = 0.0
    recall_weighted: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0

    # Entity-level (seqeval)
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0

    # Per-entity scores
    per_entity_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Latency
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    latency_samples: List[float] = field(default_factory=list)

    # Memory
    memory_peak_mb: float = 0.0
    gpu_peak_mb: float = 0.0

    # Per length bucket
    per_length_bucket: Dict[str, Dict] = field(default_factory=dict)

    # Per entity frequency
    per_entity_frequency: Dict[str, Dict] = field(default_factory=dict)

    # Confusion matrix
    confusion_matrix: Dict = field(default_factory=dict)

    # Error cases
    false_positives: List[Dict] = field(default_factory=list)
    false_negatives: List[Dict] = field(default_factory=list)

    # Aligned predictions (for significance tests)
    aligned_y_true: List[List[str]] = field(default_factory=list)
    aligned_y_pred: List[List[str]] = field(default_factory=list)

    # Metadata
    n_samples: int = 0
    n_tokens: int = 0
    n_evaluated_labels: int = 0

    def to_results_dict(self) -> Dict:
        """Convert to the results.json schema."""
        return {
            "n_samples": self.n_samples,
            "n_tokens": self.n_tokens,
            "n_evaluated_labels": self.n_evaluated_labels,
            "token_level": {
                "precision_macro": self.precision_macro,
                "precision_micro": self.precision_micro,
                "precision_weighted": self.precision_weighted,
                "recall_macro": self.recall_macro,
                "recall_micro": self.recall_micro,
                "recall_weighted": self.recall_weighted,
                "f1_macro": self.f1_macro,
                "f1_micro": self.f1_micro,
                "f1_weighted": self.f1_weighted,
            },
            "entity_level": {
                "precision": self.entity_precision,
                "recall": self.entity_recall,
                "f1": self.entity_f1,
            },
            "latency": {
                "mean_ms": self.latency_mean_ms,
                "median_ms": self.latency_median_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
                "throughput_samples_per_sec": self.throughput_samples_per_sec,
            },
            "memory": {
                "peak_mb": self.memory_peak_mb,
                "gpu_peak_mb": self.gpu_peak_mb,
            },
            "per_length_bucket": self.per_length_bucket,
            "per_entity_frequency": self.per_entity_frequency,
        }


class MetricsComputer:
    """Computes all benchmark metrics."""

    def compute_all(
        self,
        y_true_samples: List[List[str]],
        y_pred_samples: List[List[str]],
        overlap_report: LabelOverlapReport,
        latencies: List[float],
        texts: List[str],
        sample_ids: List[str],
        tokens_per_sample: List[List[str]],
    ) -> BenchmarkMetrics:
        """Compute all metrics for a system × dataset evaluation.

        Labels not in the evaluated set are masked to "O".
        """
        metrics = BenchmarkMetrics()
        metrics.n_samples = len(y_true_samples)

        evaluated_labels = overlap_report.evaluated_labels
        is_tier2 = overlap_report.tier == 2

        # In Tier 2: build mapping from system labels → dataset labels
        # e.g., PERSON → {GIVENNAME, SURNAME}
        sys_to_ds_map: Dict[str, Set[str]] = {}
        if is_tier2 and overlap_report.semantic_alignment:
            for sys_label, ds_labels in overlap_report.semantic_alignment.items():
                sys_to_ds_map[sys_label] = set(ds_labels)

        # Flatten and mask labels
        y_true_flat = []
        y_pred_flat = []
        y_true_seq = []  # For seqeval (per-sample lists)
        y_pred_seq = []

        for sample_idx, (true_labels, pred_labels) in enumerate(
            zip(y_true_samples, y_pred_samples)
        ):
            min_len = min(len(true_labels), len(pred_labels))
            true_masked = []
            pred_masked = []

            for i in range(min_len):
                true_lbl = true_labels[i]
                pred_lbl = pred_labels[i]

                true_entity = _strip_bio(true_lbl)
                pred_entity = _strip_bio(pred_lbl)

                if is_tier2 and sys_to_ds_map:
                    # Remap system predictions to dataset labels
                    # If system predicts "PERSON" and ground truth is "GIVENNAME",
                    # and PERSON maps to {GIVENNAME, SURNAME}, remap prediction to
                    # match the ground truth label for fair comparison
                    if pred_entity in sys_to_ds_map:
                        mapped_ds_labels = sys_to_ds_map[pred_entity]
                        if true_entity in mapped_ds_labels:
                            # Match: remap prediction to ground truth label
                            pred_lbl = _rebuild_bio(pred_lbl, true_entity)
                            pred_entity = true_entity
                        else:
                            # System says entity but ground truth doesn't match this alignment
                            # Keep the first dataset label as the prediction (for FP tracking)
                            first_ds = next(iter(mapped_ds_labels))
                            pred_lbl = _rebuild_bio(pred_lbl, first_ds)
                            pred_entity = first_ds

                # Mask non-evaluated labels to "O"
                if true_entity != "O" and true_entity not in evaluated_labels:
                    true_lbl = "O"
                if pred_entity != "O" and pred_entity not in evaluated_labels:
                    pred_lbl = "O"

                true_masked.append(true_lbl)
                pred_masked.append(pred_lbl)

            y_true_flat.extend(true_masked)
            y_pred_flat.extend(pred_masked)
            y_true_seq.append(true_masked)
            y_pred_seq.append(pred_masked)

        metrics.n_tokens = len(y_true_flat)
        metrics.aligned_y_true = y_true_seq
        metrics.aligned_y_pred = y_pred_seq
        metrics.n_evaluated_labels = len(evaluated_labels)

        if not y_true_flat:
            return metrics

        # --- Token-level metrics ---
        entity_labels = sorted({
            lbl for lbl in set(y_true_flat) | set(y_pred_flat) if lbl != "O"
        })

        if entity_labels:
            metrics.precision_macro = precision_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="macro", zero_division=0,
            )
            metrics.precision_micro = precision_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="micro", zero_division=0,
            )
            metrics.precision_weighted = precision_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="weighted", zero_division=0,
            )
            metrics.recall_macro = recall_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="macro", zero_division=0,
            )
            metrics.recall_micro = recall_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="micro", zero_division=0,
            )
            metrics.recall_weighted = recall_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="weighted", zero_division=0,
            )
            metrics.f1_macro = f1_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="macro", zero_division=0,
            )
            metrics.f1_micro = f1_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="micro", zero_division=0,
            )
            metrics.f1_weighted = f1_score(
                y_true_flat, y_pred_flat, labels=entity_labels,
                average="weighted", zero_division=0,
            )

            # Per-entity breakdown
            report = classification_report(
                y_true_flat, y_pred_flat, labels=entity_labels,
                output_dict=True, zero_division=0,
            )
            for lbl in entity_labels:
                if lbl in report:
                    metrics.per_entity_scores[lbl] = {
                        "precision": report[lbl]["precision"],
                        "recall": report[lbl]["recall"],
                        "f1": report[lbl]["f1-score"],
                        "support": report[lbl]["support"],
                    }

        # --- Entity-level (seqeval) ---
        try:
            metrics.entity_precision = seqeval_precision(y_true_seq, y_pred_seq, zero_division=0)
            metrics.entity_recall = seqeval_recall(y_true_seq, y_pred_seq, zero_division=0)
            metrics.entity_f1 = seqeval_f1(y_true_seq, y_pred_seq, zero_division=0)
        except Exception as e:
            logger.warning(f"seqeval failed: {e}")

        # --- Latency ---
        if latencies:
            metrics.latency_samples = latencies
            arr = np.array(latencies)
            metrics.latency_mean_ms = float(np.mean(arr))
            metrics.latency_median_ms = float(np.median(arr))
            metrics.latency_p95_ms = float(np.percentile(arr, 95))
            metrics.latency_p99_ms = float(np.percentile(arr, 99))
            total_time_sec = sum(latencies) / 1000.0
            metrics.throughput_samples_per_sec = (
                len(latencies) / total_time_sec if total_time_sec > 0 else 0
            )

        # --- Confusion matrix ---
        all_labels = sorted(set(y_true_flat) | set(y_pred_flat))
        if len(all_labels) > 1:
            cm = sklearn_confusion_matrix(y_true_flat, y_pred_flat, labels=all_labels)
            metrics.confusion_matrix = {
                "labels": all_labels,
                "matrix": cm.tolist(),
            }

        # --- Per-length bucket ---
        metrics.per_length_bucket = self._compute_length_buckets(
            y_true_samples, y_pred_samples, texts, evaluated_labels, is_tier2, sys_to_ds_map
        )

        # --- Per-entity frequency ---
        metrics.per_entity_frequency = self._compute_frequency_buckets(
            y_true_flat, y_pred_flat, entity_labels
        )

        # --- Error analysis ---
        metrics.false_positives, metrics.false_negatives = self._collect_errors(
            y_true_samples, y_pred_samples, tokens_per_sample, sample_ids, texts,
            evaluated_labels, is_tier2, sys_to_ds_map, max_errors=500,
        )

        return metrics

    def _compute_length_buckets(
        self,
        y_true_samples, y_pred_samples, texts,
        evaluated_labels, is_tier2, sys_to_ds_map,
    ) -> Dict:
        buckets = {"short": [], "medium": [], "long": []}

        for idx, text in enumerate(texts):
            text_len = len(text)
            if text_len < 128:
                bucket = "short"
            elif text_len < 512:
                bucket = "medium"
            else:
                bucket = "long"
            buckets[bucket].append(idx)

        result = {}
        for bucket_name, indices in buckets.items():
            if not indices:
                result[bucket_name] = {"f1_macro": 0.0, "n_samples": 0}
                continue

            flat_true = []
            flat_pred = []
            for idx in indices:
                true_labels = y_true_samples[idx]
                pred_labels = y_pred_samples[idx]
                min_len = min(len(true_labels), len(pred_labels))
                for i in range(min_len):
                    t = true_labels[i]
                    p = pred_labels[i]
                    te = _strip_bio(t)
                    pe = _strip_bio(p)
                    if is_tier2 and sys_to_ds_map and pe in sys_to_ds_map:
                        mapped = sys_to_ds_map[pe]
                        if te in mapped:
                            p = _rebuild_bio(p, te)
                            pe = te
                        else:
                            first_ds = next(iter(mapped))
                            p = _rebuild_bio(p, first_ds)
                            pe = first_ds
                    if te != "O" and te not in evaluated_labels:
                        t = "O"
                    if pe != "O" and pe not in evaluated_labels:
                        p = "O"
                    flat_true.append(t)
                    flat_pred.append(p)

            entity_labels = sorted({l for l in set(flat_true) | set(flat_pred) if l != "O"})
            if entity_labels:
                f1 = f1_score(flat_true, flat_pred, labels=entity_labels, average="macro", zero_division=0)
            else:
                f1 = 0.0

            result[bucket_name] = {"f1_macro": f1, "n_samples": len(indices)}

        return result

    def _compute_frequency_buckets(
        self, y_true_flat, y_pred_flat, entity_labels,
    ) -> Dict:
        # Count entity occurrences in ground truth
        entity_counts = Counter()
        for lbl in y_true_flat:
            if lbl != "O":
                entity_counts[_strip_bio(lbl)] += 1

        # Bucket: tail (<50), mid (50-500), head (>500)
        buckets = {"tail": [], "mid": [], "head": []}
        for entity in set(entity_counts.keys()):
            count = entity_counts[entity]
            if count < 50:
                buckets["tail"].append(entity)
            elif count < 500:
                buckets["mid"].append(entity)
            else:
                buckets["head"].append(entity)

        result = {}
        for bucket_name, entities in buckets.items():
            if not entities:
                result[bucket_name] = {"f1_macro": 0.0, "n_entities": 0, "entities": []}
                continue

            # Filter to BIO labels of these entities
            bio_labels = []
            for e in entities:
                bio_labels.extend([f"B-{e}", f"I-{e}"])
            bio_labels = [l for l in bio_labels if l in set(y_true_flat) | set(y_pred_flat)]

            if bio_labels:
                f1 = f1_score(y_true_flat, y_pred_flat, labels=bio_labels, average="macro", zero_division=0)
            else:
                f1 = 0.0

            result[bucket_name] = {
                "f1_macro": f1,
                "n_entities": len(entities),
                "entities": sorted(entities),
            }

        return result

    def _collect_errors(
        self,
        y_true_samples, y_pred_samples, tokens_per_sample, sample_ids, texts,
        evaluated_labels, is_tier2, sys_to_ds_map, max_errors=500,
    ) -> Tuple[List[Dict], List[Dict]]:
        fps = []
        fns = []

        for idx in range(len(y_true_samples)):
            true_labels = y_true_samples[idx]
            pred_labels = y_pred_samples[idx]
            tokens = tokens_per_sample[idx]
            sid = sample_ids[idx]
            text = texts[idx]
            min_len = min(len(true_labels), len(pred_labels), len(tokens))

            for i in range(min_len):
                t = true_labels[i]
                p = pred_labels[i]
                te = _strip_bio(t)
                pe = _strip_bio(p)

                if is_tier2 and sys_to_ds_map and pe in sys_to_ds_map:
                    mapped = sys_to_ds_map[pe]
                    if te in mapped:
                        p = _rebuild_bio(p, te)
                        pe = te
                    else:
                        first_ds = next(iter(mapped))
                        p = _rebuild_bio(p, first_ds)
                        pe = first_ds
                if te != "O" and te not in evaluated_labels:
                    continue
                if pe != "O" and pe not in evaluated_labels:
                    continue

                # Context window
                ctx_start = max(0, i - 3)
                ctx_end = min(min_len, i + 4)
                context = " ".join(tokens[ctx_start:ctx_end])

                # False positive: predicted entity, true is O
                if te == "O" and pe != "O" and len(fps) < max_errors:
                    fps.append({
                        "token": tokens[i],
                        "context": context,
                        "predicted": p,
                        "expected": "O",
                        "sample_id": sid,
                    })

                # False negative: true is entity, predicted O
                if te != "O" and pe == "O" and len(fns) < max_errors:
                    fns.append({
                        "token": tokens[i],
                        "context": context,
                        "predicted": "O",
                        "expected": t,
                        "sample_id": sid,
                    })

        return fps, fns


def _strip_bio(label: str) -> str:
    """Remove BIO prefix from a label."""
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label


def _rebuild_bio(original: str, new_entity: str) -> str:
    """Rebuild a BIO label with a new entity type, preserving the prefix."""
    if original.startswith("B-"):
        return f"B-{new_entity}"
    elif original.startswith("I-"):
        return f"I-{new_entity}"
    return new_entity
