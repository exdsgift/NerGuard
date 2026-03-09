"""Error decomposition analysis for NER false negatives.

Categorizes false negatives into three types (Definition [Span error decomposition]):

  1. boundary_error  — Gold span overlaps a predicted span with correct label
                       but boundaries differ.
  2. type_confusion  — Gold span overlaps a predicted span with maximum overlap
                       but the label is different (model predicts wrong entity type).
  3. missed_detection — No predicted span overlaps with the gold span at all.

Runs on predictions.json from an experiment directory.
Optionally compares base vs hybrid to show which error type routing fixes.

Usage:
    uv run python -m src.scripts.run_error_decomposition \
        --experiment-dir experiments/nvidia_pii_500samples_2026-03-05 \
        --base-system nerguard_base_nvidia-pii \
        --hybrid-system nerguard_hybrid_v2_gpt-4o_nvidia-pii
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ErrorDecomposition")


def bio_to_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO label sequence to (start_idx, end_idx, label) token-index spans."""
    spans = []
    cur_label = None
    cur_start = None

    for idx, label in enumerate(labels):
        if label.startswith("B-"):
            if cur_label is not None:
                spans.append((cur_start, idx - 1, cur_label))
            cur_label = label[2:]
            cur_start = idx
        elif label.startswith("I-") and cur_label is not None and label[2:] == cur_label:
            pass  # Continue current span
        else:
            if cur_label is not None:
                spans.append((cur_start, idx - 1, cur_label))
            cur_label = None
            cur_start = None

    if cur_label is not None:
        spans.append((cur_start, len(labels) - 1, cur_label))

    return spans


def token_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Number of overlapping tokens between two spans (inclusive ends)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def classify_fn_span(
    gold_start: int,
    gold_end: int,
    gold_label: str,
    pred_spans: List[Tuple[int, int, str]],
) -> str:
    """Classify a false negative span into an error type.

    Returns one of: 'boundary_error', 'type_confusion', 'missed_detection'
    """
    best_overlap = 0
    best_label = None

    for p_start, p_end, p_label in pred_spans:
        ov = token_overlap(gold_start, gold_end, p_start, p_end)
        if ov > best_overlap:
            best_overlap = ov
            best_label = p_label

    if best_overlap == 0:
        return "missed_detection"

    # There is overlap
    if best_label == gold_label:
        return "boundary_error"
    else:
        return "type_confusion"


def decompose_errors(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    system_name: str = "",
) -> Dict:
    """Run full error decomposition on a predictions dataset."""
    total_gold_spans = 0
    total_fn = 0
    error_counts = Counter()
    error_examples: Dict[str, list] = defaultdict(list)
    max_examples = 10

    # Per-entity breakdown
    per_entity_errors: Dict[str, Counter] = defaultdict(Counter)

    for doc_idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
        gold_spans = bio_to_spans(yt)
        pred_spans = bio_to_spans(yp)

        # True positives: gold span exactly matched (exact boundaries + label)
        matched_gold = set()
        for g_start, g_end, g_label in gold_spans:
            for p_start, p_end, p_label in pred_spans:
                if g_start == p_start and g_end == p_end and g_label == p_label:
                    matched_gold.add((g_start, g_end, g_label))
                    break

        total_gold_spans += len(gold_spans)
        fn_spans = [s for s in gold_spans if s not in matched_gold]
        total_fn += len(fn_spans)

        for g_start, g_end, g_label in fn_spans:
            etype = classify_fn_span(g_start, g_end, g_label, pred_spans)
            error_counts[etype] += 1
            per_entity_errors[g_label][etype] += 1

            if len(error_examples[etype]) < max_examples:
                error_examples[etype].append({
                    "doc_idx": doc_idx,
                    "gold_start": g_start,
                    "gold_end": g_end,
                    "gold_label": g_label,
                    "gold_tokens": yt[g_start: g_end + 1],
                })

    if total_fn > 0:
        fractions = {k: v / total_fn for k, v in error_counts.items()}
    else:
        fractions = {}

    return {
        "system": system_name,
        "total_gold_spans": total_gold_spans,
        "total_fn": total_fn,
        "fn_rate": total_fn / total_gold_spans if total_gold_spans > 0 else 0.0,
        "error_counts": dict(error_counts),
        "error_fractions": {k: float(v) for k, v in fractions.items()},
        "per_entity_errors": {k: dict(v) for k, v in per_entity_errors.items()},
        "error_examples": dict(error_examples),
    }


def print_decomposition(results: Dict) -> None:
    total_fn = results["total_fn"]
    print(f"\n  System: {results['system']}")
    print(f"  Gold spans: {results['total_gold_spans']}")
    print(f"  False negatives: {total_fn}  (FN rate: {results['fn_rate']*100:.1f}%)")
    if total_fn > 0:
        print(f"\n  FN breakdown:")
        for etype in ["missed_detection", "boundary_error", "type_confusion"]:
            count = results["error_counts"].get(etype, 0)
            frac = results["error_fractions"].get(etype, 0.0)
            print(f"    {etype:<20} {count:>5}  ({frac*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        default="experiments/nvidia_pii_500samples_2026-03-05",
        help="Root experiment directory containing system subdirectories",
    )
    parser.add_argument(
        "--base-system",
        default="nerguard_base_nvidia-pii",
        help="Subdirectory name for the base system",
    )
    parser.add_argument(
        "--hybrid-system",
        default="nerguard_hybrid_v2_gpt-4o_nvidia-pii",
        help="Subdirectory name for the hybrid system",
    )
    args = parser.parse_args()

    results_all = {}

    print(f"\n{'='*70}")
    print("ERROR DECOMPOSITION ANALYSIS")
    print(f"{'='*70}")

    for system_key, subdir in [("base", args.base_system), ("hybrid", args.hybrid_system)]:
        pred_path = os.path.join(args.experiment_dir, subdir, "predictions.json")
        if not os.path.exists(pred_path):
            logger.warning(f"predictions.json not found at {pred_path}, skipping {system_key}")
            continue

        with open(pred_path) as f:
            pdata = json.load(f)
        y_true = pdata["y_true"]
        y_pred = pdata["y_pred"]

        logger.info(f"Decomposing {system_key} ({subdir}): {len(y_true)} documents")
        res = decompose_errors(y_true, y_pred, system_name=system_key)
        print_decomposition(res)
        results_all[system_key] = res

    # Delta analysis (base → hybrid)
    if "base" in results_all and "hybrid" in results_all:
        base = results_all["base"]
        hyb = results_all["hybrid"]
        print(f"\n{'='*70}")
        print("DELTA (hybrid - base) per error type")
        print(f"{'='*70}")
        for etype in ["missed_detection", "boundary_error", "type_confusion"]:
            b_count = base["error_counts"].get(etype, 0)
            h_count = hyb["error_counts"].get(etype, 0)
            delta = h_count - b_count
            sign = "+" if delta > 0 else ""
            print(f"  {etype:<20} {sign}{delta:>5}  (base={b_count}, hybrid={h_count})")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = f"experiments/error_decomposition_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    # Remove examples from output to keep file small
    out_results = {}
    for k, v in results_all.items():
        out_v = {kk: vv for kk, vv in v.items() if kk != "error_examples"}
        out_results[k] = out_v

    with open(out_path, "w") as f:
        json.dump(out_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
