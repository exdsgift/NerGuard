"""Report generation for the NER PII benchmark.

Generates structured output files (JSON + TXT) per the experiment spec.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from src.benchmark.label_protocol import LabelOverlapReport
from src.benchmark.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)


def generate_experiment_output(
    system_name: str,
    dataset_name: str,
    metrics: BenchmarkMetrics,
    overlap_report: LabelOverlapReport,
    config: Dict,
    output_dir: str,
    timestamp: str,
) -> str:
    """Generate all output files for a single experiment.

    Returns the experiment directory path.
    """
    # Sanitize names for filesystem
    sys_clean = system_name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    ds_clean = dataset_name.lower().replace(" ", "_")
    exp_dir = os.path.join(output_dir, f"{sys_clean}_{ds_clean}")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. config.json
    config_data = {
        "system": system_name,
        "dataset": dataset_name,
        "timestamp": timestamp,
        **overlap_report.to_dict(),
        "benchmark_config": config,
    }
    _write_json(os.path.join(exp_dir, "config.json"), config_data)

    # 2. results.json
    results_data = {
        "system": system_name,
        "dataset": dataset_name,
        **metrics.to_results_dict(),
    }
    _write_json(os.path.join(exp_dir, "results.json"), results_data)

    # 3. results.txt
    _write_text_report(
        os.path.join(exp_dir, "results.txt"),
        system_name, dataset_name, metrics, overlap_report,
    )

    # 4. confusion_matrix.json
    if metrics.confusion_matrix:
        _write_json(os.path.join(exp_dir, "confusion_matrix.json"), metrics.confusion_matrix)

    # 5. per_entity_scores.json
    if metrics.per_entity_scores:
        _write_json(os.path.join(exp_dir, "per_entity_scores.json"), metrics.per_entity_scores)

    # 6. latency_samples.json
    if metrics.latency_samples:
        _write_json(os.path.join(exp_dir, "latency_samples.json"), {
            "unit": "ms",
            "n_samples": len(metrics.latency_samples),
            "values": metrics.latency_samples,
        })

    # 7. errors_fp.json
    if metrics.false_positives:
        _write_json(os.path.join(exp_dir, "errors_fp.json"), {
            "n_errors": len(metrics.false_positives),
            "errors": metrics.false_positives,
        })

    # 8. errors_fn.json
    if metrics.false_negatives:
        _write_json(os.path.join(exp_dir, "errors_fn.json"), {
            "n_errors": len(metrics.false_negatives),
            "errors": metrics.false_negatives,
        })

    logger.info(f"Experiment output saved to {exp_dir}")
    return exp_dir


def generate_summary(
    all_results: List[Dict],
    output_dir: str,
    timestamp: str,
) -> str:
    """Generate comparative summary across all experiments."""
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # comparison_table.json
    _write_json(os.path.join(summary_dir, "comparison_table.json"), all_results)

    # comparison_table.txt
    _write_summary_table(os.path.join(summary_dir, "comparison_table.txt"), all_results)

    logger.info(f"Summary saved to {summary_dir}")
    return summary_dir


def _write_json(path: str, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_text_report(
    path: str,
    system_name: str,
    dataset_name: str,
    metrics: BenchmarkMetrics,
    overlap_report: LabelOverlapReport,
) -> None:
    lines = [
        f"{'='*70}",
        f"NER PII Benchmark — {system_name} × {dataset_name}",
        f"{'='*70}",
        "",
        f"Tier: {overlap_report.tier}",
        f"Evaluated labels: {len(overlap_report.evaluated_labels)}",
        f"  System labels: {len(overlap_report.system_native_labels)}",
        f"  Dataset labels: {len(overlap_report.dataset_native_labels)}",
        f"  Mapping applied: {overlap_report.mapping_applied}",
        "",
        f"Samples: {metrics.n_samples}",
        f"Tokens: {metrics.n_tokens}",
        "",
        "--- Token-Level Metrics ---",
        f"  Precision (macro/micro/weighted): {metrics.precision_macro:.4f} / {metrics.precision_micro:.4f} / {metrics.precision_weighted:.4f}",
        f"  Recall    (macro/micro/weighted): {metrics.recall_macro:.4f} / {metrics.recall_micro:.4f} / {metrics.recall_weighted:.4f}",
        f"  F1        (macro/micro/weighted): {metrics.f1_macro:.4f} / {metrics.f1_micro:.4f} / {metrics.f1_weighted:.4f}",
        "",
        "--- Entity-Level Metrics (seqeval) ---",
        f"  Precision: {metrics.entity_precision:.4f}",
        f"  Recall:    {metrics.entity_recall:.4f}",
        f"  F1:        {metrics.entity_f1:.4f}",
        "",
        "--- Latency ---",
        f"  Mean:   {metrics.latency_mean_ms:.2f} ms",
        f"  Median: {metrics.latency_median_ms:.2f} ms",
        f"  P95:    {metrics.latency_p95_ms:.2f} ms",
        f"  P99:    {metrics.latency_p99_ms:.2f} ms",
        f"  Throughput: {metrics.throughput_samples_per_sec:.1f} samples/sec",
        "",
    ]

    # Per-entity scores
    if metrics.per_entity_scores:
        lines.append("--- Per-Entity F1 Scores ---")
        sorted_entities = sorted(
            metrics.per_entity_scores.items(),
            key=lambda x: x[1].get("f1", 0),
            reverse=True,
        )
        for entity, scores in sorted_entities:
            lines.append(
                f"  {entity:30s} P={scores['precision']:.4f}  R={scores['recall']:.4f}  "
                f"F1={scores['f1']:.4f}  (n={scores['support']})"
            )
        lines.append("")

    # Length buckets
    if metrics.per_length_bucket:
        lines.append("--- Per-Length Bucket ---")
        for bucket, data in metrics.per_length_bucket.items():
            lines.append(f"  {bucket:8s}: F1={data.get('f1_macro', 0):.4f} (n={data.get('n_samples', 0)})")
        lines.append("")

    # Error summary
    lines.append("--- Error Summary ---")
    lines.append(f"  False positives: {len(metrics.false_positives)}")
    lines.append(f"  False negatives: {len(metrics.false_negatives)}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def generate_per_entity_comparison(experiments_dir: str, output_dir: str) -> None:
    """Generate a per-entity cross-system comparison table.

    Reads per_entity_scores.json from all experiments in experiments_dir,
    groups by dataset, and produces side-by-side F1 comparison tables
    as both markdown and CSV files.
    """
    from pathlib import Path
    import csv

    exp_path = Path(experiments_dir)
    # Collect all per-entity scores grouped by (dataset, system)
    by_dataset: Dict[str, Dict[str, Dict]] = {}

    for pe_file in sorted(exp_path.glob("*/per_entity_scores.json")):
        # Load results.json to get system and dataset names
        results_file = pe_file.parent / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            results = json.load(f)
        system = results.get("system", "?")
        dataset = results.get("dataset", "?")

        with open(pe_file) as f:
            scores = json.load(f)

        by_dataset.setdefault(dataset, {})[system] = scores

    os.makedirs(output_dir, exist_ok=True)

    for dataset, systems_scores in by_dataset.items():
        systems = sorted(systems_scores.keys())
        # Collect all entity labels across all systems
        all_entities = sorted({
            entity for scores in systems_scores.values() for entity in scores
        })

        # Aggregate B- and I- into entity type for cleaner comparison
        entity_types = sorted({
            e.replace("B-", "").replace("I-", "") for e in all_entities if e != "O"
        })

        # Write markdown
        md_lines = [f"# Per-Entity F1 Comparison — {dataset}", ""]
        header = "| Entity |"
        sep = "|--------|"
        for sys in systems:
            short = sys.split("(")[0].strip()
            header += f" {short} |"
            sep += "--------|"
        md_lines.append(header)
        md_lines.append(sep)

        for entity_type in entity_types:
            row = f"| {entity_type} |"
            for sys in systems:
                scores = systems_scores[sys]
                # Average B- and I- F1 for this entity type
                f1_vals = []
                for prefix in ["B-", "I-"]:
                    key = f"{prefix}{entity_type}"
                    if key in scores and scores[key].get("support", 0) > 0:
                        f1_vals.append(scores[key]["f1"])
                if f1_vals:
                    avg_f1 = sum(f1_vals) / len(f1_vals)
                    row += f" {avg_f1:.3f} |"
                else:
                    row += " — |"
            md_lines.append(row)

        md_path = os.path.join(output_dir, f"per_entity_comparison_{dataset}.md")
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        # Write CSV
        csv_path = os.path.join(output_dir, f"per_entity_comparison_{dataset}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entity"] + systems)
            for entity_type in entity_types:
                row = [entity_type]
                for sys in systems:
                    scores = systems_scores[sys]
                    f1_vals = []
                    for prefix in ["B-", "I-"]:
                        key = f"{prefix}{entity_type}"
                        if key in scores and scores[key].get("support", 0) > 0:
                            f1_vals.append(scores[key]["f1"])
                    if f1_vals:
                        row.append(f"{sum(f1_vals) / len(f1_vals):.4f}")
                    else:
                        row.append("")
                writer.writerow(row)

        logger.info(f"Per-entity comparison for {dataset}: {md_path}, {csv_path}")


def _write_summary_table(path: str, all_results: List[Dict]) -> None:
    lines = [
        f"{'='*100}",
        "NER PII Benchmark — Comparative Summary",
        f"{'='*100}",
        "",
    ]

    # Group by dataset
    by_dataset = {}
    for r in all_results:
        ds = r.get("dataset", "?")
        by_dataset.setdefault(ds, []).append(r)

    for ds_name, results in by_dataset.items():
        lines.append(f"Dataset: {ds_name}")
        lines.append(f"{'-'*80}")

        header = f"{'System':<35} {'F1-macro':>10} {'F1-micro':>10} {'Entity-F1':>10} {'Latency(ms)':>12} {'Tier':>5}"
        lines.append(header)
        lines.append(f"{'-'*82}")

        for r in sorted(results, key=lambda x: x.get("f1_macro", 0), reverse=True):
            sys_name = r.get("system", "?")[:34]
            f1_macro = r.get("f1_macro", 0)
            f1_micro = r.get("f1_micro", 0)
            entity_f1 = r.get("entity_f1", 0)
            latency = r.get("latency_mean_ms", 0)
            tier = r.get("tier", 1)
            n_labels = r.get("n_evaluated_labels", 0)

            lines.append(
                f"{sys_name:<35} {f1_macro:>10.4f} {f1_micro:>10.4f} "
                f"{entity_f1:>10.4f} {latency:>12.2f} {tier:>5} ({n_labels} labels)"
            )

        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
