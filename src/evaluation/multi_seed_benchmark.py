"""
Multi-Seed Benchmark for NerGuard.

Runs evaluations with multiple random seeds to compute confidence intervals
and statistical significance tests for all metrics.

This addresses the academic rigor requirement:
- Replaces single-seed (42) evaluation with 5-seed mean ± 95% CI
- Adds paired t-test for statistical significance between models
- Computes bootstrap confidence intervals for all metrics
"""

import os
import sys
import json
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    LABEL_TO_UNIFIED,
)
from src.visualization.style import set_publication_style

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multi-seed configuration
SEEDS = [42, 123, 456, 789, 999]
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95


@dataclass
class SeedResult:
    """Results from a single seed evaluation."""
    seed: int
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    accuracy: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class MultiSeedResult:
    """Aggregated results across multiple seeds."""
    model_name: str
    seed_results: List[SeedResult]

    @property
    def n_seeds(self) -> int:
        return len(self.seed_results)

    def mean(self, metric: str) -> float:
        """Get mean of a metric across seeds."""
        values = [getattr(r, metric) for r in self.seed_results]
        return np.mean(values)

    def std(self, metric: str) -> float:
        """Get standard deviation of a metric across seeds."""
        values = [getattr(r, metric) for r in self.seed_results]
        return np.std(values, ddof=1)

    def ci_95(self, metric: str) -> Tuple[float, float]:
        """Get 95% confidence interval for a metric."""
        values = [getattr(r, metric) for r in self.seed_results]
        mean = np.mean(values)
        # Using t-distribution for small sample sizes
        t_value = stats.t.ppf(0.975, df=len(values) - 1)
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        margin = t_value * se
        return (mean - margin, mean + margin)

    def format_with_ci(self, metric: str, fmt: str = ".3f") -> str:
        """Format metric as 'mean [CI_low, CI_high]'."""
        mean = self.mean(metric)
        ci_low, ci_high = self.ci_95(metric)
        return f"{mean:{fmt}} [{ci_low:{fmt}}, {ci_high:{fmt}}]"

    def per_class_mean_ci(self, class_name: str) -> Tuple[float, float, float]:
        """Get mean and CI for per-class F1."""
        values = [r.per_class_f1.get(class_name, 0.0) for r in self.seed_results]
        mean = np.mean(values)
        t_value = stats.t.ppf(0.975, df=len(values) - 1)
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        margin = t_value * se
        return mean, mean - margin, mean + margin


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def paired_ttest(
    results_a: MultiSeedResult,
    results_b: MultiSeedResult,
    metric: str = "f1_macro",
) -> Dict[str, float]:
    """
    Perform paired t-test between two models.

    Args:
        results_a: Results from model A
        results_b: Results from model B
        metric: Metric to compare

    Returns:
        Dictionary with t-statistic, p-value, and significance
    """
    values_a = [getattr(r, metric) for r in results_a.seed_results]
    values_b = [getattr(r, metric) for r in results_b.seed_results]

    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    # Effect size (Cohen's d for paired samples)
    diff = np.array(values_a) - np.array(values_b)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "mean_diff": np.mean(diff),
    }


def bootstrap_ci(
    y_true: List[str],
    y_pred: List[str],
    metric_fn: callable,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        metric_fn: Function to compute metric (takes y_true, y_pred)
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (point_estimate, ci_low, ci_high)
    """
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    point_estimate = metric_fn(y_true, y_pred)

    bootstrap_scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(n, size=n, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_scores, alpha / 2 * 100)
    ci_high = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return point_estimate, ci_low, ci_high


class MultiSeedEvaluator:
    """
    Evaluator that runs benchmarks with multiple random seeds
    and computes confidence intervals for all metrics.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        data_path: str = DEFAULT_DATA_PATH,
        seeds: List[int] = None,
        sample_limit: Optional[int] = None,
        output_dir: str = "./plots/multi_seed_results",
    ):
        """
        Initialize the multi-seed evaluator.

        Args:
            model_path: Path to the trained model
            data_path: Path to the evaluation dataset
            seeds: List of random seeds to use
            sample_limit: Maximum samples to evaluate (None = all)
            output_dir: Directory for output files
        """
        self.model_path = model_path
        self.data_path = data_path
        self.seeds = seeds or SEEDS
        self.sample_limit = sample_limit
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Seeds: {self.seeds}")

    def _load_model(self) -> Tuple[Any, Any, Dict]:
        """Load model, tokenizer, and id2label mapping."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        model = model.to(self.device)
        model.eval()
        id2label = AutoConfig.from_pretrained(self.model_path).id2label
        return model, tokenizer, id2label

    def _load_dataset(self, seed: int):
        """Load and optionally sample the dataset with given seed."""
        set_seed(seed)
        dataset = load_from_disk(self.data_path)["validation"]

        if self.sample_limit:
            indices = np.random.permutation(len(dataset))[: self.sample_limit]
            dataset = dataset.select(indices)

        return dataset

    def _evaluate_single_seed(
        self,
        model,
        tokenizer,
        id2label: Dict,
        seed: int,
    ) -> SeedResult:
        """Run evaluation for a single seed."""
        set_seed(seed)
        dataset = self._load_dataset(seed)

        y_true_all = []
        y_pred_all = []
        latencies = []

        for sample in tqdm(dataset, desc=f"Seed {seed}", leave=False):
            input_ids = sample["input_ids"]
            labels = sample["labels"]

            input_ids_tensor = torch.tensor([input_ids]).to(self.device)
            attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)

            t0 = time.time()
            with torch.no_grad():
                logits = model(input_ids_tensor, attention_mask=attention_mask).logits[0]

            probs = F.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            latencies.append((time.time() - t0) * 1000)

            for i, lbl_id in enumerate(labels):
                if lbl_id == -100:
                    continue

                gt_lbl = id2label[lbl_id].replace("B-", "").replace("I-", "").upper()
                pred_lbl = id2label[pred_ids[i].item()].replace("B-", "").replace("I-", "").upper()

                y_true_all.append(LABEL_TO_UNIFIED.get(gt_lbl, "O"))
                y_pred_all.append(LABEL_TO_UNIFIED.get(pred_lbl, "O"))

        # Compute metrics
        report = classification_report(
            y_true_all, y_pred_all, output_dict=True, zero_division=0
        )

        # Per-class F1
        per_class_f1 = {}
        for label in set(y_true_all):
            if label != "O" and label in report:
                per_class_f1[label] = report[label]["f1-score"]

        return SeedResult(
            seed=seed,
            f1_macro=report["macro avg"]["f1-score"],
            f1_weighted=report["weighted avg"]["f1-score"],
            precision_macro=report["macro avg"]["precision"],
            recall_macro=report["macro avg"]["recall"],
            accuracy=accuracy_score(y_true_all, y_pred_all),
            per_class_f1=per_class_f1,
            latency_ms=np.mean(latencies),
        )

    def run_multi_seed_evaluation(self, model_name: str = "NerGuard") -> MultiSeedResult:
        """
        Run evaluation across all seeds and aggregate results.

        Args:
            model_name: Name identifier for the model

        Returns:
            MultiSeedResult with aggregated metrics and CIs
        """
        logger.info(f"Running multi-seed evaluation for {model_name}")
        logger.info(f"Seeds: {self.seeds}")

        model, tokenizer, id2label = self._load_model()

        seed_results = []
        for seed in self.seeds:
            logger.info(f"Evaluating with seed {seed}...")
            result = self._evaluate_single_seed(model, tokenizer, id2label, seed)
            seed_results.append(result)
            logger.info(f"  F1 Macro: {result.f1_macro:.4f}")

        return MultiSeedResult(model_name=model_name, seed_results=seed_results)

    def generate_report(
        self,
        results: MultiSeedResult,
        comparison_results: Optional[MultiSeedResult] = None,
    ) -> str:
        """
        Generate comprehensive multi-seed evaluation report.

        Args:
            results: Primary model results
            comparison_results: Optional comparison model results

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"MULTI-SEED EVALUATION REPORT: {results.model_name}")
        lines.append(f"Seeds: {[r.seed for r in results.seed_results]}")
        lines.append("=" * 80)
        lines.append("")

        # Main metrics with CI
        lines.append("MAIN METRICS (Mean ± 95% CI)")
        lines.append("-" * 40)
        lines.append(f"F1 Macro:     {results.format_with_ci('f1_macro')}")
        lines.append(f"F1 Weighted:  {results.format_with_ci('f1_weighted')}")
        lines.append(f"Precision:    {results.format_with_ci('precision_macro')}")
        lines.append(f"Recall:       {results.format_with_ci('recall_macro')}")
        lines.append(f"Accuracy:     {results.format_with_ci('accuracy')}")
        lines.append(f"Latency (ms): {results.format_with_ci('latency_ms', '.1f')}")
        lines.append("")

        # Per-seed breakdown
        lines.append("PER-SEED RESULTS")
        lines.append("-" * 40)
        lines.append(f"{'Seed':<8} {'F1 Macro':<10} {'F1 Weighted':<12} {'Precision':<10} {'Recall':<10}")
        for r in results.seed_results:
            lines.append(
                f"{r.seed:<8} {r.f1_macro:.4f}     {r.f1_weighted:.4f}       "
                f"{r.precision_macro:.4f}     {r.recall_macro:.4f}"
            )
        lines.append("")

        # Per-class F1 with CI
        lines.append("PER-CLASS F1 (Mean ± 95% CI)")
        lines.append("-" * 40)
        all_classes = set()
        for r in results.seed_results:
            all_classes.update(r.per_class_f1.keys())

        for cls in sorted(all_classes):
            mean, ci_low, ci_high = results.per_class_mean_ci(cls)
            lines.append(f"{cls:<20} {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        lines.append("")

        # Statistical comparison
        if comparison_results:
            lines.append("=" * 80)
            lines.append(f"STATISTICAL COMPARISON: {results.model_name} vs {comparison_results.model_name}")
            lines.append("=" * 80)
            lines.append("")

            ttest = paired_ttest(results, comparison_results, "f1_macro")
            lines.append("Paired t-test on F1 Macro:")
            lines.append(f"  t-statistic: {ttest['t_statistic']:.4f}")
            lines.append(f"  p-value:     {ttest['p_value']:.4f}")
            lines.append(f"  Cohen's d:   {ttest['cohens_d']:.4f}")
            lines.append(f"  Mean diff:   {ttest['mean_diff']:.4f}")
            lines.append(f"  Significant (α=0.05): {'Yes' if ttest['significant_05'] else 'No'}")
            lines.append(f"  Significant (α=0.01): {'Yes' if ttest['significant_01'] else 'No'}")
            lines.append("")

        report_text = "\n".join(lines)

        # Save report
        report_path = os.path.join(self.output_dir, "multi_seed_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Report saved to {report_path}")

        # Save JSON for later use
        json_path = os.path.join(self.output_dir, "multi_seed_results.json")
        json_data = {
            "model_name": results.model_name,
            "seeds": self.seeds,
            "metrics": {
                "f1_macro": {
                    "mean": results.mean("f1_macro"),
                    "std": results.std("f1_macro"),
                    "ci_95": results.ci_95("f1_macro"),
                },
                "f1_weighted": {
                    "mean": results.mean("f1_weighted"),
                    "std": results.std("f1_weighted"),
                    "ci_95": results.ci_95("f1_weighted"),
                },
                "precision_macro": {
                    "mean": results.mean("precision_macro"),
                    "std": results.std("precision_macro"),
                    "ci_95": results.ci_95("precision_macro"),
                },
                "recall_macro": {
                    "mean": results.mean("recall_macro"),
                    "std": results.std("recall_macro"),
                    "ci_95": results.ci_95("recall_macro"),
                },
            },
            "per_seed": [
                {
                    "seed": r.seed,
                    "f1_macro": r.f1_macro,
                    "f1_weighted": r.f1_weighted,
                    "precision_macro": r.precision_macro,
                    "recall_macro": r.recall_macro,
                    "accuracy": r.accuracy,
                    "latency_ms": r.latency_ms,
                }
                for r in results.seed_results
            ],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

        return report_text


def main():
    """Run multi-seed benchmark evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-seed NerGuard evaluation")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Max samples per seed (None = all)",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots/multi_seed_results",
        help="Output directory",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Random seeds to use",
    )
    args = parser.parse_args()

    evaluator = MultiSeedEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        seeds=args.seeds,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
    )

    results = evaluator.run_multi_seed_evaluation()
    report = evaluator.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
