"""
Plotting mixin for evaluation classes.

This module provides a mixin class with plotting methods that can be inherited
by evaluator classes.
"""

import os
import logging
from typing import List, Any, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

from src.visualization.style import COLORS, style_axis

logger = logging.getLogger(__name__)


class PlottingMixin:
    """
    Mixin class containing plotting logic for evaluation results.

    This class provides methods for visualizing:
    - Per-class accuracy improvements
    - Confidence and entropy distributions
    - LLM intervention impact
    - Error analysis
    - Confusion matrices

    Assumes access to self.config with:
    - output_dir: Directory to save plots
    - THRESHOLD_CONF: Confidence threshold
    - THRESHOLD_ENTROPY: Entropy threshold
    """

    def _save_and_close(self, filename: str) -> None:
        """Save figure to output directory and close."""
        try:
            output_path = os.path.join(self.config.output_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
        finally:
            plt.close()

    def _plot_per_class_improvements(self, res: Any) -> None:
        """
        Plot per-class accuracy comparison and improvement.

        Args:
            res: Results object with per_class_stats attribute
        """
        logger.info("Generating per-class improvement chart...")

        # Prepare data
        data = []
        for label in sorted(res.per_class_stats.keys()):
            stats = res.per_class_stats[label]
            if stats["total"] < 3:
                continue

            baseline_acc = (stats["baseline_correct"] / stats["total"]) * 100
            hybrid_acc = (stats["hybrid_correct"] / stats["total"]) * 100

            data.append({
                "class": label,
                "baseline": baseline_acc,
                "hybrid": hybrid_acc,
                "improvement": hybrid_acc - baseline_acc,
            })

        if not data:
            logger.warning("No sufficient class data for plotting.")
            return

        df = pd.DataFrame(data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.patch.set_facecolor("white")

        # 1. Accuracy Comparison
        x = np.arange(len(df))
        width = 0.4

        ax1.bar(
            x - width / 2, df["baseline"], width,
            label="Baseline", color=COLORS["baseline"], alpha=0.8, edgecolor="white"
        )
        ax1.bar(
            x + width / 2, df["hybrid"], width,
            label="Hybrid", color=COLORS["hybrid"], alpha=0.8, edgecolor="white"
        )

        style_axis(ax1, "Per-Class Accuracy Comparison", "", "Accuracy (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["class"], rotation=45, ha="right")
        ax1.legend(frameon=True, loc="lower right", edgecolor="#333333")
        ax1.set_ylim(bottom=0)

        # 2. Improvement (Diverging Bar Chart)
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in df["improvement"]]
        ax2.bar(x, df["improvement"], color=colors, alpha=0.8, edgecolor="white")

        style_axis(ax2, "Net Accuracy Improvement (Hybrid - Baseline)", "Entity Class", "Improvement (pp)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df["class"], rotation=45, ha="right")
        ax2.axhline(y=0, color="#333333", linewidth=1)

        for i, v in enumerate(df["improvement"]):
            offset = 1.5 if v >= 0 else -1.5
            ax2.text(i, v + offset, f"{v:+.1f}", ha="center", fontsize=9, fontweight="600")

        self._save_and_close("per_class_improvements.png")

    def _plot_confidence_entropy_analysis(self, res: Any) -> None:
        """
        Plot confidence and entropy distributions with LLM intervention analysis.

        Args:
            res: Results object with confidence_scores, entropy_scores, llm_interventions
        """
        logger.info("Generating confidence and entropy analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor("white")

        # 1. Confidence Distribution
        axes[0, 0].hist(
            res.confidence_scores, bins=40, alpha=0.8,
            color=COLORS["baseline"], edgecolor="white"
        )
        axes[0, 0].axvline(
            self.config.THRESHOLD_CONF, color=COLORS["negative"],
            linestyle="--", linewidth=2, label=f"Threshold ({self.config.THRESHOLD_CONF})"
        )
        style_axis(axes[0, 0], "Confidence Score Distribution", "Confidence Score", "Count")
        axes[0, 0].legend(frameon=True, edgecolor="#333333")

        # 2. Entropy Distribution
        axes[0, 1].hist(
            res.entropy_scores, bins=40, alpha=0.8,
            color=COLORS["accent_1"], edgecolor="white"
        )
        axes[0, 1].axvline(
            self.config.THRESHOLD_ENTROPY, color=COLORS["negative"],
            linestyle="--", linewidth=2, label=f"Threshold ({self.config.THRESHOLD_ENTROPY})"
        )
        style_axis(axes[0, 1], "Entropy Score Distribution", "Entropy Score", "Count")
        axes[0, 1].legend(frameon=True, edgecolor="#333333")

        # 3. Scatter: Intervention Quality
        if res.llm_interventions:
            helped = [i for i in res.llm_interventions if i["is_correct_after"] and not i["was_correct_before"]]
            hurt = [i for i in res.llm_interventions if not i["is_correct_after"] and i["was_correct_before"]]

            if helped:
                axes[1, 0].scatter(
                    [x["confidence"] for x in helped],
                    [x["entropy"] for x in helped],
                    c=COLORS["positive"], alpha=0.6, s=50, label="LLM Helped", edgecolors="white"
                )
            if hurt:
                axes[1, 0].scatter(
                    [x["confidence"] for x in hurt],
                    [x["entropy"] for x in hurt],
                    c=COLORS["negative"], alpha=0.6, s=50, label="LLM Hurt", edgecolors="white"
                )

            style_axis(axes[1, 0], "Intervention Quality vs. Uncertainty", "Confidence", "Entropy")
            axes[1, 0].axvline(self.config.THRESHOLD_CONF, color="#999999", linestyle="--", linewidth=1.5)
            axes[1, 0].axhline(self.config.THRESHOLD_ENTROPY, color="#999999", linestyle="--", linewidth=1.5)
            axes[1, 0].legend(loc="upper right", frameon=True, edgecolor="#333333")

        # 4. Success Rate by Quartile
        if res.llm_interventions:
            df = pd.DataFrame(res.llm_interventions)
            df["helped"] = (~df["was_correct_before"]) & df["is_correct_after"]
            df["hurt"] = df["was_correct_before"] & (~df["is_correct_after"])
            df["quartile"] = pd.qcut(df["confidence"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

            q_stats = df.groupby("quartile", observed=False)[["helped", "hurt"]].sum()
            x = np.arange(len(q_stats))
            width = 0.4

            axes[1, 1].bar(x - width / 2, q_stats["helped"], width, label="Helped", color=COLORS["positive"], alpha=0.8)
            axes[1, 1].bar(x + width / 2, q_stats["hurt"], width, label="Hurt", color=COLORS["negative"], alpha=0.8)

            style_axis(axes[1, 1], "LLM Impact by Confidence Quartile", "Confidence Quartile", "Count")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(q_stats.index)
            axes[1, 1].legend(frameon=True, edgecolor="#333333")

        self._save_and_close("confidence_entropy_analysis.png")

    def _plot_llm_impact_analysis(self, res: Any) -> None:
        """
        Plot LLM intervention impact analysis.

        Args:
            res: Results object with llm_calls, llm_corrections, etc.
        """
        logger.info("Generating LLM impact analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.patch.set_facecolor("white")

        # 1. Donut Chart: Call Distribution
        call_labels = ["LLM Called", "No LLM"]
        call_sizes = [res.llm_calls, len(res.baseline_preds) - res.llm_calls]

        wedges, texts, autotexts = axes[0, 0].pie(
            call_sizes, labels=call_labels, autopct="%1.1f%%", startangle=90,
            colors=[COLORS["accent_2"], COLORS["neutral"]],
            wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight("bold")
        axes[0, 0].set_title("LLM Call Distribution", fontsize=14, fontweight="bold", pad=12)

        # 2. Donut Chart: Outcomes
        if res.llm_calls > 0:
            out_labels = ["Fixed Error", "Induced Error", "Neutral"]
            out_sizes = [
                res.llm_corrections,
                res.llm_wrong_corrections,
                res.llm_calls - res.llm_corrections - res.llm_wrong_corrections,
            ]
            out_colors = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]

            wedges, texts, autotexts = axes[0, 1].pie(
                out_sizes, labels=out_labels, autopct="%1.1f%%", startangle=90,
                colors=out_colors,
                wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
            )
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_fontweight("bold")
            axes[0, 1].set_title("Intervention Outcomes", fontsize=14, fontweight="bold", pad=12)

        # 3. Horizontal Bar: Top Classes Called
        counts = Counter([i["true_label"] for i in res.llm_interventions])
        if counts:
            top_classes = dict(counts.most_common(10))
            y_pos = np.arange(len(top_classes))

            axes[1, 0].barh(y_pos, list(top_classes.values()), color=COLORS["baseline"], alpha=0.8, edgecolor="white")
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(top_classes.keys())
            axes[1, 0].invert_yaxis()
            style_axis(axes[1, 0], "Top 10 Classes: LLM Activity", "Count", "")

        # 4. Horizontal Bar: Net Impact
        net_impact = {}
        for label, stats in res.per_class_stats.items():
            if stats["total"] > 0:
                net_impact[label] = stats["llm_helped"] - stats["llm_hurt"]

        if net_impact:
            sorted_imp = dict(sorted(net_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
            colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in sorted_imp.values()]
            y_pos = np.arange(len(sorted_imp))

            axes[1, 1].barh(y_pos, list(sorted_imp.values()), color=colors, alpha=0.8, edgecolor="white")
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(sorted_imp.keys())
            axes[1, 1].axvline(0, color="#333333", linewidth=1)
            axes[1, 1].invert_yaxis()
            style_axis(axes[1, 1], "Top 10 Classes: Net Correction Impact", "Net Change", "")

        self._save_and_close("llm_impact_analysis.png")

    def _plot_error_analysis(self, y_true: List, y_base: List, y_hyb: List) -> None:
        """
        Plot error analysis comparing baseline and hybrid predictions.

        Args:
            y_true: True labels
            y_base: Baseline predictions
            y_hyb: Hybrid predictions
        """
        logger.info("Generating error analysis...")

        fixed_errors = []
        new_errors = []
        baseline_errors = []
        hybrid_errors = []

        for t, b, h in zip(y_true, y_base, y_hyb):
            b_wrong = b != t
            h_wrong = h != t

            if b_wrong and not h_wrong:
                fixed_errors.append(t)
            if h_wrong and not b_wrong:
                new_errors.append(t)
            if b_wrong:
                baseline_errors.append(t)
            if h_wrong:
                hybrid_errors.append(t)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor("white")

        # 1. Error Counts
        base_counts = Counter(baseline_errors)
        hyb_counts = Counter(hybrid_errors)

        top_base = [k for k, _ in base_counts.most_common(8)]
        top_hyb = [k for k, _ in hyb_counts.most_common(8)]
        target_classes = sorted(list(set(top_base + top_hyb)))

        x = np.arange(len(target_classes))
        width = 0.4

        axes[0, 0].bar(x - width / 2, [base_counts[c] for c in target_classes], width, label="Baseline", color=COLORS["baseline"], alpha=0.8)
        axes[0, 0].bar(x + width / 2, [hyb_counts[c] for c in target_classes], width, label="Hybrid", color=COLORS["hybrid"], alpha=0.8)

        style_axis(axes[0, 0], "Error Counts (Top Impacted Classes)", "", "Count")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(target_classes, rotation=45, ha="right")
        axes[0, 0].legend(frameon=True, edgecolor="#333333")

        # 2. Fixed vs Introduced
        fixed_c = Counter(fixed_errors)
        new_c = Counter(new_errors)
        changed_classes = sorted(
            list(set(list(fixed_c.keys()) + list(new_c.keys()))),
            key=lambda k: fixed_c[k] - new_c[k],
            reverse=True,
        )[:12]

        if changed_classes:
            x = np.arange(len(changed_classes))
            axes[0, 1].bar(x, [fixed_c[c] for c in changed_classes], label="Fixed", color=COLORS["positive"], alpha=0.8)
            axes[0, 1].bar(x, [-new_c[c] for c in changed_classes], label="New Error", color=COLORS["negative"], alpha=0.8)

            style_axis(axes[0, 1], "Error Flux: Fixed vs. Introduced", "", "Count")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(changed_classes, rotation=45, ha="right")
            axes[0, 1].axhline(0, color="#333333", linewidth=1)
            axes[0, 1].legend(frameon=True, edgecolor="#333333")

        # 3. Overall Metrics
        metrics = {
            "Baseline Errors": len(baseline_errors),
            "Hybrid Errors": len(hybrid_errors),
            "Fixed": len(fixed_errors),
            "New": len(new_errors),
        }
        colors_m = [COLORS["baseline"], COLORS["hybrid"], COLORS["positive"], COLORS["negative"]]
        bars = axes[1, 0].bar(metrics.keys(), metrics.values(), color=colors_m, alpha=0.8)
        style_axis(axes[1, 0], "Global Error Metrics", "", "Count")
        axes[1, 0].bar_label(bars, padding=3, fontsize=10, fontweight="bold")
        axes[1, 0].tick_params(axis="x", rotation=15)

        # 4. Error Rate Comparison
        total_counts = Counter(y_true)
        err_rates = {}
        for c in target_classes:
            if total_counts[c] > 0:
                err_rates[c] = {
                    "base": (base_counts[c] / total_counts[c]) * 100,
                    "hyb": (hyb_counts[c] / total_counts[c]) * 100,
                }

        if err_rates:
            x = np.arange(len(target_classes))
            axes[1, 1].bar(x - width / 2, [err_rates[c]["base"] for c in target_classes], width, label="Baseline", color=COLORS["baseline"], alpha=0.8)
            axes[1, 1].bar(x + width / 2, [err_rates[c]["hyb"] for c in target_classes], width, label="Hybrid", color=COLORS["hybrid"], alpha=0.8)

            style_axis(axes[1, 1], "Error Rate Percentage", "", "Error Rate (%)")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(target_classes, rotation=45, ha="right")
            axes[1, 1].legend(frameon=True, edgecolor="#333333")

        self._save_and_close("error_analysis.png")

    def _plot_confusion_matrices(
        self,
        y_true: List,
        y_base: List,
        y_hyb: List,
        labels: List,
    ) -> None:
        """
        Generate side-by-side confusion matrices for baseline and hybrid models.

        Args:
            y_true: True labels
            y_base: Baseline predictions
            y_hyb: Hybrid predictions
            labels: List of label names to include
        """
        if not labels:
            return

        logger.info("Generating confusion matrices...")
        fig, ax = plt.subplots(1, 2, figsize=(24, 10))
        fig.patch.set_facecolor("white")

        def plot_ax(y_t, y_p, axis, title, cmap):
            cm = confusion_matrix(y_t, y_p, labels=labels)
            with np.errstate(divide="ignore", invalid="ignore"):
                cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])

            sns.heatmap(
                cm_norm, annot=False, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                cmap=cmap, ax=axis, cbar_kws={"shrink": 0.85}
            )
            axis.set_title(title, fontsize=14, fontweight="bold")
            axis.set_xlabel("Predicted", fontsize=12)
            axis.set_ylabel("True", fontsize=12)

        plot_ax(y_true, y_base, ax[0], "Baseline Confusion Matrix", "Blues")
        plot_ax(y_true, y_hyb, ax[1], "Hybrid Confusion Matrix", "Greens")

        plt.tight_layout()
        self._save_and_close("confusion_matrix_comparison.png")
