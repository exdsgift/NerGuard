"""
Optimization plotting functions for NerGuard.

This module provides visualization functions for threshold optimization
and model quantization analysis.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from src.visualization.style import set_publication_style, COLORS, style_axis


def plot_optimization_heatmap(
    ent_grid: np.ndarray,
    conf_grid: np.ndarray,
    matrix: np.ndarray,
    best_idx: Tuple[int, int],
    metric_name: str,
    output_path: str,
    beta_score: float = 0.5,
) -> None:
    """
    Plot optimization surface heatmap.

    Args:
        ent_grid: Entropy threshold grid
        conf_grid: Confidence threshold grid
        matrix: 2D matrix of metric values
        best_idx: (i, j) index of optimal point
        metric_name: Name of the metric being plotted
        output_path: Full path to save the figure
        beta_score: F-beta score parameter
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    X, Y = np.meshgrid(conf_grid, ent_grid)
    mesh = ax.pcolormesh(X, Y, matrix, shading="auto", cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax, label=metric_name)
    cbar.ax.tick_params(labelsize=9)

    best_i, best_j = best_idx
    best_ent = ent_grid[best_i]
    best_conf = conf_grid[best_j]
    ax.scatter(
        best_conf, best_ent,
        color=COLORS["negative"], s=200, edgecolors="white", linewidths=2,
        label="Optimal Point", marker="*", zorder=5,
    )

    ax.set_title(
        f"Optimization Surface: {metric_name}\n(β={beta_score} F-Score Optimization)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Confidence Threshold (trigger if < x)", fontsize=12)
    ax.set_ylabel("Entropy Threshold (trigger if > y)", fontsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#333333")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def plot_pareto_frontier(
    results: Dict[str, np.ndarray],
    ent_grid: np.ndarray,
    conf_grid: np.ndarray,
    output_path: str,
) -> None:
    """
    Plot Pareto frontier for precision-recall trade-off.

    Args:
        results: Dict containing 'precision' and 'recall' matrices
        ent_grid: Entropy threshold grid
        conf_grid: Confidence threshold grid
        output_path: Full path to save the figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(10, 7))

    precision_flat = results["precision"].flatten()
    recall_flat = results["recall"].flatten()

    valid = (precision_flat > 0) | (recall_flat > 0)
    precision_flat = precision_flat[valid]
    recall_flat = recall_flat[valid]

    ax.scatter(
        recall_flat, precision_flat,
        alpha=0.4, s=25, c=COLORS["baseline"],
        edgecolors="white", linewidth=0.3,
        label="All configurations", zorder=3,
    )

    # Compute Pareto frontier correctly:
    # Sort by recall DESCENDING, track max precision seen
    # A point is Pareto-optimal if its precision > max seen so far
    sorted_indices = np.argsort(recall_flat)[::-1]
    pareto_indices = []
    max_precision = -np.inf

    for idx in sorted_indices:
        if precision_flat[idx] > max_precision:
            pareto_indices.append(idx)
            max_precision = precision_flat[idx]

    if pareto_indices:
        pareto_recall = recall_flat[pareto_indices]
        pareto_precision = precision_flat[pareto_indices]
        # Sort by recall ascending for proper line plotting
        sort_order = np.argsort(pareto_recall)
        pareto_recall = pareto_recall[sort_order]
        pareto_precision = pareto_precision[sort_order]

        ax.plot(
            pareto_recall, pareto_precision,
            color=COLORS["negative"], linewidth=2.5, linestyle="-",
            marker="o", markersize=6, markerfacecolor=COLORS["negative"],
            markeredgecolor="white", markeredgewidth=1.5,
            label="Pareto Frontier", zorder=10,
        )

    ax.set_xlabel("Recall (Error Detection Rate)", fontsize=12)
    ax.set_ylabel("Precision (Trigger Accuracy)", fontsize=12)
    ax.set_title(
        "Precision-Recall Tradeoff\n(Pareto-Optimal Configurations)",
        fontsize=14, fontweight="bold", pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#333333")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def plot_quantization_metrics(
    metrics: Dict[str, Dict],
    output_dir: str,
    filename: str = "unified_metrics.png",
) -> None:
    """
    Plot quantization comparison metrics (FP32 vs INT8).

    Args:
        metrics: Dict mapping model name to metric dict
        output_dir: Output directory
        filename: Output filename
    """
    set_publication_style()

    models = list(metrics.keys())
    colors = ["#b8b8b8", "#707070"]
    edge_colors = ["#0d0d0d", "#0d0d0d"]

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(
        2, 3, hspace=0.28, wspace=0.3, left=0.08, right=0.96, top=0.89, bottom=0.08
    )

    fig.suptitle(
        "Model Quantization: FP32 vs INT8 Comprehensive Analysis",
        fontsize=22, fontweight="bold", y=0.97,
    )

    def add_value_labels(ax, rects, format_str):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                format_str.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontweight="bold", fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="#555555", alpha=0.9, linewidth=1.2,
                ),
            )

    # Row 1: Performance Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    f1_scores = [metrics[m]["f1"] for m in models]
    bars1 = ax1.bar(models, f1_scores, color=colors, edgecolor=edge_colors, linewidth=2.5, width=0.6, alpha=0.9)
    ax1.set_title("F1 Score", fontsize=18, pad=14)
    ax1.set_ylabel("Weighted F1-Score", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax1, bars1, "{:.4f}")

    ax2 = fig.add_subplot(gs[0, 1])
    precisions = [metrics[m]["precision"] for m in models]
    bars2 = ax2.bar(models, precisions, color=colors, edgecolor=edge_colors, linewidth=2.5, width=0.6, alpha=0.9)
    ax2.set_title("Precision", fontsize=18, pad=14)
    ax2.set_ylabel("Weighted Precision", fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax2, bars2, "{:.4f}")

    ax3 = fig.add_subplot(gs[0, 2])
    recalls = [metrics[m]["recall"] for m in models]
    bars3 = ax3.bar(models, recalls, color=colors, edgecolor=edge_colors, linewidth=2.5, width=0.6, alpha=0.9)
    ax3.set_title("Recall", fontsize=18, pad=14)
    ax3.set_ylabel("Weighted Recall", fontsize=14)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax3, bars3, "{:.4f}")

    # Row 2: Efficiency Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    latencies = [metrics[m]["latency"] for m in models]
    bars4 = ax4.bar(models, latencies, color=colors, edgecolor=edge_colors, linewidth=2.5, width=0.6, alpha=0.9)
    ax4.set_title("Inference Latency", fontsize=18, pad=14)
    ax4.set_ylabel("Latency (ms)", fontsize=14)
    ax4.set_ylim(0, max(latencies) * 1.25)
    ax4.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax4, bars4, "{:.2f} ms")

    speedup = latencies[0] / latencies[1] if latencies[1] > 0 else 1
    speedup_text = "faster" if latencies[1] < latencies[0] else "slower"
    ax4.text(0.5, -0.14, f"INT8 is {speedup:.2f}x {speedup_text}", transform=ax4.transAxes, ha="center", fontsize=12, style="italic", color="#333333")

    ax5 = fig.add_subplot(gs[1, 1])
    sizes = [metrics[m]["size"] for m in models]
    bars5 = ax5.bar(models, sizes, color=colors, edgecolor=edge_colors, linewidth=2.5, width=0.6, alpha=0.9)
    ax5.set_title("Model Size", fontsize=18, pad=14)
    ax5.set_ylabel("Size (MB)", fontsize=14)
    ax5.set_ylim(0, max(sizes) * 1.25)
    ax5.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax5, bars5, "{:.1f} MB")

    reduction = (1 - sizes[1] / sizes[0]) * 100
    compression_ratio = sizes[0] / sizes[1] if sizes[1] > 0 else 1
    ax5.text(0.5, -0.14, f"{compression_ratio:.2f}x compression ({reduction:.1f}% reduction)", transform=ax5.transAxes, ha="center", fontsize=12, style="italic", color="#333333")

    # Accuracy Difference
    ax6 = fig.add_subplot(gs[1, 2])
    f1_diff = metrics[models[1]]["f1"] - metrics[models[0]]["f1"]
    precision_diff = metrics[models[1]]["precision"] - metrics[models[0]]["precision"]
    recall_diff = metrics[models[1]]["recall"] - metrics[models[0]]["recall"]

    metric_names = ["F1", "Precision", "Recall"]
    diffs = [f1_diff, precision_diff, recall_diff]
    diff_colors = [colors[0] if d >= 0 else "#C85450" for d in diffs]
    diff_edges = [edge_colors[0] if d >= 0 else "#A04340" for d in diffs]

    bars6 = ax6.bar(metric_names, diffs, color=diff_colors, edgecolor=diff_edges, linewidth=2.5, width=0.6, alpha=0.9)
    ax6.set_title("Accuracy Change (INT8 - FP32)", fontsize=18, pad=14)
    ax6.set_ylabel("Absolute Difference", fontsize=14)

    max_abs_diff = max(abs(min(diffs)), abs(max(diffs)))
    y_margin = max_abs_diff * 0.3
    ax6.set_ylim(-max_abs_diff - y_margin, max_abs_diff + y_margin)
    ax6.axhline(y=0, color="#333333", linestyle="-", linewidth=1.5, alpha=0.8)
    ax6.grid(True, alpha=0.35, axis="y", linewidth=0.8)

    for bar, val in zip(bars6, diffs):
        height = bar.get_height()
        y_offset = 6 if height >= 0 else -16
        va = "bottom" if height >= 0 else "top"
        ax6.annotate(
            f"{val:+.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, y_offset), textcoords="offset points",
            ha="center", va=va, fontweight="bold", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#555555", alpha=0.9, linewidth=1.2),
        )

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_quantization_radar(
    metrics: Dict[str, Dict],
    output_dir: str,
    filename: str = "overall_comparison.png",
) -> None:
    """
    Plot radar chart comparing FP32 vs INT8 across multiple dimensions.

    Args:
        metrics: Dict mapping model name to metric dict
        output_dir: Output directory
        filename: Output filename
    """
    set_publication_style()

    models = list(metrics.keys())
    colors = ["#3498db", "#e74c3c"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    categories = ["Latency\n(inv)", "Size\n(inv)", "F1", "Precision", "Recall"]
    N = len(categories)

    latencies = [metrics[m]["latency"] for m in models]
    sizes = [metrics[m]["size"] for m in models]
    f1_scores = [metrics[m]["f1"] for m in models]
    precisions = [metrics[m]["precision"] for m in models]
    recalls = [metrics[m]["recall"] for m in models]

    # Normalize metrics (invert latency and size)
    values_fp32 = [1 / latencies[0] * 100, 1 / sizes[0] * 10, f1_scores[0], precisions[0], recalls[0]]
    values_int8 = [1 / latencies[1] * 100, 1 / sizes[1] * 10, f1_scores[1], precisions[1], recalls[1]]

    # Normalize to 0-1
    max_vals = [max(values_fp32[i], values_int8[i]) for i in range(N)]
    values_fp32_norm = [values_fp32[i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(N)]
    values_int8_norm = [values_int8[i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(N)]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_fp32_norm += values_fp32_norm[:1]
    values_int8_norm += values_int8_norm[:1]
    angles += angles[:1]

    ax.plot(angles, values_fp32_norm, "o-", linewidth=2.5, label="FP32", color=colors[0], markersize=7)
    ax.fill(angles, values_fp32_norm, alpha=0.25, color=colors[0])
    ax.plot(angles, values_int8_norm, "o-", linewidth=2.5, label="INT8", color=colors[1], markersize=7)
    ax.fill(angles, values_int8_norm, alpha=0.25, color=colors[1])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_title("Overall Comparison: FP32 vs INT8", fontweight="bold", pad=20, fontsize=16, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def save_quantization_report(
    metrics: Dict[str, Dict],
    output_dir: str,
    filename: str = "report.txt",
) -> str:
    """
    Save a detailed quantization report.

    Args:
        metrics: Dict mapping model name to metric dict
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to the saved report
    """
    models = list(metrics.keys())
    fp32_metrics = metrics[models[0]]
    int8_metrics = metrics[models[1]]

    latency_ratio = fp32_metrics["latency"] / int8_metrics["latency"]
    size_reduction = (1 - int8_metrics["size"] / fp32_metrics["size"]) * 100
    f1_diff = int8_metrics["f1"] - fp32_metrics["f1"]
    f1_diff_pct = (f1_diff / fp32_metrics["f1"]) * 100
    precision_diff = int8_metrics["precision"] - fp32_metrics["precision"]
    precision_diff_pct = (precision_diff / fp32_metrics["precision"]) * 100
    recall_diff = int8_metrics["recall"] - fp32_metrics["recall"]
    recall_diff_pct = (recall_diff / fp32_metrics["recall"]) * 100
    compression_ratio = fp32_metrics["size"] / int8_metrics["size"]
    accuracy_retention = (int8_metrics["f1"] / fp32_metrics["f1"]) * 100

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("QUANTIZATION REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("CALCULATED STATISTICS:\n")
        f.write(f"Latency Ratio (FP32/INT8): {latency_ratio:.3f}x\n")
        f.write(f"Size Reduction: {size_reduction:.2f}%\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}:1\n")
        f.write(f"F1 Difference: {f1_diff:+.6f} ({f1_diff_pct:+.2f}%)\n")
        f.write(f"Precision Difference: {precision_diff:+.6f} ({precision_diff_pct:+.2f}%)\n")
        f.write(f"Recall Difference: {recall_diff:+.6f} ({recall_diff_pct:+.2f}%)\n")
        f.write(f"Accuracy Retention: {accuracy_retention:.2f}%\n")

    return report_path
