"""
Benchmark plotting functions for NerGuard evaluation.

This module provides visualization functions for model comparison benchmarks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Optional

from src.visualization.style import set_publication_style, get_color_palette


def plot_main_metrics(
    df_metrics: pd.DataFrame,
    output_dir: str,
    filename: str = "main_metrics.png",
) -> None:
    """
    Plot main performance metrics (F1-Score and Latency) as horizontal bar charts.

    Args:
        df_metrics: DataFrame with columns 'Model', 'F1-Score', 'Latency (ms)'
        output_dir: Output directory for saving the plot
        filename: Output filename
    """
    set_publication_style()

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    model_colors = _get_model_palette(df_metrics["Model"])

    # F1-Score plot
    df_f1 = df_metrics.sort_values("F1-Score")
    colors_f1 = [model_colors[m] for m in df_f1["Model"]]

    bars1 = ax1.barh(
        range(len(df_f1)),
        df_f1["F1-Score"],
        color=colors_f1,
        edgecolor="#2C2C2C",
        linewidth=1.2,
        height=0.65,
        alpha=0.9,
    )

    ax1.set_yticks(range(len(df_f1)))
    ax1.set_yticklabels(df_f1["Model"], fontsize=11)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("F1-Score", fontsize=12, fontweight="bold")
    ax1.set_title("Model Performance (F1-Score)", pad=12, fontsize=14, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, linewidth=0.8)

    for i, (bar, val) in enumerate(zip(bars1, df_f1["F1-Score"])):
        ax1.text(
            val + 0.015, i, f"{val:.3f}",
            va="center", ha="left", fontsize=10, fontweight="bold", color="#333333",
        )

    # Latency plot
    df_lat = df_metrics.sort_values("Latency (ms)", ascending=False)
    colors_lat = [model_colors[m] for m in df_lat["Model"]]

    bars2 = ax2.barh(
        range(len(df_lat)),
        df_lat["Latency (ms)"],
        color=colors_lat,
        edgecolor="#2C2C2C",
        linewidth=1.2,
        height=0.65,
        alpha=0.9,
    )

    ax2.set_yticks(range(len(df_lat)))
    ax2.set_yticklabels(df_lat["Model"], fontsize=11)
    ax2.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax2.set_title("Inference Latency", pad=12, fontsize=14, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, linewidth=0.8)

    max_lat = df_lat["Latency (ms)"].max()
    for i, (bar, val) in enumerate(zip(bars2, df_lat["Latency (ms)"])):
        ax2.text(
            val + (max_lat * 0.02), i, f"{val:.1f}",
            va="center", ha="left", fontsize=10, fontweight="bold", color="#333333",
        )

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", facecolor="white", edgecolor="none")
    plt.close()


def plot_efficiency_frontier(
    df_metrics: pd.DataFrame,
    output_dir: str,
    filename: str = "efficiency_frontier.png",
) -> None:
    """
    Plot efficiency frontier showing latency vs performance trade-off.

    Args:
        df_metrics: DataFrame with columns 'Model', 'F1-Score', 'Latency (ms)'
        output_dir: Output directory
        filename: Output filename
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(11, 7))
    df_sorted = df_metrics.sort_values("Latency (ms)")

    colors_list = get_color_palette("categorical", len(df_sorted))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]

    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        x, y = row["Latency (ms)"], row["F1-Score"]
        color = colors_list[i % len(colors_list)]
        marker = markers[i % len(markers)]

        ax.scatter(
            x, y, s=350, c=color, marker=marker,
            edgecolors="#2C2C2C", linewidth=2, alpha=0.85,
            label=row["Model"], zorder=5,
        )

        y_offset = -30 if y > df_sorted["F1-Score"].median() else 30
        va = "top" if y > df_sorted["F1-Score"].median() else "bottom"

        ax.annotate(
            row["Model"], xy=(x, y), xytext=(0, y_offset),
            textcoords="offset points", fontsize=10, fontweight="bold",
            ha="center", va=va, zorder=6, color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.8),
        )

    ax.set_xscale("log")
    ax.set_ylim(
        max(0, df_sorted["F1-Score"].min() - 0.08),
        min(1.05, df_sorted["F1-Score"].max() + 0.08),
    )

    ax.set_title("Efficiency Frontier: Latency vs Performance", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Latency (ms) - Log Scale", fontsize=13)
    ax.set_ylabel("F1-Score", fontsize=13)
    ax.legend(
        loc="best", frameon=True, facecolor="white", framealpha=0.95,
        edgecolor="#CCCCCC", fontsize=9.5, ncol=1 if len(df_sorted) <= 5 else 2,
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", facecolor="white")
    plt.close()


def plot_entity_comparison(
    results: List[Dict],
    output_dir: str,
    filename: str = "entity_comparison.png",
) -> None:
    """
    Plot performance comparison by entity type across models.

    Args:
        results: List of result dicts with 'model', 'y_true', 'y_pred' keys
        output_dir: Output directory
        filename: Output filename
    """
    set_publication_style()

    entity_data = []
    for res in results:
        report = classification_report(
            res["y_true"], res["y_pred"], output_dict=True, zero_division=0
        )
        for label, scores in report.items():
            if label not in ["accuracy", "macro avg", "weighted avg", "O"]:
                entity_data.append({
                    "Model": res["model"],
                    "Entity": label,
                    "F1-Score": scores["f1-score"],
                })

    if not entity_data:
        return

    df_ent = pd.DataFrame(entity_data)
    fig, ax = plt.subplots(figsize=(13, 6.5))

    palette = get_color_palette("categorical", len(results))

    sns.barplot(
        data=df_ent, x="Entity", y="F1-Score", hue="Model",
        palette=palette, edgecolor="#2C2C2C", linewidth=1.2, alpha=0.9, ax=ax,
    )

    ax.set_title("Performance Breakdown by Entity Type", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Entity Type", fontsize=13)
    ax.set_ylabel("F1-Score", fontsize=13)
    ax.set_ylim(0, 1.05)

    ax.legend(
        title="Model", title_fontsize=11, fontsize=10,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#CCCCCC",
    )

    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{filename}", facecolor="white")
    plt.close()


def plot_confusion_matrix_single(
    y_true: List[str],
    y_pred: List[str],
    model_name: str,
    output_dir: str,
) -> None:
    """
    Plot confusion matrix for a single model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name for title
        output_dir: Output directory
    """
    set_publication_style()

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])

    size = max(8, min(12, len(labels) * 0.7))
    fig, ax = plt.subplots(figsize=(size, size * 0.95))

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=1.5, linecolor="white", square=True,
        annot_kws={"size": 9},
        cbar_kws={"label": "Normalized Frequency", "shrink": 0.8},
        vmin=0, vmax=1, ax=ax,
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    safe_name = model_name.replace(" ", "_").replace("/", "-")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/cm_{safe_name}.png", facecolor="white")
    plt.close()


def _get_model_palette(models) -> Dict[str, str]:
    """Get color palette for models."""
    unique_models = sorted(list(set(models)))
    colors = get_color_palette("categorical", len(unique_models))
    return dict(zip(unique_models, colors))
