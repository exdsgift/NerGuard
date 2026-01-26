"""
Plotting functions for NerGuard evaluation and analysis.

This module provides functions for creating evaluation visualizations.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.visualization.style import COLORS, style_axis

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    id2label: Dict[int, str],
    title: str = "Confusion Matrix",
    filename: Optional[str] = None,
    normalize: bool = True,
    figsize: Optional[tuple] = None,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot a confusion matrix for classification results.

    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs
        id2label: Mapping from label IDs to names
        title: Plot title
        filename: Path to save the figure (optional)
        normalize: Whether to normalize by true labels (recall)
        figsize: Figure size (auto-calculated if None)
        cmap: Colormap name

    Returns:
        Matplotlib figure object
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique labels
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    label_names = [id2label.get(i, f"UNK_{i}") for i in unique_labels]
    n_labels = len(unique_labels)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Normalize if requested
    if normalize:
        cm_display = np.divide(
            cm.astype("float"),
            cm.sum(axis=1, keepdims=True),
            out=np.zeros_like(cm, dtype=float),
            where=cm.sum(axis=1, keepdims=True) != 0,
        )
    else:
        cm_display = cm.astype(float)

    # Calculate figure size
    if figsize is None:
        plot_w = min(max(12, n_labels * 0.6), 30)
        plot_h = min(max(10, n_labels * 0.5), 25)
        figsize = (plot_w, plot_h)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 9},
        cbar_kws={"label": "Recall" if normalize else "Count"},
        ax=ax,
    )

    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_title(title, pad=20, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filename}")

    return fig


def plot_entropy_separation(
    correct_entropies: Union[np.ndarray, List],
    incorrect_entropies: Union[np.ndarray, List],
    threshold: float,
    filename: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot entropy distribution for correct vs incorrect predictions.

    Args:
        correct_entropies: Entropy values for correct predictions
        incorrect_entropies: Entropy values for incorrect predictions
        threshold: Decision threshold to display
        filename: Path to save the figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    correct_entropies = np.array(correct_entropies)
    incorrect_entropies = np.array(incorrect_entropies)

    fig, ax = plt.subplots(figsize=figsize)

    # KDE plots
    if len(correct_entropies) > 0:
        sns.kdeplot(
            correct_entropies,
            fill=True,
            color=COLORS["positive"],
            label="Correct",
            clip=(0, 2.0),
            ax=ax,
        )

    if len(incorrect_entropies) > 0:
        sns.kdeplot(
            incorrect_entropies,
            fill=True,
            color=COLORS["negative"],
            label="Incorrect",
            clip=(0, 2.0),
            ax=ax,
        )

    # Threshold line
    ax.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )

    style_axis(ax, "Entropy Distribution (Uncertainty Analysis)", "Shannon Entropy", "Density")
    ax.legend(frameon=True, edgecolor="#333333")
    ax.set_xlim(0, 1.5)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filename}")

    return fig


def plot_model_comparison(
    baseline_report: Dict,
    hybrid_report: Dict,
    filename: Optional[str] = None,
    metric: str = "f1-score",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot comparison of baseline vs hybrid model performance.

    Args:
        baseline_report: Classification report dict for baseline
        hybrid_report: Classification report dict for hybrid
        filename: Path to save the figure (optional)
        metric: Metric to compare ("f1-score", "precision", "recall")
        figsize: Figure size (auto-calculated if None)

    Returns:
        Matplotlib figure object
    """
    # Get common labels (excluding aggregates)
    exclude_keys = {"accuracy", "macro avg", "weighted avg"}
    labels = [
        k for k in baseline_report.keys()
        if k not in exclude_keys and k in hybrid_report
    ]

    base_values = [baseline_report[k][metric] for k in labels]
    hybrid_values = [hybrid_report[k][metric] for k in labels]

    if figsize is None:
        figsize = (max(12, len(labels) * 0.4), 6)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x - width / 2,
        base_values,
        width,
        label="Baseline",
        color=COLORS["baseline"],
        edgecolor="white",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        hybrid_values,
        width,
        label="Hybrid",
        color=COLORS["hybrid"],
        edgecolor="white",
        alpha=0.8,
    )

    style_axis(ax, f"Baseline vs Hybrid Performance ({metric})", "Class", metric.replace("-", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="lower right", frameon=True, edgecolor="#333333")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filename}")

    return fig


def plot_metrics_radar(
    metrics: Dict[str, Dict[str, float]],
    filename: Optional[str] = None,
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """
    Create a radar chart comparing metrics across multiple models.

    Args:
        metrics: Dict of model_name -> {metric_name: value}
        filename: Path to save the figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    model_names = list(metrics.keys())
    metric_names = list(metrics[model_names[0]].keys())
    n_metrics = len(metric_names)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = [COLORS["baseline"], COLORS["hybrid"], COLORS["accent_1"], COLORS["accent_2"]]

    for i, model_name in enumerate(model_names):
        values = [metrics[model_name][m] for m in metric_names]
        values += values[:1]  # Complete the loop

        color = colors[i % len(colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filename}")

    return fig


def plot_improvement_bars(
    labels: List[str],
    improvements: List[float],
    filename: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot improvement (or degradation) as a diverging bar chart.

    Args:
        labels: Class labels
        improvements: Improvement values (positive = better, negative = worse)
        filename: Path to save the figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in improvements]

    ax.bar(x, improvements, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(y=0, color="#333333", linewidth=1)

    style_axis(ax, "Performance Improvement (Hybrid - Baseline)", "Class", "Improvement (pp)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(improvements):
        offset = 1.5 if v >= 0 else -1.5
        ax.text(i, v + offset, f"{v:+.1f}", ha="center", fontsize=9, fontweight="600")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {filename}")

    return fig
