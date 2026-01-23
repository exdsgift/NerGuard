import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Visual")

# * Plots

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, id2label: Dict, title: str, filename: str
):
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    label_names = [id2label[i] for i in unique_labels]
    n_labels = len(unique_labels)

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Normalization
    cm_norm = np.divide(
        cm.astype("float"),
        cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1, keepdims=True) != 0,
    )

    plot_w = min(max(12, n_labels * 0.6), 30)
    plot_h = min(max(10, n_labels * 0.5), 25)

    plt.figure(figsize=(plot_w, plot_h))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 9},
        cbar_kws={"label": "Recall"},
    )

    plt.ylabel("True Label", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.title(title, pad=20, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved: {filename}")

def plot_entropy_separation(
    correct_entropies: np.ndarray,
    incorrect_entropies: np.ndarray,
    threshold: float,
    filename: str,
):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        correct_entropies, fill=True, color="green", label="Correct", clip=(0, 2.0)
    )
    sns.kdeplot(
        incorrect_entropies, fill=True, color="red", label="Incorrect", clip=(0, 2.0)
    )

    plt.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )

    plt.title("Entropy Distribution (Uncertainty Analysis)")
    plt.xlabel("Shannon Entropy")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(0, 1.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved: {filename}")

def plot_model_comparison(baseline_report: Dict, hybrid_report: Dict, filename: str):
    labels = [
        k
        for k in baseline_report.keys()
        if k not in ["accuracy", "macro avg", "weighted avg"] and k in hybrid_report
    ]

    base_f1 = [baseline_report[k]["f1-score"] for k in labels]
    hybrid_f1 = [hybrid_report[k]["f1-score"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(max(12, len(labels) * 0.4), 6))

    plt.bar(
        x - width / 2,
        base_f1,
        width,
        label="Baseline",
        color="lightgray",
        edgecolor="black",
    )
    plt.bar(
        x + width / 2,
        hybrid_f1,
        width,
        label="Hybrid (Ours)",
        color="#4c72b0",
        edgecolor="black",
    )

    plt.xlabel("Class")
    plt.ylabel("F1-Score")
    plt.title("Baseline vs Hybrid Performance")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend(loc="lower right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved: {filename}")
