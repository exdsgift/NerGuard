"""Generate training curve plots from WandB output logs."""

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

WANDB_LOG = Path(
    "wandb/run-20251212_160605-p37705mz/files/output.log"
)
OUTPUT_DIR = Path("plots/training_curves")


def parse_log(log_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse WandB output.log into training and eval metric dicts."""
    train_metrics = []
    eval_metrics = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                entry = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue

            if "eval_loss" in entry:
                eval_metrics.append(entry)
            elif "loss" in entry and "epoch" in entry:
                train_metrics.append(entry)

    return train_metrics, eval_metrics


def smooth(values: list[float], weight: float = 0.9) -> list[float]:
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_loss(train_metrics: list[dict], eval_metrics: list[dict]) -> None:
    """Plot training and evaluation loss vs epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))

    epochs = [m["epoch"] for m in train_metrics]
    losses = [m["loss"] for m in train_metrics]
    smoothed_losses = smooth(losses, weight=0.85)

    ax.plot(epochs, losses, alpha=0.25, color="#2196F3", linewidth=0.8)
    ax.plot(
        epochs, smoothed_losses, color="#1565C0", linewidth=2,
        label="Training loss (smoothed)"
    )

    eval_epochs = [m["epoch"] for m in eval_metrics]
    eval_losses = [m["eval_loss"] for m in eval_metrics]
    ax.plot(
        eval_epochs, eval_losses, "D-", color="#E53935", markersize=6,
        linewidth=1.5, label="Validation loss"
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_xlim(0, 3)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUTPUT_DIR / "training_loss.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_f1(eval_metrics: list[dict]) -> None:
    """Plot evaluation F1 score vs epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))

    eval_epochs = [m["epoch"] for m in eval_metrics]
    eval_f1 = [m["eval_f1"] for m in eval_metrics]

    ax.plot(
        eval_epochs, eval_f1, "s-", color="#2E7D32", markersize=8,
        linewidth=2, label="Validation F1"
    )

    for ep, f1 in zip(eval_epochs, eval_f1):
        ax.annotate(
            f"{f1:.4f}", (ep, f1),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xlim(0, 3)
    ax.set_ylim(0.91, 0.96)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUTPUT_DIR / "eval_f1.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_metrics, eval_metrics = parse_log(WANDB_LOG)
    print(f"Parsed {len(train_metrics)} training steps, {len(eval_metrics)} eval checkpoints")
    plot_loss(train_metrics, eval_metrics)
    plot_f1(eval_metrics)


if __name__ == "__main__":
    main()
