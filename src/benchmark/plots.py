"""Generate benchmark visualization charts from experiment results."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_comparison_table(experiments_dir: str) -> list:
    """Find and load the most recent comparison_table.json."""
    exp_path = Path(experiments_dir)
    summaries = sorted(
        list(exp_path.glob("*_summary/comparison_table.json"))
        + list(exp_path.glob("summary/comparison_table.json"))
    )
    if not summaries:
        raise FileNotFoundError(f"No comparison_table.json found in {experiments_dir}")
    with open(summaries[-1]) as f:
        return json.load(f)


def load_per_entity_scores(experiments_dir: str) -> dict:
    """Load per-entity scores for all experiments."""
    scores = {}
    exp_path = Path(experiments_dir)
    for pe_file in exp_path.glob("*/per_entity_scores.json"):
        key = pe_file.parent.name
        with open(pe_file) as f:
            scores[key] = json.load(f)
    return scores


def load_all_results(experiments_dir: str) -> dict:
    """Load full results.json for all experiments."""
    results = {}
    exp_path = Path(experiments_dir)
    for rf in exp_path.glob("*/results.json"):
        if "summary" in rf.parent.name:
            continue
        with open(rf) as f:
            data = json.load(f)
            key = (data["system"], data["dataset"])
            results[key] = data
    return results


# ── Helpers ──────────────────────────────────────────────────────────────

SYSTEM_ORDER = [
    "NerGuard Base",
    "NerGuard Hybrid (gpt-4o)",
    "NerGuard Hybrid V2 (gpt-4o)",
    "Piiranha",
    "Piiranha Hybrid (gpt-4o)",
    "Presidio",
    "GLiNER",
    "spaCy (en_core_web_trf)",
    "dslim/bert-base-NER",
]

SHORT_NAMES = {
    "NerGuard Base": "NerGuard\nBase",
    "NerGuard Hybrid (gpt-4o)": "NerGuard\nHybrid",
    "NerGuard Hybrid V2 (gpt-4o)": "NerGuard\nHybrid V2",
    "Piiranha": "Piiranha",
    "Piiranha Hybrid (gpt-4o)": "Piiranha\nHybrid",
    "Presidio": "Presidio",
    "GLiNER": "GLiNER",
    "spaCy (en_core_web_trf)": "spaCy",
    "dslim/bert-base-NER": "BERT-NER",
}

DATASET_LABELS = {
    "ai4privacy": "AI4Privacy",
    "nvidia-pii": "NVIDIA-PII",
    "wikineural": "WikiNeural",
}


def _sort_key(system_name):
    try:
        return SYSTEM_ORDER.index(system_name)
    except ValueError:
        return len(SYSTEM_ORDER)


def _short(name):
    return SHORT_NAMES.get(name, name)


# ── Chart 1: Grouped bar — F1-macro per dataset ─────────────────────────

def plot_f1_by_dataset(data, output_dir):
    """Grouped bar chart: F1-macro for each system, grouped by dataset.

    With a single dataset, generates one bar chart per dataset instead of
    a grouped chart with one cluster.
    """
    datasets = sorted(set(r["dataset"] for r in data))
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["f1_macro"] for r in data}

    if len(datasets) == 1:
        # Single dataset: one bar per system
        ds = datasets[0]
        vals = [lookup.get((sys, ds), 0) for sys in systems]
        names = [_short(sys).replace("\n", " ") for sys in systems]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, vals)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("F1-macro")
        ax.set_title(f"F1-macro — {DATASET_LABELS.get(ds, ds)}")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"f1_macro_{ds}.pdf"), dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(datasets))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sys in enumerate(systems):
        vals = [lookup.get((sys, ds), 0) for ds in datasets]
        offset = (i - len(systems) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=_short(sys))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("F1-macro")
    ax.set_title("F1-macro by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_macro_by_dataset.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 2: Grouped bar — Entity-F1 per dataset ────────────────────────

def plot_entity_f1_by_dataset(data, output_dir):
    """Grouped bar chart: Entity-level F1 for each system, grouped by dataset.

    With a single dataset, generates one bar chart per dataset.
    """
    datasets = sorted(set(r["dataset"] for r in data))
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["entity_f1"] for r in data}

    if len(datasets) == 1:
        ds = datasets[0]
        vals = [lookup.get((sys, ds), 0) for sys in systems]
        names = [_short(sys).replace("\n", " ") for sys in systems]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, vals)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Entity-level F1")
        ax.set_title(f"Entity-level F1 — {DATASET_LABELS.get(ds, ds)}")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"entity_f1_{ds}.pdf"), dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(datasets))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sys in enumerate(systems):
        vals = [lookup.get((sys, ds), 0) for ds in datasets]
        offset = (i - len(systems) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=_short(sys))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Entity-level F1")
    ax.set_title("Entity-level F1 by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "entity_f1_by_dataset.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 3: Scatter — F1 vs Latency (per dataset) ────────────────────

def plot_f1_vs_latency(data, output_dir, dataset="nvidia-pii"):
    """Scatter plot: F1-macro vs mean latency (ms) for a single dataset."""
    subset = [r for r in data if r["dataset"] == dataset]
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    systems = sorted(set(r["system"] for r in subset), key=_sort_key)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(systems)))

    for i, sys in enumerate(systems):
        for r in subset:
            if r["system"] != sys:
                continue
            short_name = _short(sys).replace(chr(10), " ")
            ax.scatter(
                r["latency_mean_ms"], r["f1_macro"],
                marker=markers[i % len(markers)],
                color=colors[i],
                s=120,
                label=short_name,
                zorder=3,
            )
            ax.annotate(
                short_name,
                (r["latency_mean_ms"], r["f1_macro"]),
                textcoords="offset points", xytext=(8, 5), fontsize=8,
            )

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"F1-macro vs Latency — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"f1_vs_latency_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 4: Horizontal bar — System ranking on AI4Privacy ──────────────

def plot_system_ranking(data, output_dir, dataset="ai4privacy"):
    """Horizontal bar chart ranking systems by F1-macro on a single dataset."""
    subset = [r for r in data if r["dataset"] == dataset]
    subset.sort(key=lambda r: r["f1_macro"])

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [_short(r["system"]).replace("\n", " ") for r in subset]
    vals = [r["f1_macro"] for r in subset]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))

    bars = ax.barh(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)

    ax.set_xlabel("F1-macro")
    ax.set_title(f"System Ranking — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ranking_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 5: Precision vs Recall scatter ────────────────────────────────

def plot_precision_vs_recall(all_results, output_dir, dataset="ai4privacy"):
    """Scatter: token-level precision vs recall for each system on one dataset."""
    fig, ax = plt.subplots(figsize=(8, 8))

    systems = sorted(
        [k for k in all_results if k[1] == dataset],
        key=lambda k: _sort_key(k[0])
    )

    for sys_name, ds in systems:
        r = all_results[(sys_name, ds)]
        tl = r["token_level"]
        ax.scatter(
            tl["recall_micro"], tl["precision_micro"],
            s=120, label=_short(sys_name).replace("\n", " "),
            zorder=3,
        )
        ax.annotate(
            _short(sys_name).replace("\n", " "),
            (tl["recall_micro"], tl["precision_micro"]),
            textcoords="offset points", xytext=(8, 5), fontsize=8,
        )

    # Diagonal iso-F1 lines
    for f1_val in [0.3, 0.5, 0.7, 0.9]:
        r_vals = np.linspace(0.01, 1.0, 100)
        p_vals = (f1_val * r_vals) / (2 * r_vals - f1_val)
        mask = (p_vals > 0) & (p_vals <= 1)
        ax.plot(r_vals[mask], p_vals[mask], "--", color="gray", alpha=0.3, linewidth=0.8)
        # Label the iso-F1 curve
        idx = np.argmin(np.abs(r_vals - f1_val))
        if mask[idx]:
            ax.text(r_vals[idx], p_vals[idx] + 0.02, f"F1={f1_val}", fontsize=7, color="gray")

    ax.set_xlabel("Recall (micro)")
    ax.set_ylabel("Precision (micro)")
    ax.set_title(f"Precision vs Recall — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"precision_recall_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 6: Line graph — F1 across datasets ────────────────────────────

def plot_f1_across_datasets(data, output_dir):
    """Line graph: F1-macro trend across datasets for each system.

    Skipped when fewer than 2 datasets are present (a line chart needs >=2 points).
    """
    datasets = sorted(set(r["dataset"] for r in data))
    if len(datasets) < 2:
        return
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["f1_macro"] for r in data}

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, sys in enumerate(systems):
        vals = [lookup.get((sys, ds), None) for ds in datasets]
        ax.plot(
            range(len(datasets)), vals,
            marker=markers[i % len(markers)],
            label=_short(sys).replace("\n", " "),
            linewidth=1.5,
            markersize=8,
        )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylabel("F1-macro")
    ax.set_title("F1-macro Across Datasets")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_across_datasets.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 7: Bar — Latency comparison ───────────────────────────────────

def plot_latency_comparison(data, output_dir, dataset="ai4privacy"):
    """Bar chart comparing mean latency across systems for one dataset."""
    subset = [r for r in data if r["dataset"] == dataset]
    subset.sort(key=lambda r: _sort_key(r["system"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [_short(r["system"]).replace("\n", " ") for r in subset]
    vals = [r["latency_mean_ms"] for r in subset]

    bars = ax.bar(names, vals)
    for bar, v in zip(bars, vals):
        label = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f"{label}ms", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title(f"Inference Latency — {DATASET_LABELS.get(dataset, dataset)}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"latency_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 8: Heatmap — F1-macro system×dataset ──────────────────────────

def plot_f1_heatmap(data, output_dir):
    """Heatmap of F1-macro for all system×dataset combinations.

    Skipped when fewer than 2 datasets are present (a heatmap with 1 column
    is just a bar chart — already covered by ranking_*.png).
    """
    datasets = sorted(set(r["dataset"] for r in data))
    if len(datasets) < 2:
        return
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["f1_macro"] for r in data}

    matrix = np.zeros((len(systems), len(datasets)))
    for i, sys in enumerate(systems):
        for j, ds in enumerate(datasets):
            matrix[i, j] = lookup.get((sys, ds), 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([_short(s).replace("\n", " ") for s in systems])

    # Annotate cells
    for i in range(len(systems)):
        for j in range(len(datasets)):
            v = matrix[i, j]
            color = "white" if v > 0.6 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10, color=color)

    ax.set_title("F1-macro Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_heatmap.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 9: Per-entity F1 for top system (NerGuard Base on AI4Privacy) ─

def plot_per_entity_f1(all_results, per_entity, output_dir,
                       system="NerGuard Base", dataset="ai4privacy"):
    """Bar chart of F1 per entity type for a specific system×dataset."""
    # Find matching per_entity file by checking if system words appear in key
    target_key = None
    sys_words = system.lower().replace("(", "").replace(")", "").split()
    for key in per_entity:
        kl = key.lower()
        if dataset in kl and all(w in kl for w in sys_words if w not in ("×",)):
            target_key = key
            break

    if target_key is None:
        return

    scores = per_entity[target_key]
    # Aggregate B- and I- into entity type
    entity_f1 = {}
    for label, metrics in scores.items():
        if label == "O":
            continue
        entity = label.replace("B-", "").replace("I-", "")
        if entity not in entity_f1:
            entity_f1[entity] = {"f1_sum": 0, "count": 0}
        entity_f1[entity]["f1_sum"] += metrics["f1"]
        entity_f1[entity]["count"] += 1

    entities = sorted(entity_f1.keys())
    f1_vals = [entity_f1[e]["f1_sum"] / entity_f1[e]["count"] for e in entities]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["tab:green" if v >= 0.9 else "tab:orange" if v >= 0.5 else "tab:red" for v in f1_vals]
    bars = ax.bar(entities, f1_vals, color=colors)

    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_ylabel("F1")
    ax.set_title(f"Per-Entity F1 — {_short(system).replace(chr(10), ' ')} on {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    sys_clean = system.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(os.path.join(output_dir, f"per_entity_{sys_clean}_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 10: Token vs Entity F1 comparison ─────────────────────────────

def plot_token_vs_entity_f1(data, output_dir, dataset="ai4privacy"):
    """Grouped bar: token-level F1-macro vs entity-level F1 side by side."""
    subset = sorted(
        [r for r in data if r["dataset"] == dataset],
        key=lambda r: _sort_key(r["system"])
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(subset))
    width = 0.35

    token_vals = [r["f1_macro"] for r in subset]
    entity_vals = [r["entity_f1"] for r in subset]
    names = [_short(r["system"]).replace("\n", " ") for r in subset]

    ax.bar(x - width / 2, token_vals, width, label="Token-level F1-macro")
    ax.bar(x + width / 2, entity_vals, width, label="Entity-level F1")

    ax.set_ylabel("F1 Score")
    ax.set_title(f"Token-level vs Entity-level F1 — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"token_vs_entity_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def generate_all_plots(experiments_dir: str, output_dir: str = None):
    """Generate all benchmark plots."""
    if output_dir is None:
        output_dir = os.path.join(experiments_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {experiments_dir}...")
    data = load_comparison_table(experiments_dir)
    all_results = load_all_results(experiments_dir)
    per_entity = load_per_entity_scores(experiments_dir)

    print(f"Generating plots in {output_dir}...")

    # 1. F1-macro grouped bar by dataset
    plot_f1_by_dataset(data, output_dir)
    print("  [1/10] f1_macro_by_dataset.pdf")

    # 2. Entity-F1 grouped bar by dataset
    plot_entity_f1_by_dataset(data, output_dir)
    print("  [2/10] entity_f1_by_dataset.pdf")

    # 3. F1 vs Latency scatter (one per dataset)
    datasets_in_data = sorted(set(r["dataset"] for r in data))
    for ds in datasets_in_data:
        plot_f1_vs_latency(data, output_dir, dataset=ds)
    print("  [3/10] f1_vs_latency_*.pdf")

    # 4. System rankings (one per dataset)
    for ds in datasets_in_data:
        plot_system_ranking(data, output_dir, dataset=ds)
    print("  [4/10] ranking_*.pdf")

    # 5. Precision vs Recall scatter (one per dataset)
    for ds in datasets_in_data:
        if any(k[1] == ds for k in all_results):
            plot_precision_vs_recall(all_results, output_dir, dataset=ds)
    print("  [5/10] precision_recall_*.pdf")

    # 6. F1 across datasets (line graph, skipped if < 2 datasets)
    plot_f1_across_datasets(data, output_dir)
    if len(datasets_in_data) >= 2:
        print("  [6/10] f1_across_datasets.pdf")
    else:
        print("  [6/10] f1_across_datasets.png (skipped: single dataset)")

    # 7. Latency comparison (one per dataset)
    for ds in datasets_in_data:
        plot_latency_comparison(data, output_dir, dataset=ds)
    print("  [7/10] latency_*.pdf")

    # 8. F1 heatmap (skipped if < 2 datasets)
    plot_f1_heatmap(data, output_dir)
    if len(datasets_in_data) >= 2:
        print("  [8/10] f1_heatmap.pdf")
    else:
        print("  [8/10] f1_heatmap.png (skipped: single dataset)")

    # 9. Per-entity F1 for key systems on available datasets
    for ds in datasets_in_data:
        for sys in ["NerGuard Hybrid V2 (gpt-4o)", "NerGuard Hybrid (gpt-4o)", "Piiranha", "Presidio"]:
            plot_per_entity_f1(all_results, per_entity, output_dir, system=sys, dataset=ds)
    print("  [9/10] per_entity_*.pdf")

    # 10. Token vs Entity F1 comparison
    for ds in datasets_in_data:
        plot_token_vs_entity_f1(data, output_dir, dataset=ds)
    print("  [10/10] token_vs_entity_*.pdf")

    n_files = len(list(Path(output_dir).glob("*.pdf")))
    print(f"\nDone! {n_files} plots saved to {output_dir}")


if __name__ == "__main__":
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "./experiments/test_50"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_all_plots(exp_dir, out_dir)
