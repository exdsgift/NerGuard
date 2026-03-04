"""Generate benchmark visualization charts from experiment results."""

import json
import math
import os
import sys
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


# ── Static metadata ───────────────────────────────────────────────────────

SYSTEM_ORDER = [
    "NerGuard Base",
    "NerGuard Hybrid (gpt-4o)",
    "NerGuard Hybrid V2 (gpt-4o)",
    "NerGuard Hybrid V2 (llama3.1:8b)",
    "NerGuard Hybrid V2 (qwen2.5:7b)",
    "NerGuard Hybrid V2 (qwen2.5:14b)",
    "NerGuard Hybrid V2 (mistral-nemo:12b)",
    "NerGuard Hybrid V2 (phi4:14b)",
    "NerGuard Hybrid V2 (deepseek-r1:14b)",
    "NerGuard Hybrid V2 (gpt-oss:20b)",
    "Piiranha",
    "Piiranha Hybrid (gpt-4o)",
    "Presidio",
    "GLiNER",
    "spaCy (en_core_web_trf)",
    "dslim/bert-base-NER",
]

SHORT_NAMES = {
    "NerGuard Base": "NerGuard Base",
    "NerGuard Hybrid (gpt-4o)": "Hybrid (gpt-4o)",
    "NerGuard Hybrid V2 (gpt-4o)": "Hybrid V2 (gpt-4o)",
    "Piiranha": "Piiranha",
    "Piiranha Hybrid (gpt-4o)": "Piiranha Hybrid",
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
    """Return short display name. Falls back to extracting model name from parentheses."""
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    # For local Ollama models: "NerGuard Hybrid V2 (llama3.1:8b)" → "V2 (llama3.1:8b)"
    if "Hybrid V2 (" in name:
        model = name.split("Hybrid V2 (")[1].rstrip(")")
        return f"V2 ({model})"
    if "Hybrid (" in name:
        model = name.split("Hybrid (")[1].rstrip(")")
        return f"Hybrid ({model})"
    return name


# ── Axis helpers ──────────────────────────────────────────────────────────

def _ylim_from_values(values, pad=0.4, min_span=0.05):
    """Compute y-axis limits zoomed to the actual data range.

    pad: fraction of span to add as padding above and below.
    min_span: minimum axis span to avoid degenerate single-value plots.
    """
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return 0.0, 1.1
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, min_span)
    bottom = max(0.0, vmin - span * pad)
    top = min(1.0, vmax + span * 0.25)
    return bottom, top


def _xlim_from_values(values, right_pad=0.15, min_span=0.05):
    """Compute x-axis limits for horizontal bar charts with label room."""
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return 0.0, 1.1
    vmin, vmax = min(vals), max(vals)
    span = max(vmax - vmin, min_span)
    left = max(0.0, vmin - span * 0.1)
    right = min(1.0, vmax + span * right_pad + 0.04)
    return left, right


def _bar_label_offset(vals):
    """Vertical offset for bar value labels, relative to bar top."""
    span = max(vals) - min(v for v in vals if v > 0) if vals else 1
    return max(span * 0.015, 0.001)


# ── Annotation helper ────────────────────────────────────────────────────

def _annotate_no_overlap(ax, pts, fontsize=8, xscale="linear"):
    """Place scatter annotations without label overlap.

    Strategy: sort by y-value and assign evenly-spaced vertical offsets
    so that close points get staggered labels. Always draws a leader line.

    pts: list of (x, y, text)
    """
    if not pts:
        return

    n = len(pts)
    # Sort by y so adjacent ranks correspond to adjacent points
    sorted_pts = sorted(pts, key=lambda p: p[1])

    # Spread labels over ~160pt total; minimum 18pt step
    step = max(18, 160 // max(n, 1))
    total = (n - 1) * step

    xlim = ax.get_xlim()

    for rank, (x, y, text) in enumerate(sorted_pts):
        oy = rank * step - total // 2

        # Choose left vs right placement based on relative x position
        if xscale == "log":
            try:
                lx = math.log10(max(x, 1e-10))
                ll0 = math.log10(max(xlim[0], 1e-10))
                ll1 = math.log10(max(xlim[1], 1e-10))
                rel_x = (lx - ll0) / (ll1 - ll0) if ll1 != ll0 else 0.5
            except (ValueError, ZeroDivisionError):
                rel_x = 0.5
        else:
            denom = xlim[1] - xlim[0]
            rel_x = (x - xlim[0]) / denom if denom else 0.5

        if rel_x < 0.65:
            ox, ha = 12, "left"
        else:
            ox, ha = -(len(text) * 6 + 15), "right"

        ax.annotate(
            text, (x, y),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=fontsize,
            ha=ha,
            va="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
        )


# ── Chart 1: Bar — F1-macro per system ───────────────────────────────────

def plot_f1_by_dataset(data, output_dir):
    """Bar chart of F1-macro for each system (single dataset) or grouped bars."""
    datasets = sorted(set(r["dataset"] for r in data))
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["f1_macro"] for r in data}

    if len(datasets) == 1:
        ds = datasets[0]
        vals = [lookup.get((sys, ds), 0) for sys in systems]
        names = [_short(sys) for sys in systems]

        fig, ax = plt.subplots(figsize=(max(10, len(systems) * 1.4), 6))
        bars = ax.bar(names, vals)
        off = _bar_label_offset(vals)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("F1-macro")
        ax.set_title(f"F1-macro — {DATASET_LABELS.get(ds, ds)}")
        ax.set_ylim(*_ylim_from_values(vals))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"f1_macro_{ds}.pdf"), dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(datasets))
    width = 0.8 / len(systems)
    all_vals = [lookup.get((sys, ds), 0) for sys in systems for ds in datasets]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sys in enumerate(systems):
        vals = [lookup.get((sys, ds), 0) for ds in datasets]
        offset = (i - len(systems) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=_short(sys))
        off = _bar_label_offset(all_vals)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("F1-macro")
    ax.set_title("F1-macro by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(*_ylim_from_values(all_vals))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_macro_by_dataset.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 2: Bar — Entity-F1 per system ──────────────────────────────────

def plot_entity_f1_by_dataset(data, output_dir):
    """Bar chart of entity-level F1 for each system."""
    datasets = sorted(set(r["dataset"] for r in data))
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["entity_f1"] for r in data}

    if len(datasets) == 1:
        ds = datasets[0]
        vals = [lookup.get((sys, ds), 0) for sys in systems]
        names = [_short(sys) for sys in systems]

        fig, ax = plt.subplots(figsize=(max(10, len(systems) * 1.4), 6))
        bars = ax.bar(names, vals)
        off = _bar_label_offset(vals)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Entity-level F1")
        ax.set_title(f"Entity-level F1 — {DATASET_LABELS.get(ds, ds)}")
        ax.set_ylim(*_ylim_from_values(vals))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"entity_f1_{ds}.pdf"), dpi=150)
        plt.close(fig)
        return

    x = np.arange(len(datasets))
    width = 0.8 / len(systems)
    all_vals = [lookup.get((sys, ds), 0) for sys in systems for ds in datasets]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sys in enumerate(systems):
        vals = [lookup.get((sys, ds), 0) for ds in datasets]
        offset = (i - len(systems) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=_short(sys))
        off = _bar_label_offset(all_vals)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Entity-level F1")
    ax.set_title("Entity-level F1 by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(*_ylim_from_values(all_vals))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "entity_f1_by_dataset.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 3: Scatter — F1 vs Latency ─────────────────────────────────────

def plot_f1_vs_latency(data, output_dir, dataset="nvidia-pii"):
    """Scatter: F1-macro vs mean latency with non-overlapping annotations."""
    subset = [r for r in data if r["dataset"] == dataset]
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    systems = sorted(set(r["system"] for r in subset), key=_sort_key)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "H", "+", "x"]
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(systems), 10)))
    if len(systems) > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, len(systems)))

    pts = []
    for i, sys in enumerate(systems):
        for r in subset:
            if r["system"] != sys:
                continue
            short = _short(sys)
            ax.scatter(
                r["latency_mean_ms"], r["f1_macro"],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                s=100,
                label=short,
                zorder=3,
            )
            pts.append((r["latency_mean_ms"], r["f1_macro"], short))

    # Dynamic axis limits
    lats = [p[0] for p in pts]
    f1s = [p[1] for p in pts]
    ybot, ytop = _ylim_from_values(f1s, pad=0.5)
    ax.set_ylim(ybot, ytop)
    ax.set_xscale("log")

    # Non-overlapping annotations (run after set_xscale so xlim is correct)
    _annotate_no_overlap(ax, pts, fontsize=8, xscale="log")

    ax.set_xlabel("Mean Latency (ms, log scale)")
    ax.set_ylabel("F1-macro")
    ax.set_title(f"F1-macro vs Latency — {DATASET_LABELS.get(dataset, dataset)}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"f1_vs_latency_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 4: Horizontal bar — System ranking ──────────────────────────────

def plot_system_ranking(data, output_dir, dataset="ai4privacy"):
    """Horizontal bar chart ranking systems by F1-macro."""
    subset = [r for r in data if r["dataset"] == dataset]
    subset.sort(key=lambda r: r["f1_macro"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(subset) * 0.55)))
    names = [_short(r["system"]) for r in subset]
    vals = [r["f1_macro"] for r in subset]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))

    bars = ax.barh(names, vals, color=colors)
    xleft, xright = _xlim_from_values(vals, right_pad=0.2)
    label_off = (xright - xleft) * 0.01
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + label_off, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    ax.set_xlabel("F1-macro")
    ax.set_title(f"System Ranking — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xlim(xleft, xright)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ranking_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 5: Precision vs Recall scatter ─────────────────────────────────

def plot_precision_vs_recall(all_results, output_dir, dataset="ai4privacy"):
    """Scatter: precision vs recall (micro) with non-overlapping annotations."""
    systems = sorted(
        [k for k in all_results if k[1] == dataset],
        key=lambda k: _sort_key(k[0])
    )
    if not systems:
        return

    # Collect points first to compute dynamic limits
    pts = []
    for sys_name, ds in systems:
        r = all_results[(sys_name, ds)]
        tl = r["token_level"]
        pts.append((tl["recall_micro"], tl["precision_micro"], _short(sys_name)))

    rec_vals = [p[0] for p in pts]
    pre_vals = [p[1] for p in pts]
    xbot, xtop = _ylim_from_values(rec_vals, pad=0.5)
    ybot, ytop = _ylim_from_values(pre_vals, pad=0.5)
    # Add margin for annotations
    xtop = min(1.05, xtop + 0.05)
    ytop = min(1.05, ytop + 0.05)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(systems), 10)))
    if len(systems) > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, len(systems)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "H", "+", "x"]

    for i, ((sys_name, ds), (rx, ry, _)) in enumerate(zip(systems, pts)):
        ax.scatter(rx, ry, marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], s=100,
                   label=_short(sys_name), zorder=3)

    ax.set_xlim(xbot, xtop)
    ax.set_ylim(ybot, ytop)

    # Iso-F1 curves clipped to visible range
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_vals = np.linspace(max(xbot, 0.01), xtop, 200)
        denom = 2 * r_vals - f1_val
        with np.errstate(divide="ignore", invalid="ignore"):
            p_vals = np.where(denom > 0, f1_val * r_vals / denom, np.nan)
        mask = (p_vals >= ybot) & (p_vals <= ytop)
        if mask.any():
            ax.plot(r_vals[mask], p_vals[mask], "--", color="gray", alpha=0.3, lw=0.8)
            # Label at midpoint of visible segment
            mid = np.where(mask)[0][len(np.where(mask)[0]) // 2]
            ax.text(r_vals[mid], p_vals[mid] + (ytop - ybot) * 0.01,
                    f"F1={f1_val}", fontsize=7, color="gray", alpha=0.6)

    _annotate_no_overlap(ax, pts, fontsize=8, xscale="linear")

    ax.set_xlabel("Recall (micro)")
    ax.set_ylabel("Precision (micro)")
    ax.set_title(f"Precision vs Recall — {DATASET_LABELS.get(dataset, dataset)}")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"precision_recall_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 6: Line — F1 across datasets ───────────────────────────────────

def plot_f1_across_datasets(data, output_dir):
    """Line graph: F1-macro trend across datasets. Skipped if < 2 datasets."""
    datasets = sorted(set(r["dataset"] for r in data))
    if len(datasets) < 2:
        return
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["f1_macro"] for r in data}
    all_vals = [v for v in lookup.values() if v]

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, sys in enumerate(systems):
        vals = [lookup.get((sys, ds), None) for ds in datasets]
        ax.plot(range(len(datasets)), vals,
                marker=markers[i % len(markers)],
                label=_short(sys),
                linewidth=1.5, markersize=8)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylabel("F1-macro")
    ax.set_title("F1-macro Across Datasets")
    ax.set_ylim(*_ylim_from_values(all_vals, pad=0.3))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_across_datasets.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 7: Bar — Latency comparison ────────────────────────────────────

def plot_latency_comparison(data, output_dir, dataset="ai4privacy"):
    """Bar chart comparing mean latency across systems (log scale if range > 10×)."""
    subset = [r for r in data if r["dataset"] == dataset]
    subset.sort(key=lambda r: _sort_key(r["system"]))
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(subset) * 1.3), 5))
    names = [_short(r["system"]) for r in subset]
    vals = [r["latency_mean_ms"] for r in subset]

    bars = ax.bar(names, vals)
    top_off = max(vals) * 0.025
    for bar, v in zip(bars, vals):
        label = f"{v:.0f}ms" if v >= 10 else f"{v:.1f}ms"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top_off,
                label, ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title(f"Inference Latency — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"latency_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 8: Heatmap — F1-macro system×dataset ───────────────────────────

def plot_f1_heatmap(data, output_dir):
    """Heatmap of F1-macro. Skipped if < 2 datasets."""
    datasets = sorted(set(r["dataset"] for r in data))
    if len(datasets) < 2:
        return
    systems = sorted(set(r["system"] for r in data), key=_sort_key)
    lookup = {(r["system"], r["dataset"]): r["f1_macro"] for r in data}

    matrix = np.zeros((len(systems), len(datasets)))
    for i, sys in enumerate(systems):
        for j, ds in enumerate(datasets):
            matrix[i, j] = lookup.get((sys, ds), 0)

    # Use data-driven colormap range
    vmin = matrix[matrix > 0].min() if (matrix > 0).any() else 0
    vmax = matrix.max()

    fig, ax = plt.subplots(figsize=(8, max(4, len(systems) * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([_short(s) for s in systems])

    mid = (vmin + vmax) / 2
    for i in range(len(systems)):
        for j in range(len(datasets)):
            v = matrix[i, j]
            color = "white" if v > mid else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("F1-macro Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_heatmap.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 9: Per-entity F1 for a system ──────────────────────────────────

def plot_per_entity_f1(all_results, per_entity, output_dir,
                       system="NerGuard Base", dataset="ai4privacy"):
    """Bar chart of F1 per entity type for a specific system×dataset."""
    target_key = None
    # Normalise both the query words and the candidate key the same way
    # so that Ollama-style "llama3.1:8b" → "llama3.18b" matches the dir key
    def _norm(s):
        return s.lower().replace("(", "").replace(")", "").replace(":", "")
    sys_words = _norm(system).split()
    for key in per_entity:
        kl = _norm(key)
        if dataset in kl and all(w in kl for w in sys_words if w not in ("×",)):
            target_key = key
            break

    if target_key is None:
        return

    scores = per_entity[target_key]
    entity_f1 = {}
    for label, metrics in scores.items():
        if label == "O":
            continue
        entity = label.replace("B-", "").replace("I-", "")
        if entity not in entity_f1:
            entity_f1[entity] = {"f1_sum": 0, "count": 0}
        entity_f1[entity]["f1_sum"] += metrics["f1"]
        entity_f1[entity]["count"] += 1

    if not entity_f1:
        return

    entities = sorted(entity_f1.keys())
    f1_vals = [entity_f1[e]["f1_sum"] / entity_f1[e]["count"] for e in entities]

    fig, ax = plt.subplots(figsize=(max(12, len(entities) * 0.9), 5))
    bar_colors = ["tab:green" if v >= 0.9 else "tab:orange" if v >= 0.5 else "tab:red"
                  for v in f1_vals]
    bars = ax.bar(entities, f1_vals, color=bar_colors)

    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_ylabel("F1")
    ax.set_title(f"Per-Entity F1 — {_short(system)} on {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    sys_clean = system.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    fig.savefig(os.path.join(output_dir, f"per_entity_{sys_clean}_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Chart 10: Token vs Entity F1 ─────────────────────────────────────────

def plot_token_vs_entity_f1(data, output_dir, dataset="ai4privacy"):
    """Grouped bar: token-level F1-macro vs entity-level F1 side by side."""
    subset = sorted([r for r in data if r["dataset"] == dataset],
                    key=lambda r: _sort_key(r["system"]))
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(subset) * 1.5), 6))
    x = np.arange(len(subset))
    width = 0.35

    token_vals = [r["f1_macro"] for r in subset]
    entity_vals = [r["entity_f1"] for r in subset]
    names = [_short(r["system"]) for r in subset]
    all_vals = token_vals + entity_vals

    b1 = ax.bar(x - width / 2, token_vals, width, label="Token-level F1-macro")
    b2 = ax.bar(x + width / 2, entity_vals, width, label="Entity-level F1")

    off = _bar_label_offset(all_vals)
    for bar, v in list(zip(b1, token_vals)) + list(zip(b2, entity_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + off,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("F1 Score")
    ax.set_title(f"Token-level vs Entity-level F1 — {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(*_ylim_from_values(all_vals))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"token_vs_entity_{dataset}.pdf"), dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

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

    plot_f1_by_dataset(data, output_dir)
    print("  [1/10] f1_macro_*.pdf")

    plot_entity_f1_by_dataset(data, output_dir)
    print("  [2/10] entity_f1_*.pdf")

    datasets_in_data = sorted(set(r["dataset"] for r in data))
    for ds in datasets_in_data:
        plot_f1_vs_latency(data, output_dir, dataset=ds)
    print("  [3/10] f1_vs_latency_*.pdf")

    for ds in datasets_in_data:
        plot_system_ranking(data, output_dir, dataset=ds)
    print("  [4/10] ranking_*.pdf")

    for ds in datasets_in_data:
        if any(k[1] == ds for k in all_results):
            plot_precision_vs_recall(all_results, output_dir, dataset=ds)
    print("  [5/10] precision_recall_*.pdf")

    plot_f1_across_datasets(data, output_dir)
    if len(datasets_in_data) >= 2:
        print("  [6/10] f1_across_datasets.pdf")
    else:
        print("  [6/10] f1_across_datasets (skipped: single dataset)")

    for ds in datasets_in_data:
        plot_latency_comparison(data, output_dir, dataset=ds)
    print("  [7/10] latency_*.pdf")

    plot_f1_heatmap(data, output_dir)
    if len(datasets_in_data) >= 2:
        print("  [8/10] f1_heatmap.pdf")
    else:
        print("  [8/10] f1_heatmap (skipped: single dataset)")

    for ds in datasets_in_data:
        systems_in_ds = sorted({k[0] for k in all_results if k[1] == ds})
        for sys in systems_in_ds:
            plot_per_entity_f1(all_results, per_entity, output_dir, system=sys, dataset=ds)
    print("  [9/10] per_entity_*.pdf")

    for ds in datasets_in_data:
        plot_token_vs_entity_f1(data, output_dir, dataset=ds)
    print("  [10/10] token_vs_entity_*.pdf")

    n_files = len(list(Path(output_dir).glob("*.pdf")))
    print(f"\nDone! {n_files} plots saved to {output_dir}")


if __name__ == "__main__":
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "./experiments/test_50"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_all_plots(exp_dir, out_dir)
