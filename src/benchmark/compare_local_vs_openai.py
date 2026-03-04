"""Compare local Ollama models vs OpenAI across benchmark experiment directories.

Usage:
    uv run python -m src.benchmark.compare_local_vs_openai \
        --openai-dir ./experiments/2026-03-03_13-23 \
        --local-dirs ./experiments/<T1> ./experiments/<T2> ... \
        --output-dir ./experiments/comparison_local_vs_openai
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────

def _load_results_from_dir(exp_dir: Path) -> list[dict]:
    """Load all results.json from an experiment directory."""
    results = []
    for rf in exp_dir.glob("*/results.json"):
        if "summary" in rf.parent.name:
            continue
        with open(rf) as f:
            results.append(json.load(f))
    return results


def load_all_experiments(openai_dir: str, local_dirs: list[str]) -> list[dict]:
    """Merge results from the OpenAI experiment and all local experiment dirs.

    Local dirs contribute only hybrid systems (those that use LLM routing).
    The OpenAI dir contributes all systems (baselines + hybrid).
    """
    all_results = []

    # Load OpenAI experiment (all systems)
    openai_path = Path(openai_dir)
    openai_results = _load_results_from_dir(openai_path)
    for r in openai_results:
        r["_source"] = "openai"
    all_results.extend(openai_results)
    print(f"Loaded {len(openai_results)} results from OpenAI dir: {openai_dir}")

    # Load local experiments (only hybrid systems to avoid duplicating baselines)
    for local_dir in local_dirs:
        local_path = Path(local_dir)
        if not local_path.exists():
            print(f"  WARNING: {local_dir} does not exist, skipping")
            continue
        local_results = _load_results_from_dir(local_path)
        hybrid = [r for r in local_results if "hybrid" in r.get("system", "").lower()]
        for r in hybrid:
            r["_source"] = "local"
        all_results.extend(hybrid)
        print(f"  Loaded {len(hybrid)} hybrid results from: {local_dir}")

    return all_results


def _flatten(r: dict) -> dict:
    """Flatten a results.json into a single-level dict for easy plotting."""
    tl = r.get("token_level", {})
    el = r.get("entity_level", {})
    lat = r.get("latency", {})
    return {
        "system": r["system"],
        "dataset": r["dataset"],
        "f1_macro": tl.get("f1_macro", 0.0),
        "f1_micro": tl.get("f1_micro", 0.0),
        "entity_f1": el.get("f1", 0.0),
        "latency_mean_ms": lat.get("mean_ms", 0.0),
        "n_evaluated_labels": r.get("n_evaluated_labels", 0),
        "tier": r.get("tier", 2),
        "n_samples": r.get("n_samples", 0),
        "_source": r.get("_source", "unknown"),
    }


# ── Color / style helpers ─────────────────────────────────────────────────

# Systems that don't use LLM (shown in grey)
BASELINE_SYSTEMS = {"NerGuard Base", "Piiranha", "Presidio", "spaCy (en_core_web_trf)", "dslim/bert-base-NER"}

# Color palette: OpenAI in blue, local models in a gradient of warm colors
OPENAI_COLOR = "#2563EB"  # blue
LOCAL_COLORS = [
    "#DC2626",  # red
    "#D97706",  # amber
    "#16A34A",  # green
    "#7C3AED",  # violet
    "#DB2777",  # pink
    "#0891B2",  # cyan
    "#65A30D",  # lime
]
BASELINE_COLOR = "#9CA3AF"  # grey


def _get_system_color(system: str, local_model_colors: dict) -> str:
    """Return a consistent color for a system."""
    if system in BASELINE_SYSTEMS:
        return BASELINE_COLOR
    if "gpt-4o" in system:
        return OPENAI_COLOR
    # Extract model name from system name, e.g. "NerGuard Hybrid (llama3.1:8b)"
    for model, color in local_model_colors.items():
        if model in system:
            return color
    return "#6B7280"


def _build_model_color_map(flat_results: list[dict]) -> dict:
    """Assign a color to each local model name."""
    local_models = set()
    for r in flat_results:
        if r["_source"] == "local":
            # Extract model name from system name like "NerGuard Hybrid (llama3.1:8b)"
            sys_name = r["system"]
            if "(" in sys_name and ")" in sys_name:
                model = sys_name[sys_name.rfind("(") + 1: sys_name.rfind(")")]
                local_models.add(model)
    return {m: LOCAL_COLORS[i % len(LOCAL_COLORS)] for i, m in enumerate(sorted(local_models))}


# ── Chart 1: F1-macro bar chart — all systems including local models ──────

def plot_f1_macro_all_systems(flat: list[dict], model_colors: dict, output_dir: str, dataset: str = "nvidia-pii"):
    """Horizontal bar chart ranking all systems by F1-macro, colored by model/source."""
    subset = [r for r in flat if r["dataset"] == dataset]
    if not subset:
        return

    # Sort by F1 ascending (best at top for horizontal bar)
    subset.sort(key=lambda r: r["f1_macro"])

    fig, ax = plt.subplots(figsize=(12, max(6, len(subset) * 0.45)))

    names = [r["system"] for r in subset]
    vals = [r["f1_macro"] for r in subset]
    colors = [_get_system_color(r["system"], model_colors) for r in subset]

    bars = ax.barh(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)

    # Legend
    legend_patches = [mpatches.Patch(color=OPENAI_COLOR, label="OpenAI (gpt-4o)")]
    for model, color in model_colors.items():
        legend_patches.append(mpatches.Patch(color=color, label=f"Local: {model}"))
    legend_patches.append(mpatches.Patch(color=BASELINE_COLOR, label="Baseline (no LLM)"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    ax.set_xlabel("F1-macro (token-level)")
    ax.set_title(f"F1-macro — All Systems (NVIDIA-PII, {subset[0].get('n_samples', '?')} samples)")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.axvline(x=max(v for r, v in zip(subset, vals) if "gpt-4o" in r["system"]),
               color=OPENAI_COLOR, linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"f1_macro_all_systems_{dataset}.pdf"), dpi=150)
    plt.close(fig)
    print("  [1] f1_macro_all_systems.pdf")


# ── Chart 2: Entity-level F1 bar chart ───────────────────────────────────

def plot_entity_f1_all_systems(flat: list[dict], model_colors: dict, output_dir: str, dataset: str = "nvidia-pii"):
    """Horizontal bar chart of entity-level F1 for all systems."""
    subset = [r for r in flat if r["dataset"] == dataset]
    if not subset:
        return

    subset.sort(key=lambda r: r["entity_f1"])

    fig, ax = plt.subplots(figsize=(12, max(6, len(subset) * 0.45)))

    names = [r["system"] for r in subset]
    vals = [r["entity_f1"] for r in subset]
    colors = [_get_system_color(r["system"], model_colors) for r in subset]

    bars = ax.barh(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)

    legend_patches = [mpatches.Patch(color=OPENAI_COLOR, label="OpenAI (gpt-4o)")]
    for model, color in model_colors.items():
        legend_patches.append(mpatches.Patch(color=color, label=f"Local: {model}"))
    legend_patches.append(mpatches.Patch(color=BASELINE_COLOR, label="Baseline (no LLM)"))
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    ax.set_xlabel("Entity-level F1 (seqeval strict)")
    ax.set_title(f"Entity-level F1 — All Systems (NVIDIA-PII, {subset[0].get('n_samples', '?')} samples)")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"entity_f1_all_systems_{dataset}.pdf"), dpi=150)
    plt.close(fig)
    print("  [2] entity_f1_all_systems.pdf")


# ── Chart 3: F1-macro vs Latency scatter ─────────────────────────────────

def plot_f1_vs_latency(flat: list[dict], model_colors: dict, output_dir: str, dataset: str = "nvidia-pii"):
    """Scatter: F1-macro vs mean latency. X axis log-scale."""
    subset = [r for r in flat if r["dataset"] == dataset and r["latency_mean_ms"] > 0]
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    markers_map = {}
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    for r in subset:
        sys_name = r["system"]
        color = _get_system_color(sys_name, model_colors)
        # Assign marker by model family
        if sys_name in markers_map:
            marker = markers_map[sys_name]
        else:
            marker = marker_cycle[len(markers_map) % len(marker_cycle)]
            markers_map[sys_name] = marker

        ax.scatter(r["latency_mean_ms"], r["f1_macro"],
                   color=color, marker=marker, s=120, zorder=3)
        label = sys_name.replace("NerGuard Hybrid V2", "NH-V2").replace("NerGuard Hybrid", "NH")
        ax.annotate(label, (r["latency_mean_ms"], r["f1_macro"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("Mean Latency (ms) [log scale]")
    ax.set_ylabel("F1-macro (token-level)")
    ax.set_title(f"F1-macro vs Latency — Local vs OpenAI (NVIDIA-PII)")
    ax.grid(True, alpha=0.3)

    legend_patches = [mpatches.Patch(color=OPENAI_COLOR, label="OpenAI (gpt-4o)")]
    for model, color in model_colors.items():
        legend_patches.append(mpatches.Patch(color=color, label=f"Local: {model}"))
    legend_patches.append(mpatches.Patch(color=BASELINE_COLOR, label="Baseline (no LLM)"))
    ax.legend(handles=legend_patches, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"f1_vs_latency_{dataset}.pdf"), dpi=150)
    plt.close(fig)
    print("  [3] f1_vs_latency.pdf")


# ── Chart 4: Hybrid models comparison — grouped bar (V1 vs V2) ───────────

def plot_hybrid_comparison(flat: list[dict], model_colors: dict, output_dir: str, dataset: str = "nvidia-pii"):
    """Grouped bar: NerGuard Hybrid vs Hybrid V2 for each LLM model."""
    subset = [r for r in flat if r["dataset"] == dataset and "hybrid" in r["system"].lower()]
    if not subset:
        return

    # Group by model
    by_model: dict[str, dict] = {}
    for r in subset:
        sys_name = r["system"]
        # Extract model name
        if "(" in sys_name and ")" in sys_name:
            model = sys_name[sys_name.rfind("(") + 1: sys_name.rfind(")")]
        else:
            model = "unknown"
        version = "V2" if "v2" in sys_name.lower() else "V1"
        if model not in by_model:
            by_model[model] = {}
        by_model[model][version] = r["f1_macro"]

    if not by_model:
        return

    # Sort models: gpt-4o first, then alphabetical local models
    models = sorted(by_model.keys(), key=lambda m: (0 if m == "gpt-4o" else 1, m))
    v1_vals = [by_model[m].get("V1", 0) for m in models]
    v2_vals = [by_model[m].get("V2", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 6))
    bars1 = ax.bar(x - width / 2, v1_vals, width, label="NerGuard Hybrid")
    bars2 = ax.bar(x + width / 2, v2_vals, width, label="NerGuard Hybrid V2", alpha=0.85)

    # Color bars by model
    for bar, model in zip(bars1, models):
        bar.set_color(_get_system_color(f"NerGuard Hybrid ({model})", model_colors))
    for bar, model in zip(bars2, models):
        bar.set_color(_get_system_color(f"NerGuard Hybrid ({model})", model_colors))
        bar.set_edgecolor("black")
        bar.set_linewidth(0.8)

    for bars in (bars1, bars2):
        for bar in bars:
            v = bar.get_height()
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("F1-macro")
    ax.set_title("NerGuard Hybrid vs Hybrid V2 — F1-macro by LLM Model")
    ax.set_ylim(0, max(max(v1_vals), max(v2_vals)) * 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"hybrid_v1_vs_v2_{dataset}.pdf"), dpi=150)
    plt.close(fig)
    print("  [4] hybrid_v1_vs_v2.pdf")


# ── Chart 5: Heatmap — F1-macro by model × system variant ────────────────

def plot_model_heatmap(flat: list[dict], model_colors: dict, output_dir: str, dataset: str = "nvidia-pii"):
    """Heatmap: rows = LLM model, cols = system variant (Hybrid / Hybrid V2)."""
    subset = [r for r in flat if r["dataset"] == dataset and "hybrid" in r["system"].lower()]
    if not subset:
        return

    # Extract (model, variant, f1) triples
    by_model: dict[str, dict] = {}
    for r in subset:
        sys_name = r["system"]
        if "(" in sys_name and ")" in sys_name:
            model = sys_name[sys_name.rfind("(") + 1: sys_name.rfind(")")]
        else:
            model = "unknown"
        version = "Hybrid V2" if "v2" in sys_name.lower() else "Hybrid"
        if model not in by_model:
            by_model[model] = {}
        by_model[model][version] = r["f1_macro"]

    models = sorted(by_model.keys(), key=lambda m: (0 if m == "gpt-4o" else 1, m))
    variants = ["Hybrid", "Hybrid V2"]

    matrix = np.zeros((len(models), len(variants)))
    for i, model in enumerate(models):
        for j, variant in enumerate(variants):
            matrix[i, j] = by_model[model].get(variant, 0)

    fig, ax = plt.subplots(figsize=(6, max(4, len(models) * 0.6)))
    vmin = max(0, matrix[matrix > 0].min() - 0.02) if matrix.any() else 0
    vmax = matrix.max() + 0.01
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(variants)):
            v = matrix[i, j]
            color = "white" if v > (vmin + vmax) / 2 + 0.05 else "black"
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=10, color=color)

    ax.set_title("F1-macro Heatmap — NerGuard Hybrid by LLM Model")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"hybrid_heatmap_{dataset}.pdf"), dpi=150)
    plt.close(fig)
    print("  [5] hybrid_heatmap.pdf")


# ── Chart 6: Delta F1 vs gpt-4o (local model improvement/regression) ─────

def plot_delta_vs_openai(flat: list[dict], model_colors: dict, output_dir: str, dataset: str = "nvidia-pii"):
    """Bar chart: ΔF1-macro relative to gpt-4o for Hybrid V2 across local models."""
    subset = [r for r in flat if r["dataset"] == dataset and "hybrid v2" in r["system"].lower()]
    if not subset:
        return

    # Get gpt-4o baseline
    openai_row = next((r for r in subset if "gpt-4o" in r["system"]), None)
    if openai_row is None:
        return
    baseline_f1 = openai_row["f1_macro"]

    # Compute delta for all other models
    local_rows = [r for r in subset if "gpt-4o" not in r["system"]]
    local_rows.sort(key=lambda r: r["f1_macro"] - baseline_f1)

    if not local_rows:
        return

    models = []
    deltas = []
    for r in local_rows:
        sys_name = r["system"]
        if "(" in sys_name and ")" in sys_name:
            model = sys_name[sys_name.rfind("(") + 1: sys_name.rfind(")")]
        else:
            model = sys_name
        models.append(model)
        deltas.append(r["f1_macro"] - baseline_f1)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    colors = [_get_system_color(f"NerGuard Hybrid V2 ({m})", model_colors) for m in models]
    bars = ax.bar(models, deltas, color=colors)

    for bar, v in zip(bars, deltas):
        ypos = bar.get_height() + 0.001 if v >= 0 else bar.get_height() - 0.003
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    ax.axhline(0, color=OPENAI_COLOR, linestyle="--", linewidth=1.5, label=f"gpt-4o baseline ({baseline_f1:.4f})")
    ax.set_ylabel("ΔF1-macro vs gpt-4o")
    ax.set_title("NerGuard Hybrid V2 — F1-macro Delta vs gpt-4o (NVIDIA-PII)")
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"delta_vs_openai_{dataset}.pdf"), dpi=150)
    plt.close(fig)
    print("  [6] delta_vs_openai.pdf")


# ── Summary table ─────────────────────────────────────────────────────────

def write_summary_table(flat: list[dict], output_dir: str, dataset: str = "nvidia-pii"):
    """Write a markdown summary table of all results."""
    subset = [r for r in flat if r["dataset"] == dataset]
    subset.sort(key=lambda r: r["f1_macro"], reverse=True)

    lines = [
        f"# Benchmark Results — {dataset} ({subset[0].get('n_samples', '?')} samples)\n",
        "| System | F1-macro | F1-micro | Entity-F1 | Latency (ms) | Source |",
        "|--------|----------|----------|-----------|--------------|--------|",
    ]
    for r in subset:
        source = "OpenAI" if "gpt-4o" in r["system"] else ("Local" if r["_source"] == "local" else "Baseline")
        lines.append(
            f"| {r['system']} | {r['f1_macro']:.4f} | {r['f1_micro']:.4f} | "
            f"{r['entity_f1']:.4f} | {r['latency_mean_ms']:.1f} | {source} |"
        )

    table_path = os.path.join(output_dir, "comparison_summary.md")
    with open(table_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [table] comparison_summary.md")

    # Also write JSON
    json_path = os.path.join(output_dir, "comparison_summary.json")
    with open(json_path, "w") as f:
        json.dump(subset, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare local vs OpenAI benchmark results")
    parser.add_argument("--openai-dir", required=True, help="Path to OpenAI experiment directory")
    parser.add_argument("--local-dirs", nargs="+", required=True, help="Paths to local experiment directories")
    parser.add_argument("--output-dir", default="./experiments/comparison_local_vs_openai",
                        help="Output directory for plots and summary")
    parser.add_argument("--dataset", default="nvidia-pii", help="Dataset to compare on")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and merge all results
    raw_results = load_all_experiments(args.openai_dir, args.local_dirs)
    flat = [_flatten(r) for r in raw_results]

    if not flat:
        print("ERROR: No results found!", file=sys.stderr)
        sys.exit(1)

    # Build color map for local models
    model_colors = _build_model_color_map(flat)
    print(f"\nLocal models found: {list(model_colors.keys())}")
    print(f"Total results: {len(flat)} entries across {len(set(r['dataset'] for r in flat))} dataset(s)")
    print(f"\nGenerating plots in {args.output_dir}...")

    dataset = args.dataset

    # Generate all comparison plots
    plot_f1_macro_all_systems(flat, model_colors, args.output_dir, dataset)
    plot_entity_f1_all_systems(flat, model_colors, args.output_dir, dataset)
    plot_f1_vs_latency(flat, model_colors, args.output_dir, dataset)
    plot_hybrid_comparison(flat, model_colors, args.output_dir, dataset)
    plot_model_heatmap(flat, model_colors, args.output_dir, dataset)
    plot_delta_vs_openai(flat, model_colors, args.output_dir, dataset)
    write_summary_table(flat, args.output_dir, dataset)

    n_files = len(list(Path(args.output_dir).glob("*.pdf")))
    print(f"\nDone! {n_files} plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
