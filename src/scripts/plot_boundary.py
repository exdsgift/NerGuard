"""Plot boundary condition: Entity-F1 delta vs log(n_classes).

Shows when LLM routing helps vs hurts, with correlation analysis.

Usage:
    uv run python -m src.scripts.plot_boundary
"""

import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("BoundaryPlot")

# Domain data: manually specified or loaded from experiments
DOMAIN_DATA = {
    "BC5CDR": {"n_classes": 2, "base_dir_pattern": "biomedical_base_bc5cdr", "hybrid_dir_pattern": "biomedical_hybrid_gpt-4o_bc5cdr"},
    "BUSTER": {"n_classes": 6, "base_dir_pattern": "financial_base_buster", "hybrid_dir_pattern": "financial_hybrid_gpt-4o_buster"},
    "PII": {"n_classes": 20, "base_dir_pattern": "nerguard_base_nvidia-pii", "hybrid_dir_pattern": "nerguard_hybrid_v2_gpt-4o_nvidia-pii"},
    "FiNER-139": {"n_classes": 139, "base_dir_pattern": "finer139_base_finer-139", "hybrid_dir_pattern": "finer139_hybrid_gpt-4o_finer-139"},
}


def find_result(experiment_dirs, pattern):
    """Find results.json matching pattern."""
    for exp_dir in experiment_dirs:
        for entry in os.listdir(exp_dir):
            if pattern in entry:
                path = os.path.join(exp_dir, entry, "results.json")
                if os.path.exists(path):
                    with open(path) as f:
                        return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dirs", nargs="+", default=None)
    parser.add_argument("--output", default="plots/boundary_n_classes.png")
    # Allow manual overrides for results not yet re-run
    parser.add_argument("--manual", action="store_true", help="Use hardcoded results from notes")
    args = parser.parse_args()

    # Search experiment directories
    search_dirs = args.experiment_dirs or []
    if not search_dirs:
        exp_root = "experiments"
        if os.path.exists(exp_root):
            for entry in sorted(os.listdir(exp_root)):
                full = os.path.join(exp_root, entry)
                if os.path.isdir(full):
                    search_dirs.append(full)

    domains = []
    n_classes_list = []
    deltas = []

    for name, info in DOMAIN_DATA.items():
        base_result = find_result(search_dirs, info["base_dir_pattern"]) if search_dirs else None
        hybrid_result = find_result(search_dirs, info["hybrid_dir_pattern"]) if search_dirs else None

        if base_result and hybrid_result:
            base_f1 = base_result.get("entity_level", {}).get("f1", 0)
            hybrid_f1 = hybrid_result.get("entity_level", {}).get("f1", 0)
            delta = hybrid_f1 - base_f1
        elif args.manual:
            # Fallback to hardcoded values from notes
            manual_deltas = {
                "BC5CDR": 0.0117,   # 0.8881 - 0.8764
                "BUSTER": 0.0137,   # 0.6872 - 0.6735
                "PII": 0.0187,      # from previous experiments
                "FiNER-139": -0.1102,  # 0.5540 - 0.6642
            }
            delta = manual_deltas.get(name)
            if delta is None:
                continue
        else:
            logger.warning(f"  {name}: results not found, skipping")
            continue

        domains.append(name)
        n_classes_list.append(info["n_classes"])
        deltas.append(delta)
        logger.info(f"  {name}: n_classes={info['n_classes']}, delta={delta:+.4f}")

    if len(domains) < 2:
        logger.error("Need at least 2 domains for correlation analysis")
        return

    # Correlation analysis
    log_n = np.log(n_classes_list)
    deltas_arr = np.array(deltas)

    pearson_r, pearson_p = stats.pearsonr(log_n, deltas_arr)
    spearman_r, spearman_p = stats.spearmanr(log_n, deltas_arr)

    print(f"\n{'='*60}")
    print("BOUNDARY CONDITION ANALYSIS")
    print(f"{'='*60}")
    print(f"\n{'Domain':<15} {'n_classes':>10} {'log(n)':>8} {'Delta Entity-F1':>16}")
    print(f"{'-'*55}")
    for d, n, delta in zip(domains, n_classes_list, deltas):
        print(f"{d:<15} {n:>10} {np.log(n):>8.2f} {delta:>+16.4f}")

    print(f"\nPearson  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    print(f"Spearman r = {spearman_r:.4f}, p = {spearman_p:.4f}")

    # Linear regression on log(n_classes)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, deltas_arr)
    print(f"\nLinear fit: delta = {slope:.4f} * ln(n_classes) + {intercept:.4f}")
    print(f"  R^2 = {r_value**2:.4f}, p = {p_value:.4f}")

    # Estimated crossover point (delta = 0)
    if slope != 0:
        crossover_log = -intercept / slope
        crossover_n = np.exp(crossover_log)
        print(f"  Crossover (delta=0): ~{crossover_n:.0f} classes (ln={crossover_log:.2f})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data points
    colors = []
    for d in deltas:
        colors.append("#2ecc71" if d > 0 else "#e74c3c")

    ax.scatter(log_n, deltas_arr, c=colors, s=120, zorder=5, edgecolors="black", linewidth=0.5)

    # Labels
    for d, x, y in zip(domains, log_n, deltas_arr):
        offset = (0.1, 0.003) if y > 0 else (0.1, -0.008)
        ax.annotate(d, (x, y), xytext=(x + offset[0], y + offset[1]), fontsize=10)

    # Regression line
    x_fit = np.linspace(min(log_n) - 0.3, max(log_n) + 0.3, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.7, linewidth=1.5,
            label=f"$\\Delta$ = {slope:.4f} ln(n) + {intercept:.4f}\n$R^2$ = {r_value**2:.3f}")

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)

    # Shading
    ax.axhspan(0, max(deltas) * 1.3, alpha=0.05, color="green")
    ax.axhspan(min(deltas) * 1.3, 0, alpha=0.05, color="red")

    ax.set_xlabel("ln(Number of Entity Classes)", fontsize=12)
    ax.set_ylabel("$\\Delta$ Entity-F1 (Hybrid - Base)", fontsize=12)
    ax.set_title("NerGuard: LLM Routing Effectiveness vs Task Complexity", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")

    # Custom x-ticks showing both log and actual values
    tick_positions = log_n
    tick_labels = [f"{int(n)}\n(ln={np.log(n):.1f})" for n in n_classes_list]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info(f"\nPlot saved to {args.output}")

    # Save analysis data
    analysis = {
        "domains": domains,
        "n_classes": n_classes_list,
        "deltas": deltas,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "linear_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
        },
    }
    analysis_path = args.output.replace(".png", "_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
