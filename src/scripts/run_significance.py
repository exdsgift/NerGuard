"""Run significance tests across all domains.

Loads predictions.json from experiment directories and runs:
1. Paired bootstrap test (entity-level F1)
2. McNemar's test (token-level correctness)

Usage:
    uv run python -m src.scripts.run_significance
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Significance")

# Domain configurations: (base_dir_pattern, hybrid_dir_pattern, domain_name)
DOMAIN_CONFIGS = [
    {
        "name": "PII (nvidia-pii)",
        "base_pattern": "nerguard_base_nvidia-pii",
        "hybrid_pattern": "nerguard_hybrid_v2_gpt-4o_nvidia-pii",
    },
    {
        "name": "Biomedical (BC5CDR)",
        "base_pattern": "biomedical_base_bc5cdr",
        "hybrid_pattern": "biomedical_hybrid_gpt-4o_bc5cdr",
    },
    {
        "name": "Financial (BUSTER)",
        "base_pattern": "financial_base_buster",
        "hybrid_pattern": "financial_hybrid_gpt-4o_buster",
    },
    {
        "name": "Financial (FiNER-139)",
        "base_pattern": "finer139_base_finer-139",
        "hybrid_pattern": "finer139_hybrid_gpt-4o_finer-139",
    },
]


def find_predictions(experiment_dir: str, pattern: str) -> str | None:
    """Find predictions.json matching pattern in experiment directory."""
    for entry in os.listdir(experiment_dir):
        if pattern in entry:
            pred_path = os.path.join(experiment_dir, entry, "predictions.json")
            if os.path.exists(pred_path):
                return pred_path
    return None


def load_predictions(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["y_true"], data["y_pred"]


def main():
    from src.analysis.significance import mcnemar_test, paired_bootstrap_test

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dirs", nargs="+", help="Experiment directories to search")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    # Search in default locations if not specified
    if args.experiment_dirs:
        search_dirs = args.experiment_dirs
    else:
        search_dirs = []
        exp_root = "experiments"
        if os.path.exists(exp_root):
            for entry in sorted(os.listdir(exp_root)):
                full = os.path.join(exp_root, entry)
                if os.path.isdir(full):
                    search_dirs.append(full)

    if not search_dirs:
        logger.error("No experiment directories found")
        return

    logger.info(f"Searching {len(search_dirs)} experiment directories")

    results = []

    for domain in DOMAIN_CONFIGS:
        base_path = None
        hybrid_path = None

        for exp_dir in search_dirs:
            if base_path is None:
                base_path = find_predictions(exp_dir, domain["base_pattern"])
            if hybrid_path is None:
                hybrid_path = find_predictions(exp_dir, domain["hybrid_pattern"])

        if base_path is None or hybrid_path is None:
            logger.warning(f"  {domain['name']}: Missing predictions (base={base_path is not None}, hybrid={hybrid_path is not None})")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Domain: {domain['name']}")
        logger.info(f"  Base:   {base_path}")
        logger.info(f"  Hybrid: {hybrid_path}")

        y_true_base, y_pred_base = load_predictions(base_path)
        y_true_hybrid, y_pred_hybrid = load_predictions(hybrid_path)

        # Verify same ground truth
        assert len(y_true_base) == len(y_true_hybrid), "Sample count mismatch"

        # Paired bootstrap
        bootstrap = paired_bootstrap_test(
            y_true=y_true_base,
            y_pred_a=y_pred_base,
            y_pred_b=y_pred_hybrid,
            n_bootstrap=args.n_bootstrap,
        )

        # McNemar's test
        mcnemar = mcnemar_test(
            y_true=y_true_base,
            y_pred_a=y_pred_base,
            y_pred_b=y_pred_hybrid,
        )

        sig_bootstrap = "***" if bootstrap["p_value"] < 0.001 else "**" if bootstrap["p_value"] < 0.01 else "*" if bootstrap["p_value"] < 0.05 else "n.s."
        sig_mcnemar = "***" if mcnemar["p_value"] < 0.001 else "**" if mcnemar["p_value"] < 0.01 else "*" if mcnemar["p_value"] < 0.05 else "n.s."

        print(f"\n  Paired Bootstrap (Entity-F1, n={bootstrap['n_samples']}, B={args.n_bootstrap}):")
        print(f"    Base mean:   {bootstrap['scores_a_mean']:.4f}")
        print(f"    Hybrid mean: {bootstrap['scores_b_mean']:.4f}")
        print(f"    Delta:       {bootstrap['mean_delta']:+.4f}")
        print(f"    95% CI:      [{bootstrap['ci_lower']:+.4f}, {bootstrap['ci_upper']:+.4f}]")
        print(f"    p-value:     {bootstrap['p_value']:.4f} {sig_bootstrap}")

        print(f"\n  McNemar's Test (token-level, n={mcnemar['n_tokens']} tokens):")
        print(f"    Base correct, Hybrid wrong: {mcnemar['n_a_correct_b_wrong']}")
        print(f"    Hybrid correct, Base wrong: {mcnemar['n_b_correct_a_wrong']}")
        print(f"    Chi2:    {mcnemar['chi2']:.2f}")
        print(f"    p-value: {mcnemar['p_value']:.4f} {sig_mcnemar}")

        results.append({
            "domain": domain["name"],
            "n_samples": bootstrap["n_samples"],
            "bootstrap": bootstrap,
            "mcnemar": mcnemar,
        })

    # Summary table
    if results:
        print(f"\n{'='*90}")
        print("SIGNIFICANCE TEST SUMMARY")
        print(f"{'='*90}")
        print(f"{'Domain':<25} {'Delta':>8} {'Bootstrap p':>13} {'McNemar p':>12} {'n_samples':>10}")
        print(f"{'-'*90}")
        for r in results:
            b = r["bootstrap"]
            m = r["mcnemar"]
            print(
                f"{r['domain']:<25} {b['mean_delta']:>+8.4f} "
                f"{b['p_value']:>13.4f} {m['p_value']:>12.4f} {r['n_samples']:>10}"
            )

        # Save results
        out_path = "experiments/significance_results.json"
        os.makedirs("experiments", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
