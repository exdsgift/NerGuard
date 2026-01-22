import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/processed/tokenized_data"
OUTPUT_DIR = "./optimization_plots"
BATCH_SIZE = 32
SAMPLE_LIMIT = None

LLM_COST_PER_TOKEN = 0.00002  # GPT-4 Turbo pricing $0.02/1K tokens
ERROR_COST_PER_TOKEN = 0.001  # Cost violantion per token GDPR (example value)

# F-beta: precision-recall tradeoff
BETA_SCORE = 0.5

# Grid search
GRID_RESOLUTION = 30

# Bootstrap parameters
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


@dataclass
class OptimizationResult:
    ent_threshold: float
    conf_threshold: float
    precision: float
    recall: float
    f_score: float
    intervention_rate: float
    expected_cost: float
    cost_savings: float
    precision_ci: Tuple[float, float] = None
    recall_ci: Tuple[float, float] = None
    f_score_ci: Tuple[float, float] = None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif", "Times", "serif"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


def collect_inference_stats_batched(model, dataset, device):
    all_entropies = []
    all_confidences = []
    all_is_error = []

    print("   Running Batched Inference...")

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Inference"):
        batch = dataset[i : i + BATCH_SIZE]

        max_len = max(len(x) for x in batch["input_ids"])

        input_ids = torch.tensor(
            [x + [0] * (max_len - len(x)) for x in batch["input_ids"]]
        ).to(device)

        attention_mask = torch.tensor(
            [x + [0] * (max_len - len(x)) for x in batch["attention_mask"]]
        ).to(device)

        labels = torch.tensor(
            [x + [-100] * (max_len - len(x)) for x in batch["labels"]]
        ).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        probs = F.softmax(outputs.logits, dim=-1)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        entropy = -torch.sum(probs * log_probs, dim=-1)
        confidences, pred_ids = torch.max(probs, dim=-1)

        mask = labels != -100
        is_error = (pred_ids != labels) & mask

        for b in range(len(batch["input_ids"])):
            valid_mask = mask[b]
            if valid_mask.sum() == 0:
                continue

            all_entropies.extend(entropy[b][valid_mask].cpu().numpy())
            all_confidences.extend(confidences[b][valid_mask].cpu().numpy())
            all_is_error.extend(is_error[b][valid_mask].cpu().numpy())

    return np.array(all_entropies), np.array(all_confidences), np.array(all_is_error)


def bootstrap_metric(trigger_mask, is_errors, metric_fn, n_bootstrap=N_BOOTSTRAP):
    metrics = []
    n_samples = len(is_errors)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sampled_triggers = trigger_mask[indices]
        sampled_errors = is_errors[indices]

        metric_value = metric_fn(sampled_triggers, sampled_errors)
        metrics.append(metric_value)

    metrics = np.array(metrics)
    alpha = 1 - CONFIDENCE_LEVEL
    ci_low = np.percentile(metrics, 100 * alpha / 2)
    ci_high = np.percentile(metrics, 100 * (1 - alpha / 2))

    return ci_low, ci_high


def calculate_expected_cost(intervention_rate, recall, n_errors, n_total):
    n_llm_calls = intervention_rate * n_total
    n_missed_errors = (1 - recall) * n_errors

    llm_cost = n_llm_calls * LLM_COST_PER_TOKEN
    error_cost = n_missed_errors * ERROR_COST_PER_TOKEN

    return llm_cost + error_cost


def vectorized_grid_search(entropies, confidences, is_errors):
    """Grid search with vectorization for speed."""
    print("\n   Calculating Optimization Matrix...")

    ent_grid = np.linspace(0.1, 1.5, GRID_RESOLUTION)
    conf_grid = np.linspace(0.5, 0.99, GRID_RESOLUTION)

    results = {
        "f_score": np.zeros((len(ent_grid), len(conf_grid))),
        "precision": np.zeros((len(ent_grid), len(conf_grid))),
        "recall": np.zeros((len(ent_grid), len(conf_grid))),
        "intervention_rate": np.zeros((len(ent_grid), len(conf_grid))),
        "expected_cost": np.zeros((len(ent_grid), len(conf_grid))),
        "specificity": np.zeros((len(ent_grid), len(conf_grid))),  # Per ROC
    }

    total_samples = len(is_errors)
    total_errors = np.sum(is_errors)
    total_correct = total_samples - total_errors

    baseline_cost = total_samples * LLM_COST_PER_TOKEN

    if total_errors == 0:
        raise ValueError("Dataset empty of errors; cannot optimize thresholds.")

    for i, ent_th in enumerate(tqdm(ent_grid, desc="Grid Search")):
        for j, conf_th in enumerate(conf_grid):
            trigger_mask = (entropies > ent_th) & (confidences < conf_th)
            n_triggers = np.sum(trigger_mask)

            if n_triggers == 0:
                results["precision"][i, j] = 0
                results["recall"][i, j] = 0
                results["f_score"][i, j] = 0
                results["intervention_rate"][i, j] = 0
                results["expected_cost"][i, j] = total_errors * ERROR_COST_PER_TOKEN
                results["specificity"][i, j] = 1.0  # No false positives
                continue

            tp = np.sum(trigger_mask & is_errors)
            fp = np.sum(trigger_mask & ~is_errors)
            fn = np.sum(~trigger_mask & is_errors)
            tn = np.sum(~trigger_mask & ~is_errors)

            precision = tp / n_triggers if n_triggers > 0 else 0
            recall = tp / total_errors
            specificity = tn / total_correct if total_correct > 0 else 0

            # F-beta
            if precision + recall > 0:
                beta_sq = BETA_SCORE**2
                f_score = (
                    (1 + beta_sq)
                    * (precision * recall)
                    / ((beta_sq * precision) + recall)
                )
            else:
                f_score = 0

            intervention_rate = n_triggers / total_samples
            expected_cost = calculate_expected_cost(
                intervention_rate, recall, total_errors, total_samples
            )

            results["f_score"][i, j] = f_score
            results["precision"][i, j] = precision
            results["recall"][i, j] = recall
            results["intervention_rate"][i, j] = intervention_rate
            results["expected_cost"][i, j] = expected_cost
            results["specificity"][i, j] = specificity

    results["baseline_cost"] = baseline_cost
    return ent_grid, conf_grid, results


def plot_heatmap(ent_grid, conf_grid, matrix, best_idx, metric_name, filename):
    """Heatmap publication-quality."""
    plt.figure(figsize=(10, 8))

    X, Y = np.meshgrid(conf_grid, ent_grid)
    c = plt.pcolormesh(X, Y, matrix, shading="auto", cmap="viridis")
    cbar = plt.colorbar(c, label=metric_name)

    best_i, best_j = best_idx
    best_ent = ent_grid[best_i]
    best_conf = conf_grid[best_j]
    plt.scatter(
        best_conf,
        best_ent,
        color="red",
        s=150,
        edgecolors="white",
        linewidths=2,
        label="Optimal Point",
        marker="*",
        zorder=5,
    )

    plt.title(
        f"Optimization Surface: {metric_name}\n(β={BETA_SCORE} F-Score Optimization)",
        pad=15,
        fontweight="bold",
    )
    plt.xlabel("Confidence Threshold (trigger if < x)", fontsize=12)
    plt.ylabel("Entropy Threshold (trigger if > y)", fontsize=12)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   [PLOT] Saved: {filename}")


def plot_pareto_frontier(results, ent_grid, conf_grid, output_path):
    plt.figure(figsize=(10, 7))

    precision_flat = results["precision"].flatten()
    recall_flat = results["recall"].flatten()

    valid = (precision_flat > 0) | (recall_flat > 0)
    precision_flat = precision_flat[valid]
    recall_flat = recall_flat[valid]

    plt.scatter(
        recall_flat,
        precision_flat,
        alpha=0.3,
        s=20,
        c="blue",
        label="All configurations",
    )

    sorted_indices = np.argsort(recall_flat)
    pareto_recall = []
    pareto_precision = []
    max_precision = 0

    for idx in sorted_indices:
        if precision_flat[idx] >= max_precision:
            pareto_recall.append(recall_flat[idx])
            pareto_precision.append(precision_flat[idx])
            max_precision = precision_flat[idx]

    plt.plot(
        pareto_recall,
        pareto_precision,
        "r-",
        linewidth=2,
        label="Pareto Frontier",
        zorder=5,
    )

    plt.xlabel("Recall (Error Detection Rate)", fontsize=12)
    plt.ylabel("Precision (Trigger Accuracy)", fontsize=12)
    plt.title(
        "Precision-Recall Tradeoff\n(Pareto-Optimal Configurations)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   [PLOT] Saved: {output_path}")


def optimize():
    print(f"=" * 70)
    print("   ACADEMIC HYPERPARAMETER OPTIMIZATION ENGINE")
    print(
        f"   Seed: {RANDOM_SEED} | β: {BETA_SCORE} | CI: {CONFIDENCE_LEVEL * 100:.0f}%\n\n"
    )

    ensure_dir(OUTPUT_DIR)
    set_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n   Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    dataset = load_from_disk(DATA_PATH)["validation"]

    if SAMPLE_LIMIT:
        print(f"   - Using only {SAMPLE_LIMIT} samples!")
        dataset = dataset.select(range(SAMPLE_LIMIT))

    entropies, confidences, is_errors = collect_inference_stats_batched(
        model, dataset, device
    )
    n_errors = np.sum(is_errors)
    n_total = len(is_errors)
    error_rate = n_errors / n_total

    print(f"   Tokens used: {n_total:,}")
    print(f"   Error find: {n_errors:,} ({error_rate:.2%})")
    print(
        f"   - entropy distrib: μ={np.mean(entropies):.3f}, σ={np.std(entropies):.3f}"
    )
    print(
        f"   - confidence distrib: μ={np.mean(confidences):.3f}, σ={np.std(confidences):.3f}"
    )

    # Grid Search
    ent_grid, conf_grid, results = vectorized_grid_search(
        entropies, confidences, is_errors
    )

    max_idx = np.unravel_index(np.argmax(results["f_score"]), results["f_score"].shape)

    best_ent = ent_grid[max_idx[0]]
    best_conf = conf_grid[max_idx[1]]

    trigger_mask = (entropies > best_ent) & (confidences < best_conf)

    tp = np.sum(trigger_mask & is_errors)
    fp = np.sum(trigger_mask & ~is_errors)
    fn = np.sum(~trigger_mask & is_errors)
    tn = np.sum(~trigger_mask & ~is_errors)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = results["f_score"][max_idx]
    intervention_rate = np.sum(trigger_mask) / n_total

    baseline_cost = results["baseline_cost"]
    expected_cost = results["expected_cost"][max_idx]
    cost_savings = baseline_cost - expected_cost

    # Bootstrap CI
    def precision_fn(triggers, errors):
        tp = np.sum(triggers & errors)
        return tp / np.sum(triggers) if np.sum(triggers) > 0 else 0

    def recall_fn(triggers, errors):
        tp = np.sum(triggers & errors)
        return tp / np.sum(errors) if np.sum(errors) > 0 else 0

    def fscore_fn(triggers, errors):
        p = precision_fn(triggers, errors)
        r = recall_fn(triggers, errors)
        if p + r == 0:
            return 0
        beta_sq = BETA_SCORE**2
        return (1 + beta_sq) * (p * r) / ((beta_sq * p) + r)

    precision_ci = bootstrap_metric(trigger_mask, is_errors, precision_fn)
    recall_ci = bootstrap_metric(trigger_mask, is_errors, recall_fn)
    f_score_ci = bootstrap_metric(trigger_mask, is_errors, fscore_fn)

    result = OptimizationResult(
        ent_threshold=best_ent,
        conf_threshold=best_conf,
        precision=precision,
        recall=recall,
        f_score=f_score,
        intervention_rate=intervention_rate,
        expected_cost=expected_cost,
        cost_savings=cost_savings,
        precision_ci=precision_ci,
        recall_ci=recall_ci,
        f_score_ci=f_score_ci,
        n_samples=n_total,
        n_errors=n_errors,
        n_triggers=int(np.sum(trigger_mask)),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
    )

    print("\n" + "=" * 70)
    print("   RISULTATI FINALI (PUBLICATION-READY)")
    print("=" * 70)
    print(f"\n   Soglie Ottimali:")
    print(f"   • Entropy:    > {best_ent:.3f}")
    print(f"   • Confidence: < {best_conf:.3f}")
    print(f"\n   Metriche di Performance:")
    print(
        f"   • Precision:  {precision:.3f} (95% CI: [{precision_ci[0]:.3f}, {precision_ci[1]:.3f}])"
    )
    print(
        f"   • Recall:     {recall:.3f} (95% CI: [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}])"
    )
    print(
        f"   • F{BETA_SCORE}-Score:  {f_score:.3f} (95% CI: [{f_score_ci[0]:.3f}, {f_score_ci[1]:.3f}])"
    )
    print(f"\n   Metriche Economiche:")
    print(f"   • Intervention Rate: {intervention_rate:.2%}")
    print(f"   • Expected Cost:     ${expected_cost:.2f}")
    print(
        f"   • Cost Savings:      ${cost_savings:.2f} ({(cost_savings / baseline_cost) * 100:.1f}%)"
    )
    print(f"\n   Confusion Matrix:")
    print(f"   • True Positives:  {tp:,} (errori rilevati correttamente)")
    print(f"   • False Positives: {fp:,} (falsi allarmi)")
    print(f"   • False Negatives: {fn:,} (errori mancati)")
    print(f"   • True Negatives:  {tn:,}")
    print("=" * 70)

    print("\n[6/6] Generating publication-quality plots...")
    plot_heatmap(
        ent_grid,
        conf_grid,
        results["f_score"],
        max_idx,
        f"F{BETA_SCORE}-Score",
        f"{OUTPUT_DIR}/opt_fscore_heatmap.png",
    )
    plot_heatmap(
        ent_grid,
        conf_grid,
        results["precision"],
        max_idx,
        "Precision",
        f"{OUTPUT_DIR}/opt_precision_heatmap.png",
    )
    plot_heatmap(
        ent_grid,
        conf_grid,
        results["expected_cost"],
        max_idx,
        "Expected Cost ($)",
        f"{OUTPUT_DIR}/opt_cost_heatmap.png",
    )
    plot_pareto_frontier(
        results, ent_grid, conf_grid, f"{OUTPUT_DIR}/pareto_frontier.png"
    )

    def convert_to_native(obj):
        """Converte NumPy types a Python nativi per JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(convert_to_native(item) for item in obj)
        return obj

    result_dict = asdict(result)
    result_dict_native = {k: convert_to_native(v) for k, v in result_dict.items()}

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(result_dict_native, f, indent=2)

    # TXT leggibile
    with open(f"{OUTPUT_DIR}/optimal_params.txt", "w") as f:
        f.write(f"# Optimal Hyperparameters (F{BETA_SCORE}-Score Optimization)\n")
        f.write(f"# Random Seed: {RANDOM_SEED}\n")
        f.write(f"# Dataset Size: {n_total:,} tokens\n\n")
        for key, value in asdict(result).items():
            f.write(f"{key}: {value}\n")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("JSON export: results.json (per riproducibilità)")

    return result


if __name__ == "__main__":
    result = optimize()
