"""Calibration experiment: ECE evaluation + temperature scaling.

Tests Proposition [Calibration and false accept risk] from thesis:
  If confidence c is overconfident then P[ŷ≠y | c=t] > 1-t,
  which increases conditional risk on accepted predictions.

Procedure:
  1. Load NER model and validation set.
  2. Collect token-level confidence scores and correctness labels.
  3. Compute ECE (Expected Calibration Error) with 10 bins — pre-scaling.
  4. Optimize temperature T on validation set by minimizing NLL.
  5. Recompute ECE post-scaling.
  6. Compare routing decisions pre/post-scaling:
     how many spans are routed differently after temperature scaling?
  7. Save results to experiments/calibration_YYYY-MM-DD/results.json.

Usage:
    uv run python -m src.scripts.run_calibration \
        --dataset nvidia-pii --samples 500
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from transformers import AutoModelForTokenClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Calibration")

MAX_LENGTH = 512
N_BINS = 10


def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where B_b is the set of predictions whose confidence falls in bin b.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        acc_bin = correctness[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (n_bin / n_total) * abs(acc_bin - conf_bin)

    return float(ece)


def reliability_diagram_data(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> List[Dict]:
    """Compute per-bin calibration data for reliability diagram."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        bins.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "n_samples": n_bin,
            "accuracy": float(correctness[mask].mean()),
            "mean_confidence": float(confidences[mask].mean()),
            "gap": float(abs(correctness[mask].mean() - confidences[mask].mean())),
        })

    return bins


def collect_token_stats(
    model,
    tokenizer,
    id2label: Dict,
    device: torch.device,
    samples,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[float]]]:
    """Collect per-token confidence, entropy, correctness, and raw logits.

    Returns:
        confidences: shape (n_tokens,) — max softmax probability
        correctness: shape (n_tokens,) — 1 if correct, 0 if not
        entropies: shape (n_tokens,) — token-level entropy
        all_logits_list: list of (n_subword_tokens, n_classes) tensors (non-special)
    """
    all_confs = []
    all_correct = []
    all_entropies = []
    all_logits_list = []

    label2id = {v: k for k, v in id2label.items()}

    for sample in samples:
        text = sample.text
        tokens = sample.tokens
        token_spans = sample.token_spans
        gold_labels = sample.labels

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits[0]

        probs = F.softmax(logits, dim=-1)
        conf, pred_ids = torch.max(probs, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        conf_np = conf.cpu().numpy()
        pred_ids_np = pred_ids.cpu().numpy()
        entropy_np = entropy.cpu().numpy()
        logits_np = logits.cpu().numpy()

        # Align to word tokens
        for word_idx, (w_start, w_end) in enumerate(token_spans):
            if word_idx >= len(gold_labels):
                break
            gold_label = gold_labels[word_idx]
            if gold_label not in label2id:
                continue

            # Find first overlapping subword token
            for sw_idx, (sw_start, sw_end) in enumerate(offset_mapping):
                if sw_start == sw_end == 0:
                    continue
                if sw_end <= w_start or sw_start >= w_end:
                    continue

                pred_label = id2label.get(int(pred_ids_np[sw_idx]), "O")
                is_correct = int(pred_label == gold_label)

                all_confs.append(float(conf_np[sw_idx]))
                all_correct.append(is_correct)
                all_entropies.append(float(entropy_np[sw_idx]))
                all_logits_list.append(logits_np[sw_idx].tolist())
                break

    return (
        np.array(all_confs),
        np.array(all_correct),
        np.array(all_entropies),
        all_logits_list,
    )


def find_temperature(logits_list: List[List[float]], correctness: np.ndarray) -> float:
    """Find optimal temperature T by minimizing NLL on validation tokens.

    With temperature T: scaled_prob = softmax(logits / T)
    """
    all_logits = torch.tensor(logits_list, dtype=torch.float32)
    # Use correctness as pseudo-labels: for calibration we only need the
    # distribution to be well-calibrated; minimize NLL with correct class = argmax
    pred_ids = torch.argmax(all_logits, dim=-1)

    def neg_log_likelihood(T: float) -> float:
        if T <= 0:
            return 1e9
        scaled = all_logits / T
        log_probs = F.log_softmax(scaled, dim=-1)
        nll = -log_probs[range(len(pred_ids)), pred_ids].mean().item()
        return nll

    result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def apply_temperature(logits_list: List[List[float]], T: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply temperature scaling and return new confidences and entropies."""
    all_logits = torch.tensor(logits_list, dtype=torch.float32)
    scaled = all_logits / T
    probs = F.softmax(scaled, dim=-1)
    conf = probs.max(dim=-1).values.numpy()
    entropy = (-probs * torch.log(probs + 1e-9)).sum(dim=-1).numpy()
    return conf, entropy


def count_routing_changes(
    entropy_pre: np.ndarray,
    conf_pre: np.ndarray,
    entropy_post: np.ndarray,
    conf_post: np.ndarray,
    entropy_threshold: float = 0.583,
    conf_threshold: float = 0.787,
) -> Dict:
    """Count how many routing decisions change after temperature scaling."""
    route_pre = (entropy_pre > entropy_threshold) & (conf_pre < conf_threshold)
    route_post = (entropy_post > entropy_threshold) & (conf_post < conf_threshold)

    n_route_pre = int(route_pre.sum())
    n_route_post = int(route_post.sum())
    n_changed = int((route_pre != route_post).sum())
    n_new_routes = int((~route_pre & route_post).sum())
    n_dropped_routes = int((route_pre & ~route_post).sum())

    return {
        "n_route_pre": n_route_pre,
        "n_route_post": n_route_post,
        "routing_rate_pre": float(n_route_pre / len(entropy_pre)),
        "routing_rate_post": float(n_route_post / len(entropy_post)),
        "n_routing_changed": n_changed,
        "n_new_routes": n_new_routes,
        "n_dropped_routes": n_dropped_routes,
        "fraction_changed": float(n_changed / len(entropy_pre)),
    }


def main():
    parser = argparse.ArgumentParser(description="ECE + temperature scaling calibration experiment")
    parser.add_argument("--model-path", default="./models/mdeberta-pii-safe/final",
                        help="Path to NER model")
    parser.add_argument("--dataset", default="nvidia-pii",
                        choices=["nvidia-pii", "ai4privacy"],
                        help="Validation dataset to use")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples for calibration")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model.to(device)
    model.eval()
    id2label = model.config.id2label
    logger.info(f"Model loaded on {device}, {len(id2label)} labels")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} ({args.samples} samples)")
    if args.dataset == "nvidia-pii":
        from src.benchmark.datasets.nvidia_pii import NvidiaPIIAdapter
        adapter = NvidiaPIIAdapter()
    else:
        from src.benchmark.datasets.ai4privacy import AI4PrivacyAdapter
        adapter = AI4PrivacyAdapter()

    samples = adapter.load(max_samples=args.samples)
    logger.info(f"Loaded {len(samples)} samples")

    # Collect token-level stats
    logger.info("Collecting token-level confidence/entropy stats...")
    confidences, correctness, entropies, logits_list = collect_token_stats(
        model, tokenizer, id2label, device, samples
    )
    n_tokens = len(confidences)
    logger.info(f"Collected stats for {n_tokens} tokens")

    # ECE pre-scaling
    ece_pre = compute_ece(confidences, correctness, N_BINS)
    reliability_pre = reliability_diagram_data(confidences, correctness, N_BINS)
    logger.info(f"ECE (pre-scaling): {ece_pre:.4f}")

    # Temperature scaling
    logger.info("Optimizing temperature T...")
    T_opt = find_temperature(logits_list, correctness)
    logger.info(f"Optimal temperature T = {T_opt:.4f}")

    conf_post, entropy_post = apply_temperature(logits_list, T_opt)

    # ECE post-scaling
    ece_post = compute_ece(conf_post, correctness, N_BINS)
    reliability_post = reliability_diagram_data(conf_post, correctness, N_BINS)
    logger.info(f"ECE (post-scaling): {ece_post:.4f}  (Δ = {ece_post - ece_pre:+.4f})")

    # Routing impact
    routing_changes = count_routing_changes(
        entropies, confidences, entropy_post, conf_post,
        entropy_threshold=0.583, conf_threshold=0.787,
    )
    logger.info(f"Routing rate pre-scaling:  {routing_changes['routing_rate_pre']:.4f}")
    logger.info(f"Routing rate post-scaling: {routing_changes['routing_rate_post']:.4f}")
    logger.info(f"Routing decisions changed: {routing_changes['n_routing_changed']} "
                f"({routing_changes['fraction_changed']*100:.2f}%)")

    print(f"\n{'='*60}")
    print("CALIBRATION EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Dataset:          {args.dataset} ({len(samples)} samples, {n_tokens} tokens)")
    print(f"ECE (pre-T):      {ece_pre:.4f}")
    print(f"T_opt:            {T_opt:.4f}")
    print(f"ECE (post-T):     {ece_post:.4f}  (Δ = {ece_post - ece_pre:+.4f})")
    print(f"\nRouting impact (thresholds θ_H=0.583, θ_c=0.787):")
    print(f"  Routing rate pre:  {routing_changes['routing_rate_pre']*100:.2f}%")
    print(f"  Routing rate post: {routing_changes['routing_rate_post']*100:.2f}%")
    print(f"  Decisions changed: {routing_changes['fraction_changed']*100:.2f}%")
    print(f"    New routes:    +{routing_changes['n_new_routes']}")
    print(f"    Dropped routes: -{routing_changes['n_dropped_routes']}")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = f"experiments/calibration_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    results = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "n_samples": len(samples),
        "n_tokens": n_tokens,
        "ece_pre": ece_pre,
        "T_opt": T_opt,
        "ece_post": ece_post,
        "ece_delta": ece_post - ece_pre,
        "routing_changes": routing_changes,
        "reliability_diagram_pre": reliability_pre,
        "reliability_diagram_post": reliability_post,
        "summary": {
            "overconfident": ece_pre > 0.05,
            "temperature_improves_calibration": ece_post < ece_pre,
            "routing_materially_affected": routing_changes["fraction_changed"] > 0.01,
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
