#!/usr/bin/env python
"""
Evaluation script for quantized ONNX models.

Usage:
    python -m src.scripts.evaluate_quantized
    python -m src.scripts.evaluate_quantized --compare-baseline
    python -m src.scripts.evaluate_quantized --sample-limit 500 --output-dir ./results
"""

import os
import time
import argparse
import logging
import warnings

import torch
import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification
from tqdm import tqdm

from src.core.constants import DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH
from src.visualization.optimization_plots import (
    plot_quantization_metrics,
    plot_quantization_radar,
    save_quantization_report,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuantizedEval")

DEFAULT_QUANTIZED_DIR = "./plots/quantization_plots"
DEFAULT_QUANTIZED_MODEL = "model_quantized.onnx"
DEFAULT_OUTPUT_DIR = "./plots/quantized_evaluation"


def get_model_size(path):
    """Get model size in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def evaluate_onnx_model(model, tokenizer, dataset, desc="Evaluating ONNX"):
    """Evaluate an ONNX model on the dataset."""
    latencies = []
    all_preds = []
    all_labels = []

    # Warmup
    dummy_text = "Warmup routine for stable latency measurement."
    dummy_inputs = tokenizer(
        dummy_text, return_tensors="np", padding=True, truncation=True
    )
    dummy_inputs = {k: v.astype(np.int64) for k, v in dummy_inputs.items()}
    for _ in range(10):
        _ = model(**dummy_inputs)

    logger.info(f"--- {desc} ---")

    for sample in tqdm(dataset, desc=desc):
        labels = sample["labels"]

        inputs = {
            "input_ids": np.array([sample["input_ids"]], dtype=np.int64),
            "attention_mask": np.array([sample["attention_mask"]], dtype=np.int64),
        }

        start_time = time.perf_counter()
        outputs = model(**inputs)
        end_time = time.perf_counter()

        predictions = np.argmax(outputs.logits, axis=-1)[0]
        latencies.append((end_time - start_time) * 1000)

        active_labels = [l for l in labels if l != -100]
        active_preds = [p for p, l in zip(predictions, labels) if l != -100]

        all_labels.extend(active_labels)
        all_preds.extend(active_preds)

    return {
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
    }


def evaluate_pytorch_model(model, tokenizer, dataset, device="cpu", desc="Evaluating PyTorch"):
    """Evaluate a PyTorch model on the dataset."""
    model.to(device)
    model.eval()
    latencies = []
    all_preds = []
    all_labels = []

    # Warmup
    dummy_text = "Warmup routine for stable latency measurement."
    dummy_inputs = tokenizer(
        dummy_text, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(**dummy_inputs)

    logger.info(f"--- {desc} ---")

    for sample in tqdm(dataset, desc=desc):
        labels = sample["labels"]

        input_ids = torch.tensor([sample["input_ids"]], device=device)
        attention_mask = torch.tensor([sample["attention_mask"]], device=device)

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        end_time = time.perf_counter()

        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
        latencies.append((end_time - start_time) * 1000)

        active_labels = [l for l in labels if l != -100]
        active_preds = [p for p, l in zip(predictions, labels) if l != -100]

        all_labels.extend(active_labels)
        all_preds.extend(active_preds)

    return {
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            all_labels, all_preds, average="weighted", zero_division=0
        ),
    }


def print_results(results, model_name, size_mb):
    """Print evaluation results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Results for {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Model Size:    {size_mb:.2f} MB")
    logger.info(f"  Latency:       {results['latency']:.2f} ms (+/- {results['latency_std']:.2f})")
    logger.info(f"  F1 Score:      {results['f1']:.4f}")
    logger.info(f"  Precision:     {results['precision']:.4f}")
    logger.info(f"  Recall:        {results['recall']:.4f}")
    logger.info(f"{'='*60}\n")


def print_comparison(baseline_results, quantized_results, baseline_size, quantized_size):
    """Print comparison between baseline and quantized models."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{'Metric':<20} {'Baseline (FP32)':<20} {'Quantized (INT8)':<20} {'Improvement':<20}")
    logger.info(f"{'-'*80}")

    speedup = baseline_results['latency'] / quantized_results['latency']
    logger.info(
        f"{'Latency (ms)':<20} {baseline_results['latency']:<20.2f} {quantized_results['latency']:<20.2f} {speedup:<.2f}x faster"
    )

    size_reduction = (1 - quantized_size / baseline_size) * 100
    logger.info(
        f"{'Size (MB)':<20} {baseline_size:<20.2f} {quantized_size:<20.2f} {size_reduction:<.1f}% smaller"
    )

    f1_diff = quantized_results['f1'] - baseline_results['f1']
    logger.info(
        f"{'F1 Score':<20} {baseline_results['f1']:<20.4f} {quantized_results['f1']:<20.4f} {f1_diff:<+.4f}"
    )

    prec_diff = quantized_results['precision'] - baseline_results['precision']
    logger.info(
        f"{'Precision':<20} {baseline_results['precision']:<20.4f} {quantized_results['precision']:<20.4f} {prec_diff:<+.4f}"
    )

    recall_diff = quantized_results['recall'] - baseline_results['recall']
    logger.info(
        f"{'Recall':<20} {baseline_results['recall']:<20.4f} {quantized_results['recall']:<20.4f} {recall_diff:<+.4f}"
    )
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized ONNX model")

    parser.add_argument(
        "--quantized-model-dir",
        type=str,
        default=DEFAULT_QUANTIZED_DIR,
        help="Directory containing the quantized ONNX model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_QUANTIZED_MODEL,
        help="Name of the quantized ONNX model file",
    )
    parser.add_argument(
        "--original-model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to original PyTorch model (for tokenizer and baseline comparison)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="Maximum samples to evaluate (0 for all)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also evaluate baseline PyTorch model for comparison",
    )

    args = parser.parse_args()

    # Validate paths
    quantized_model_path = os.path.join(args.quantized_model_dir, args.model_name)
    if not os.path.exists(quantized_model_path):
        logger.error(f"Quantized model not found at {quantized_model_path}")
        return

    if not os.path.exists(args.data_path):
        logger.error(f"Dataset not found at {args.data_path}")
        return

    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)["validation"]
    if args.sample_limit > 0:
        dataset = dataset.select(range(min(args.sample_limit, len(dataset))))
    logger.info(f"  Loaded {len(dataset)} samples")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.original_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_path)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate quantized ONNX model
    logger.info(f"Loading quantized model from {quantized_model_path}...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = os.cpu_count()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    quantized_model = ORTModelForTokenClassification.from_pretrained(
        args.quantized_model_dir,
        file_name=args.model_name,
        provider="CPUExecutionProvider",
        session_options=session_options,
    )

    quantized_results = evaluate_onnx_model(
        quantized_model, tokenizer, dataset, desc="Evaluating Quantized INT8"
    )
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)

    print_results(quantized_results, "Quantized (INT8)", quantized_size)

    # Optionally evaluate baseline
    if args.compare_baseline:
        logger.info(f"Loading baseline model from {args.original_model_path}...")
        baseline_model = AutoModelForTokenClassification.from_pretrained(
            args.original_model_path
        )
        baseline_results = evaluate_pytorch_model(
            baseline_model, tokenizer, dataset, device="cpu", desc="Evaluating Baseline FP32"
        )
        baseline_size = get_model_size(args.original_model_path)

        print_results(baseline_results, "Baseline (FP32)", baseline_size)
        print_comparison(baseline_results, quantized_results, baseline_size, quantized_size)

        # Generate plots and report
        metrics = {
            "Original (FP32)": {**baseline_results, "size": baseline_size},
            "Quantized (INT8)": {**quantized_results, "size": quantized_size},
        }

        plot_quantization_metrics(metrics, args.output_dir)
        plot_quantization_radar(metrics, args.output_dir)
        save_quantization_report(metrics, args.output_dir)

        del baseline_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
