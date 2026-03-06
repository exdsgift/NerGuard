"""
Quantization pipeline for NerGuard ONNX models.

Supports multiple strategies:
  - dynamic-selective: Dynamic INT8, only MatMul operators
  - static:            Static INT8 with calibration data
  - static-mixed:      Static INT8 excluding sensitive layers

Usage:
    uv run python -m src.optimization.quantizer --strategy dynamic-selective
    uv run python -m src.optimization.quantizer --strategy static --calibration-samples 256
    uv run python -m src.optimization.quantizer --strategy static-mixed
"""

import os
import json
import time
import argparse
import logging
import warnings

import torch
import numpy as np
import onnx
import onnxruntime as ort
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from tqdm import tqdm

from src.core.constants import DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelQuant")

DEFAULT_OUTPUT_DIR = "./models/quantized_model"
SAMPLE_LIMIT = 1000


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_file_size_mb(path):
    """Get size of a file or directory in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def load_id2label(model_path):
    """Load id2label mapping from model config."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return {int(k): v for k, v in config["id2label"].items()}


def get_sensitive_nodes(onnx_model_path):
    """Identify MatMul nodes that should stay in FP32 for mixed-precision.

    Excludes:
      - Classifier head
      - First and last encoder layers (most sensitive to quantization)
      - Attention score computations (QK^T, position attention)
    """
    model = onnx.load(onnx_model_path)
    exclude = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        name = node.name
        # Classifier head
        if "classifier" in name.lower():
            exclude.append(name)
        # First and last encoder layers
        elif "/layer.0/" in name or "/layer_0/" in name:
            exclude.append(name)
        elif "/layer.11/" in name or "/layer_11/" in name:
            exclude.append(name)
        # Attention score computations (not projections or FFN)
        elif "/attention/" in name and "proj" not in name and "dense" not in name:
            exclude.append(name)
    logger.info(f"Mixed-precision: excluding {len(exclude)} sensitive MatMul nodes")
    return exclude


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_span_level(model, tokenizer, dataset, id2label, desc="Evaluating"):
    """Evaluate using span-level (seqeval) metrics — consistent with benchmark."""
    if hasattr(model, "to"):
        model.to("cpu")
    is_ort = isinstance(model, ORTModelForTokenClassification)

    latencies = []
    all_labels_seq = []
    all_preds_seq = []

    # Warmup
    dummy_text = "Warmup routine for stable latency measurement."
    if is_ort:
        dummy_inputs = tokenizer(
            dummy_text, return_tensors="np", padding=True, truncation=True
        )
        dummy_inputs = {k: v.astype(np.int64) for k, v in dummy_inputs.items()}
        for _ in range(10):
            model(**dummy_inputs)
    else:
        dummy_inputs = tokenizer(
            dummy_text, return_tensors="pt", padding=True, truncation=True
        )
        model.eval()
        for _ in range(10):
            with torch.no_grad():
                model(**dummy_inputs)

    logger.info(f"--- {desc} ---")

    for sample in tqdm(dataset, desc=desc):
        labels = sample["labels"]

        if is_ort:
            inputs = {
                "input_ids": np.array([sample["input_ids"]], dtype=np.int64),
                "attention_mask": np.array([sample["attention_mask"]], dtype=np.int64),
            }
            start = time.perf_counter()
            outputs = model(**inputs)
            end = time.perf_counter()
            preds = np.argmax(outputs.logits, axis=-1)[0]
        else:
            input_ids = torch.tensor([sample["input_ids"]])
            attention_mask = torch.tensor([sample["attention_mask"]])
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            end = time.perf_counter()
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

        latencies.append((end - start) * 1000)

        sample_labels = []
        sample_preds = []
        for p, l in zip(preds, labels):
            if l != -100:
                sample_labels.append(id2label[int(l)])
                sample_preds.append(id2label[int(p)])
        all_labels_seq.append(sample_labels)
        all_preds_seq.append(sample_preds)

    return {
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies),
        "f1": seqeval_f1(all_labels_seq, all_preds_seq, zero_division=0),
        "precision": seqeval_precision(all_labels_seq, all_preds_seq, zero_division=0),
        "recall": seqeval_recall(all_labels_seq, all_preds_seq, zero_division=0),
    }


# ── ONNX Export ──────────────────────────────────────────────────────────────


def export_to_onnx(model_path, output_dir):
    """Export PyTorch model to ONNX format."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info("Exporting to ONNX...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = os.cpu_count()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_model = ORTModelForTokenClassification.from_pretrained(
        model_path,
        export=True,
        provider="CPUExecutionProvider",
        session_options=session_options,
    )
    ort_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    onnx_path = os.path.join(output_dir, "model.onnx")
    logger.info(f"  ONNX model exported to {onnx_path}")
    return onnx_path


# ── Quantization Strategies ──────────────────────────────────────────────────


def quantize_dynamic_selective(output_dir, operators=None, nodes_to_exclude=None):
    """Exp 1: Dynamic INT8 quantization with selective operators."""
    if operators is None:
        operators = ["MatMul"]

    logger.info(f"Dynamic quantization: operators={operators}")
    qconfig = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=True,
        operators_to_quantize=operators,
    )

    quantizer = ORTQuantizer.from_pretrained(output_dir, file_name="model.onnx")

    if nodes_to_exclude:
        qconfig.nodes_to_exclude = nodes_to_exclude
        logger.info(f"  Excluding {len(nodes_to_exclude)} nodes")

    quantizer.quantize(
        save_dir=output_dir, quantization_config=qconfig, file_suffix="quantized"
    )
    quantized_path = os.path.join(output_dir, "model_quantized.onnx")
    logger.info(f"  Quantized model saved to {quantized_path}")
    return quantized_path


def quantize_static(output_dir, data_path, num_samples=256, operators=None,
                    nodes_to_exclude=None, calibration_method="MinMax"):
    """Exp 2/3: Static INT8 quantization with calibration data."""
    from optimum.onnxruntime.configuration import CalibrationConfig
    from onnxruntime.quantization import CalibrationMethod

    if operators is None:
        operators = ["MatMul"]

    logger.info(f"Static quantization: operators={operators}, "
                f"calibration_samples={num_samples}, method={calibration_method}")

    # Prepare calibration dataset
    logger.info("Preparing calibration dataset...")
    dataset = load_from_disk(data_path)["validation"]
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    # Keep only model input columns
    keep_cols = {"input_ids", "attention_mask"}
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    dataset = dataset.remove_columns(remove_cols)
    logger.info(f"  Calibration dataset: {len(dataset)} samples")

    # Create quantizer
    quantizer = ORTQuantizer.from_pretrained(output_dir, file_name="model.onnx")

    # Create calibration config
    method_map = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    cal_config = CalibrationConfig(
        dataset_name="nerguard-calibration",
        dataset_config_name="default",
        dataset_split="validation",
        dataset_num_samples=num_samples,
        method=method_map[calibration_method],
        num_bins=2048 if calibration_method == "Entropy" else None,
        percentile=99.999 if calibration_method == "Percentile" else None,
    )

    # Create quantization config
    qconfig = AutoQuantizationConfig.avx2(
        is_static=True,
        per_channel=True,
        operators_to_quantize=operators,
    )

    if nodes_to_exclude:
        qconfig.nodes_to_exclude = nodes_to_exclude
        logger.info(f"  Excluding {len(nodes_to_exclude)} nodes")

    # Run calibration
    logger.info("Running calibration...")
    ranges = quantizer.fit(
        dataset=dataset,
        calibration_config=cal_config,
        operators_to_quantize=operators,
        batch_size=8,
    )
    logger.info("  Calibration complete")

    # Quantize with calibration ranges
    logger.info("Quantizing with calibration ranges...")
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=qconfig,
        calibration_tensors_range=ranges,
        file_suffix="quantized",
    )

    quantized_path = os.path.join(output_dir, "model_quantized.onnx")
    logger.info(f"  Quantized model saved to {quantized_path}")
    return quantized_path


# ── Results Display ──────────────────────────────────────────────────────────


def print_comparison(fp32_results, int8_results, fp32_size, int8_size):
    """Print FP32 vs INT8 comparison table."""
    logger.info("\n" + "=" * 80)
    logger.info(f"{'Metric':<20} {'FP32':<18} {'INT8':<18} {'Retention':<20}")
    logger.info("-" * 80)

    logger.info(
        f"{'Size (MB)':<20} {fp32_size:<18.2f} {int8_size:<18.2f} "
        f"{int8_size / fp32_size * 100:<20.1f}%"
    )
    logger.info(
        f"{'Latency (ms)':<20} {fp32_results['latency']:<18.2f} {int8_results['latency']:<18.2f} "
        f"{int8_results['latency'] / fp32_results['latency'] * 100:<20.1f}%"
    )
    for metric in ["f1", "precision", "recall"]:
        fp32_val = fp32_results[metric]
        int8_val = int8_results[metric]
        retention = int8_val / fp32_val * 100 if fp32_val > 0 else 0
        logger.info(
            f"{'Span ' + metric.upper():<20} {fp32_val:<18.4f} {int8_val:<18.4f} "
            f"{retention:<20.1f}%"
        )
    logger.info("=" * 80 + "\n")


# ── Main Pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NerGuard ONNX Quantization Pipeline")
    parser.add_argument(
        "--strategy",
        choices=["dynamic-selective", "static", "static-mixed"],
        default="dynamic-selective",
        help="Quantization strategy",
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH,
        help="Path to FP32 PyTorch model",
    )
    parser.add_argument(
        "--data-path", default=DEFAULT_DATA_PATH,
        help="Path to tokenized dataset",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=256,
        help="Number of calibration samples (static strategies only)",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=SAMPLE_LIMIT,
        help="Evaluation sample limit",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip FP32 baseline evaluation",
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip ONNX export (reuse existing model.onnx)",
    )
    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}...")
    dataset = load_from_disk(args.data_path)["validation"]
    if args.sample_limit > 0:
        dataset = dataset.select(range(min(args.sample_limit, len(dataset))))
    logger.info(f"  Loaded {len(dataset)} samples for evaluation")

    # Load id2label
    id2label = load_id2label(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # --- FP32 Baseline ---
    fp32_size = get_file_size_mb(args.model_path)
    if not args.skip_baseline:
        logger.info("Evaluating FP32 baseline...")
        model_fp32 = AutoModelForTokenClassification.from_pretrained(args.model_path)
        fp32_results = evaluate_span_level(
            model_fp32, tokenizer, dataset, id2label, desc="Evaluating FP32"
        )
        del model_fp32
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info(f"  FP32 Span F1: {fp32_results['f1']:.4f}")
    else:
        fp32_results = None

    # --- ONNX Export ---
    if not args.skip_export:
        onnx_path = export_to_onnx(args.model_path, args.output_dir)
    else:
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            logger.error(f"model.onnx not found at {onnx_path}, cannot skip export")
            return
        logger.info(f"Reusing existing ONNX model at {onnx_path}")

    # --- Quantize ---
    if args.strategy == "dynamic-selective":
        quantized_path = quantize_dynamic_selective(args.output_dir)

    elif args.strategy == "static":
        quantized_path = quantize_static(
            args.output_dir, args.data_path,
            num_samples=args.calibration_samples,
        )

    elif args.strategy == "static-mixed":
        nodes_to_exclude = get_sensitive_nodes(onnx_path)
        quantized_path = quantize_static(
            args.output_dir, args.data_path,
            num_samples=args.calibration_samples,
            nodes_to_exclude=nodes_to_exclude,
        )

    # --- Evaluate INT8 ---
    logger.info("Loading quantized model...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = os.cpu_count()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    model_int8 = ORTModelForTokenClassification.from_pretrained(
        args.output_dir,
        file_name="model_quantized.onnx",
        provider="CPUExecutionProvider",
        session_options=session_options,
    )

    int8_results = evaluate_span_level(
        model_int8, tokenizer, dataset, id2label, desc="Evaluating INT8"
    )
    int8_size = os.path.getsize(quantized_path) / (1024 * 1024)

    logger.info(f"  INT8 Span F1: {int8_results['f1']:.4f}")

    # --- Comparison ---
    if fp32_results:
        print_comparison(fp32_results, int8_results, fp32_size, int8_size)

    logger.info("Quantization pipeline completed!")


if __name__ == "__main__":
    main()
