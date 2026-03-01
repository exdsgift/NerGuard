import os
import time
import logging
import warnings

import torch
import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from tqdm import tqdm

from src.core.constants import DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelQuant")

DATA_PATH = DEFAULT_DATA_PATH
INPUT_MODEL = DEFAULT_MODEL_PATH
OUTPUT_DIR = "./plots/quantization_plots"
SAMPLE_LIMIT = 1000


def get_dir_size(path):
    """Calculates total size of a directory in MB."""
    total_size = 0
    if os.path.isfile(path):
        total_size = os.path.getsize(path)
    else:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

# Main functions

def evaluate_model_performance(
    model, tokenizer, dataset, device="cpu", desc="Evaluating"
):
    """Computes Latency, F1-Score, Precision, and Recall."""
    model.to(device) if hasattr(model, "to") else None
    latencies = []
    all_preds = []
    all_labels = []

    is_ort = isinstance(model, ORTModelForTokenClassification)

    # Warmup
    dummy_text = "Warmup routine for stable latency measurement."
    if is_ort:
        dummy_inputs = tokenizer(
            dummy_text, return_tensors="np", padding=True, truncation=True
        )
        dummy_inputs = {k: v.astype(np.int64) for k, v in dummy_inputs.items()}
        for _ in range(10):
            _ = model(**dummy_inputs)
    else:
        dummy_inputs = tokenizer(
            dummy_text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(**dummy_inputs)

    logger.info(f"--- {desc} ---")

    for sample in tqdm(dataset, desc=desc):
        labels = sample["labels"]

        if is_ort:
            inputs = {
                "input_ids": np.array([sample["input_ids"]], dtype=np.int64),
                "attention_mask": np.array([sample["attention_mask"]], dtype=np.int64),
            }
            start_time = time.perf_counter()
            outputs = model(**inputs)
            end_time = time.perf_counter()
            predictions = np.argmax(outputs.logits, axis=-1)[0]
        else:
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

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "latency": avg_latency,
        "latency_std": std_latency,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def quantize_and_evaluate(model_path, data_path, output_dir):
    """Pipeline: Export -> Optimize -> Quantize -> Evaluate"""

    logger.info(f"Loading Dataset from {data_path}...")
    try:
        dataset = load_from_disk(data_path)["validation"]
        if SAMPLE_LIMIT:
            dataset = dataset.select(range(min(SAMPLE_LIMIT, len(dataset))))
        logger.info(f"  Loaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f" Error loading dataset: {e}")
        return

    # --- 1. Baseline Evaluation ---
    logger.info("Loading Original Model (FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_fp32 = AutoModelForTokenClassification.from_pretrained(model_path)

    results_fp32 = evaluate_model_performance(
        model_fp32, tokenizer, dataset, device="cpu", desc="Evaluating FP32"
    )
    size_fp32 = get_dir_size(model_path)

    del model_fp32
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- 2. ONNX Export & Optimization ---
    logger.info("Starting ONNX pipeline...")
    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    logger.info("Exporting to ONNX...")
    try:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = os.cpu_count()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        ort_model = ORTModelForTokenClassification.from_pretrained(
            model_path,
            export=True,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )
        onnx_path = os.path.join(output_dir, "model.onnx")
        ort_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"  ONNX model exported to {onnx_path}")
    except Exception as e:
        logger.error(f" ONNX export failed: {e}")
        return

    logger.info("Skipping ONNX graph optimization (can cause hangs with DeBERTa)...")
    optimized_model_path = onnx_path
    logger.info(f"Using base ONNX model for quantization: {optimized_model_path}")

    # --- 3. Dynamic Quantization ---
    logger.info("Quantizing model to INT8...")
    try:
        logger.info("Creating quantization config...")
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

        model_basename = os.path.basename(optimized_model_path)
        logger.info(f"Loading quantizer for {model_basename}...")
        quantizer = ORTQuantizer.from_pretrained(output_dir, file_name=model_basename)
        logger.info("Starting quantization process...")
        quantizer.quantize(
            save_dir=output_dir, quantization_config=qconfig, file_suffix="quantized"
        )

        quantized_name = model_basename.replace(".onnx", "_quantized.onnx")
        quantized_path = os.path.join(output_dir, quantized_name)
        logger.info(f"  Quantized model: {quantized_path}")

    except Exception as e:
        logger.error(f" Quantization failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return

    # --- 4. Quantized Evaluation ---
    if not os.path.exists(quantized_path):
        logger.error(f" Quantized model not found at {quantized_path}")
        return

    logger.info("Loading quantized model...")
    try:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = os.cpu_count()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        model_int8 = ORTModelForTokenClassification.from_pretrained(
            output_dir,
            file_name=quantized_name,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )
        logger.info("  Quantized model loaded successfully")

        results_int8 = evaluate_model_performance(
            model_int8, tokenizer, dataset, device="cpu", desc="Evaluating INT8"
        )
        size_int8 = os.path.getsize(quantized_path) / (1024 * 1024)
        logger.info("  Quantized model evaluation complete")

    except Exception as e:
        logger.error(f"Error loading/evaluating quantized model: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return

    # --- 5. Results & Visualization ---
    metrics = {
        "Original (FP32)": {**results_fp32, "size": size_fp32},
        "Quantized (INT8)": {**results_int8, "size": size_int8},
    }

    logger.info("\n" + "=" * 80)
    logger.info(f"{'Metric':<20} {'FP32':<18} {'INT8':<18} {'Improvement':<20}")
    logger.info("-" * 80)
    logger.info(
        f"{'Latency (ms)':<20} {results_fp32['latency']:<18.2f} {results_int8['latency']:<18.2f} {results_fp32['latency'] / results_int8['latency']:<20.2f}× faster"
    )
    logger.info(
        f"{'Size (MB)':<20} {size_fp32:<18.2f} {size_int8:<18.2f} {(1 - size_int8 / size_fp32) * 100:<20.1f}% smaller"
    )
    logger.info(
        f"{'F1 Score':<20} {results_fp32['f1']:<18.4f} {results_int8['f1']:<18.4f} {results_int8['f1'] - results_fp32['f1']:<+20.4f}"
    )
    logger.info(
        f"{'Precision':<20} {results_fp32['precision']:<18.4f} {results_int8['precision']:<18.4f} {results_int8['precision'] - results_fp32['precision']:<+20.4f}"
    )
    logger.info(
        f"{'Recall':<20} {results_fp32['recall']:<18.4f} {results_int8['recall']:<18.4f} {results_int8['recall'] - results_fp32['recall']:<+20.4f}"
    )
    logger.info("=" * 80 + "\n")

    logger.info("Quantization pipeline completed successfully!")


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        logger.error(f"DATA_PATH {DATA_PATH} not found.")
    else:
        try:
            quantize_and_evaluate(INPUT_MODEL, DATA_PATH, OUTPUT_DIR)
        except Exception as e:
            logger.error(f"Fatal error in main pipeline: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
