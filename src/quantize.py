import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import f1_score, precision_score, recall_score
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from tqdm import tqdm
import onnxruntime as ort
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelQuant")

DATA_PATH = "./data/processed/tokenized_data"
INPUT_MODEL = "./models/mdeberta-pii-safe/final"
OUTPUT_DIR = "./quantized_model"
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

# Plots

def set_publication_style():
    """Sets modern, publication-ready plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
        }
    )

def plot_unified_metrics(metrics, output_dir):
    """Generates unified bar chart with all metrics in one figure."""
    set_publication_style()

    models = list(metrics.keys())
    # Vibrant academic color palette
    colors = ["#b8b8b8", "#707070"]
    edge_colors = ["#0d0d0d", "#0d0d0d"]

    # Create larger figure with 2 rows and 3 columns
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(
        2, 3, hspace=0.28, wspace=0.3, left=0.08, right=0.96, top=0.89, bottom=0.08
    )

    # Add main title
    fig.suptitle(
        "Model Quantization: FP32 vs INT8 Comprehensive Analysis",
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )

    def add_value_labels(ax, rects, format_str):
        """Add value labels on top of bars."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                format_str.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=12,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#555555",
                    alpha=0.9,
                    linewidth=1.2,
                ),
            )

    # Row 1: Performance Metrics (F1, Precision, Recall)
    # F1 Score
    ax1 = fig.add_subplot(gs[0, 0])
    f1_scores = [metrics[m]["f1"] for m in models]
    bars1 = ax1.bar(
        models,
        f1_scores,
        color=colors,
        edgecolor=edge_colors,
        linewidth=2.5,
        width=0.6,
        alpha=0.9,
    )
    ax1.set_title("F1 Score", fontsize=18, pad=14)
    ax1.set_ylabel("Weighted F1-Score", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax1, bars1, "{:.4f}")

    # Precision
    ax2 = fig.add_subplot(gs[0, 1])
    precisions = [metrics[m]["precision"] for m in models]
    bars2 = ax2.bar(
        models,
        precisions,
        color=colors,
        edgecolor=edge_colors,
        linewidth=2.5,
        width=0.6,
        alpha=0.9,
    )
    ax2.set_title("Precision", fontsize=18, pad=14)
    ax2.set_ylabel("Weighted Precision", fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax2, bars2, "{:.4f}")

    # Recall
    ax3 = fig.add_subplot(gs[0, 2])
    recalls = [metrics[m]["recall"] for m in models]
    bars3 = ax3.bar(
        models,
        recalls,
        color=colors,
        edgecolor=edge_colors,
        linewidth=2.5,
        width=0.6,
        alpha=0.9,
    )
    ax3.set_title("Recall", fontsize=18, pad=14)
    ax3.set_ylabel("Weighted Recall", fontsize=14)
    ax3.set_ylim(0, 1.05)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax3, bars3, "{:.4f}")

    # Row 2: Efficiency Metrics (Latency, Size, Accuracy Change)
    # Latency
    ax4 = fig.add_subplot(gs[1, 0])
    latencies = [metrics[m]["latency"] for m in models]
    bars4 = ax4.bar(
        models,
        latencies,
        color=colors,
        edgecolor=edge_colors,
        linewidth=2.5,
        width=0.6,
        alpha=0.9,
    )
    ax4.set_title("Inference Latency", fontsize=18, pad=14)
    ax4.set_ylabel("Latency (ms)", fontsize=14)
    ax4.set_ylim(0, max(latencies) * 1.25)
    ax4.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax4, bars4, "{:.2f} ms")

    # Add speedup annotation (plain text, closer to the graph)
    speedup = latencies[0] / latencies[1] if latencies[1] > 0 else 1
    speedup_text = "faster" if latencies[1] < latencies[0] else "slower"
    ax4.text(
        0.5,
        -0.14,
        f"INT8 is {speedup:.2f}× {speedup_text}",
        transform=ax4.transAxes,
        ha="center",
        fontsize=12,
        style="italic",
        color="#333333",
    )

    # Model Size
    ax5 = fig.add_subplot(gs[1, 1])
    sizes = [metrics[m]["size"] for m in models]
    bars5 = ax5.bar(
        models,
        sizes,
        color=colors,
        edgecolor=edge_colors,
        linewidth=2.5,
        width=0.6,
        alpha=0.9,
    )
    ax5.set_title("Model Size", fontsize=18, pad=14)
    ax5.set_ylabel("Size (MB)", fontsize=14)
    ax5.set_ylim(0, max(sizes) * 1.25)
    ax5.grid(True, alpha=0.35, axis="y", linewidth=0.8)
    add_value_labels(ax5, bars5, "{:.1f} MB")

    # Add compression annotation (plain text, closer to the graph)
    reduction = (1 - sizes[1] / sizes[0]) * 100
    compression_ratio = sizes[0] / sizes[1] if sizes[1] > 0 else 1
    ax5.text(
        0.5,
        -0.14,
        f"{compression_ratio:.2f}× compression ({reduction:.1f}% reduction)",
        transform=ax5.transAxes,
        ha="center",
        fontsize=12,
        style="italic",
        color="#333333",
    )

    # Accuracy Difference (absolute changes)
    ax6 = fig.add_subplot(gs[1, 2])

    # Calculate absolute differences
    f1_diff = metrics[models[1]]["f1"] - metrics[models[0]]["f1"]
    precision_diff = metrics[models[1]]["precision"] - metrics[models[0]]["precision"]
    recall_diff = metrics[models[1]]["recall"] - metrics[models[0]]["recall"]

    metric_names = ["F1", "Precision", "Recall"]
    diffs = [f1_diff, precision_diff, recall_diff]

    # Color based on positive or negative change
    diff_colors = [colors[0] if d >= 0 else "#C85450" for d in diffs]
    diff_edges = [edge_colors[0] if d >= 0 else "#A04340" for d in diffs]

    bars6 = ax6.bar(
        metric_names,
        diffs,
        color=diff_colors,
        edgecolor=diff_edges,
        linewidth=2.5,
        width=0.6,
        alpha=0.9,
    )
    ax6.set_title("Accuracy Change (INT8 - FP32)", fontsize=18, pad=14)
    ax6.set_ylabel("Absolute Difference", fontsize=14)

    # Set y-axis limits dynamically
    max_abs_diff = max(abs(min(diffs)), abs(max(diffs)))
    y_margin = max_abs_diff * 0.3
    ax6.set_ylim(-max_abs_diff - y_margin, max_abs_diff + y_margin)

    ax6.axhline(y=0, color="#333333", linestyle="-", linewidth=1.5, alpha=0.8)
    ax6.grid(True, alpha=0.35, axis="y", linewidth=0.8)

    # Add value labels with sign
    for bar, val in zip(bars6, diffs):
        height = bar.get_height()
        y_offset = 6 if height >= 0 else -16
        va = "bottom" if height >= 0 else "top"
        ax6.annotate(
            f"{val:+.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontweight="bold",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#555555",
                alpha=0.9,
                linewidth=1.2,
            ),
        )

    # Save figure
    save_path = os.path.join(output_dir, "unified_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"  Unified metrics plot saved to {save_path}")
    plt.close()

def plot_radar_comparison(metrics, output_dir):
    """Generates radar chart for overall comparison."""
    set_publication_style()

    models = list(metrics.keys())
    colors = ["#3498db", "#e74c3c"]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    categories = ["Latency\n(inv)", "Size\n(inv)", "F1", "Precision", "Recall"]
    N = len(categories)

    latencies = [metrics[m]["latency"] for m in models]
    sizes = [metrics[m]["size"] for m in models]
    f1_scores = [metrics[m]["f1"] for m in models]
    precisions = [metrics[m]["precision"] for m in models]
    recalls = [metrics[m]["recall"] for m in models]

    # Normalize metrics (invert latency and size for radar)
    values_fp32 = [
        1 / latencies[0] * 100,
        1 / sizes[0] * 10,
        f1_scores[0],
        precisions[0],
        recalls[0],
    ]
    values_int8 = [
        1 / latencies[1] * 100,
        1 / sizes[1] * 10,
        f1_scores[1],
        precisions[1],
        recalls[1],
    ]

    # Normalize to 0-1
    max_vals = [max(values_fp32[i], values_int8[i]) for i in range(N)]
    values_fp32_norm = [
        values_fp32[i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(N)
    ]
    values_int8_norm = [
        values_int8[i] / max_vals[i] if max_vals[i] > 0 else 0 for i in range(N)
    ]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_fp32_norm += values_fp32_norm[:1]
    values_int8_norm += values_int8_norm[:1]
    angles += angles[:1]

    ax.plot(
        angles,
        values_fp32_norm,
        "o-",
        linewidth=2.5,
        label="FP32",
        color=colors[0],
        markersize=7,
    )
    ax.fill(angles, values_fp32_norm, alpha=0.25, color=colors[0])
    ax.plot(
        angles,
        values_int8_norm,
        "o-",
        linewidth=2.5,
        label="INT8",
        color=colors[1],
        markersize=7,
    )
    ax.fill(angles, values_int8_norm, alpha=0.25, color=colors[1])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.set_title(
        "Overall Comparison: FP32 vs INT8",
        fontweight="bold",
        pad=20,
        fontsize=16,
        y=1.08,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "overall_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info(f"  Overall comparison plot saved to {save_path}")
    plt.close()

def save_detailed_report(metrics, output_dir):
    """Saves a simple statistical report with calculated metrics."""
    report_path = os.path.join(output_dir, "report.txt")

    models = list(metrics.keys())
    fp32_metrics = metrics[models[0]]
    int8_metrics = metrics[models[1]]

    # Calculate all statistics
    latency_ratio = fp32_metrics["latency"] / int8_metrics["latency"]
    size_reduction = (1 - int8_metrics["size"] / fp32_metrics["size"]) * 100
    f1_diff = int8_metrics["f1"] - fp32_metrics["f1"]
    f1_diff_pct = (f1_diff / fp32_metrics["f1"]) * 100
    precision_diff = int8_metrics["precision"] - fp32_metrics["precision"]
    precision_diff_pct = (precision_diff / fp32_metrics["precision"]) * 100
    recall_diff = int8_metrics["recall"] - fp32_metrics["recall"]
    recall_diff_pct = (recall_diff / fp32_metrics["recall"]) * 100
    compression_ratio = fp32_metrics["size"] / int8_metrics["size"]
    accuracy_retention = (int8_metrics["f1"] / fp32_metrics["f1"]) * 100

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("QUANTIZATION REPORT\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("CALCULATED STATISTICS:\n")
        f.write(f"Latency Ratio (FP32/INT8): {latency_ratio:.3f}x\n")
        f.write(f"Size Reduction: {size_reduction:.2f}%\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}:1\n")
        f.write(f"F1 Difference: {f1_diff:+.6f} ({f1_diff_pct:+.2f}%)\n")
        f.write(
            f"Precision Difference: {precision_diff:+.6f} ({precision_diff_pct:+.2f}%)\n"
        )
        f.write(f"Recall Difference: {recall_diff:+.6f} ({recall_diff_pct:+.2f}%)\n")
        f.write(f"Accuracy Retention: {accuracy_retention:.2f}%\n")

    logger.info(f"  Report saved to {report_path}")
    return report_path

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

    plot_unified_metrics(metrics, output_dir)
    plot_radar_comparison(metrics, output_dir)
    save_detailed_report(metrics, output_dir)
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
