"""
Test weight-only INT8 quantization via onnxruntime.quantization directly.

Uses MatMulConstBOnly=True to only quantize weight matrices in MatMul ops,
keeping activations in FP32. Also tests INT4 weight-only quantization.
"""

import os
import json
import time
import warnings

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from datasets import load_from_disk
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/ai4privacy/open-pii-masking-500k-ai4privacy/processed/tokenized_data"
ONNX_DIR = "./models/quantized_exp2"  # reuse existing ONNX export


def load_id2label():
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = json.load(f)
    return {int(k): v for k, v in config["id2label"].items()}


def evaluate_onnx(model_path, dataset, id2label, tokenizer, desc):
    """Evaluate ONNX model with span-level seqeval."""
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = os.cpu_count()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    latencies = []
    all_labels_seq = []
    all_preds_seq = []

    # Warmup — only pass expected inputs
    input_names = {inp.name for inp in session.get_inputs()}
    dummy = tokenizer("warmup text", return_tensors="np", padding=True, truncation=True)
    dummy = {k: v.astype(np.int64) for k, v in dummy.items() if k in input_names}
    for _ in range(5):
        session.run(None, dummy)

    for sample in tqdm(dataset, desc=desc):
        labels = sample["labels"]
        inputs = {
            "input_ids": np.array([sample["input_ids"]], dtype=np.int64),
            "attention_mask": np.array([sample["attention_mask"]], dtype=np.int64),
        }

        start = time.perf_counter()
        outputs = session.run(None, inputs)
        end = time.perf_counter()

        preds = np.argmax(outputs[0], axis=-1)[0]
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
        "f1": seqeval_f1(all_labels_seq, all_preds_seq, zero_division=0),
        "precision": seqeval_precision(all_labels_seq, all_preds_seq, zero_division=0),
        "recall": seqeval_recall(all_labels_seq, all_preds_seq, zero_division=0),
        "latency": np.mean(latencies),
    }


def main():
    id2label = load_id2label()
    dataset = load_from_disk(DATA_PATH)["validation"].select(range(1000))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    onnx_input = os.path.join(ONNX_DIR, "model.onnx")
    if not os.path.exists(onnx_input):
        print(f"ERROR: {onnx_input} not found. Run quantizer.py first to export ONNX.")
        return

    results = {}

    # --- Test 1: Weight-only INT8 with reduce_range ---
    print("\n=== Weight-Only INT8 (MatMulConstBOnly + reduce_range) ===")
    out_path = "./models/quantized_weight_only/model_wo_int8.onnx"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    quantize_dynamic(
        model_input=onnx_input,
        model_output=out_path,
        per_channel=True,
        reduce_range=True,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul"],
        extra_options={"MatMulConstBOnly": True},
    )

    size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Model size: {size:.2f} MB")

    res = evaluate_onnx(out_path, dataset, id2label, tokenizer, "Eval WO-INT8")
    res["size"] = size
    results["WO-INT8"] = res
    print(f"  Span F1:   {res['f1']:.4f}")
    print(f"  Precision: {res['precision']:.4f}")
    print(f"  Recall:    {res['recall']:.4f}")
    print(f"  Latency:   {res['latency']:.2f} ms")

    # --- Test 2: Weight-only INT8 without reduce_range ---
    print("\n=== Weight-Only INT8 (no reduce_range) ===")
    out_path2 = "./models/quantized_weight_only/model_wo_int8_norr.onnx"

    quantize_dynamic(
        model_input=onnx_input,
        model_output=out_path2,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul"],
        extra_options={"MatMulConstBOnly": True},
    )

    size2 = os.path.getsize(out_path2) / (1024 * 1024)
    print(f"  Model size: {size2:.2f} MB")

    res2 = evaluate_onnx(out_path2, dataset, id2label, tokenizer, "Eval WO-INT8-noRR")
    res2["size"] = size2
    results["WO-INT8-noRR"] = res2
    print(f"  Span F1:   {res2['f1']:.4f}")
    print(f"  Precision: {res2['precision']:.4f}")
    print(f"  Recall:    {res2['recall']:.4f}")
    print(f"  Latency:   {res2['latency']:.2f} ms")

    # --- Summary ---
    fp32_size = os.path.getsize(onnx_input) / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"{'Method':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Lat(ms)':>10} {'Size(MB)':>10} {'Retain%':>10}")
    print(f"{'-'*70}")
    print(f"{'FP32 (ref)':<25} {'0.9457':>8} {'0.9431':>8} {'0.9484':>8} {'283':>10} {fp32_size:>10.1f} {'100.0':>10}")
    for name, r in results.items():
        retention = r['f1'] / 0.9457 * 100
        print(f"{name:<25} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['latency']:>10.1f} {r['size']:>10.1f} {retention:>10.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
