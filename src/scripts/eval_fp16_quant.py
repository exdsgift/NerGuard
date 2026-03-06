"""Evaluate FP16 (half-precision) and ONNX FP16 quantization for NerGuard."""

import os
import json
import time
import warnings

import torch
import numpy as np
import onnxruntime as ort
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer
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


def load_id2label():
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = json.load(f)
    return {int(k): v for k, v in config["id2label"].items()}


def evaluate(model, dataset, id2label, desc, is_ort=False):
    """Span-level seqeval evaluation."""
    latencies = []
    all_labels_seq = []
    all_preds_seq = []

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
        "f1": seqeval_f1(all_labels_seq, all_preds_seq, zero_division=0),
        "precision": seqeval_precision(all_labels_seq, all_preds_seq, zero_division=0),
        "recall": seqeval_recall(all_labels_seq, all_preds_seq, zero_division=0),
        "latency": np.mean(latencies),
    }


def main():
    id2label = load_id2label()
    dataset = load_from_disk(DATA_PATH)["validation"].select(range(1000))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    results = {}

    # --- Test 1: ONNX FP16 ---
    print("\n=== ONNX FP16 ===")
    fp16_dir = "./models/quantized_fp16"
    os.makedirs(fp16_dir, exist_ok=True)

    # Export to ONNX with FP16
    from onnxruntime.transformers import optimizer as ort_optimizer
    from optimum.exporters.onnx import main_export

    print("Exporting ONNX model...")
    main_export(
        MODEL_PATH,
        output=fp16_dir,
        task="token-classification",
        fp16=True,
        device="cpu",
    )

    # Check if FP16 export worked
    onnx_path = os.path.join(fp16_dir, "model.onnx")
    if os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ONNX FP16 size: {onnx_size:.2f} MB")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = os.cpu_count()
        ort_model = ORTModelForTokenClassification.from_pretrained(
            fp16_dir,
            provider="CPUExecutionProvider",
            session_options=session_options,
        )

        res = evaluate(ort_model, dataset, id2label, "Eval ONNX-FP16", is_ort=True)
        res["size"] = onnx_size
        results["ONNX-FP16"] = res
        print(f"  Span F1:   {res['f1']:.4f}")
        print(f"  Precision: {res['precision']:.4f}")
        print(f"  Recall:    {res['recall']:.4f}")
        print(f"  Latency:   {res['latency']:.2f} ms")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary of results")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}: F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"Lat={r['latency']:.1f}ms Size={r.get('size', 'N/A')}MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
