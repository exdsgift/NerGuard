"""Quick evaluation of PyTorch native dynamic quantization."""

import os
import json
import time
import warnings

import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_PATH = "./models/mdeberta-pii-safe/final"
DATA_PATH = "./data/ai4privacy/open-pii-masking-500k-ai4privacy/processed/tokenized_data"

# Load id2label
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
id2label = {int(k): v for k, v in config["id2label"].items()}

# Load dataset
dataset = load_from_disk(DATA_PATH)["validation"].select(range(1000))
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load and quantize model
print("Loading FP32 model...")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
print("Applying PyTorch dynamic quantization (Linear layers)...")
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized.eval()

# Evaluate
print("Evaluating span-level metrics (1000 samples)...")
latencies = []
all_labels_seq = []
all_preds_seq = []

for sample in tqdm(dataset, desc="Eval INT8-PyTorch"):
    labels = sample["labels"]
    input_ids = torch.tensor([sample["input_ids"]])
    attention_mask = torch.tensor([sample["attention_mask"]])

    start = time.perf_counter()
    with torch.no_grad():
        outputs = quantized(input_ids, attention_mask=attention_mask)
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

f1 = seqeval_f1(all_labels_seq, all_preds_seq, zero_division=0)
prec = seqeval_precision(all_labels_seq, all_preds_seq, zero_division=0)
rec = seqeval_recall(all_labels_seq, all_preds_seq, zero_division=0)
lat = np.mean(latencies)

# Save quantized model
save_dir = "./models/quantized_pytorch"
os.makedirs(save_dir, exist_ok=True)
torch.save(quantized.state_dict(), os.path.join(save_dir, "model_int8.pt"))
model_size = os.path.getsize(os.path.join(save_dir, "model_int8.pt")) / (1024 * 1024)

print(f"\n{'='*60}")
print(f"PyTorch Dynamic INT8 (Linear layers only)")
print(f"{'='*60}")
print(f"  Model Size:  {model_size:.2f} MB")
print(f"  Span F1:     {f1:.4f}")
print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print(f"  Latency:     {lat:.2f} ms")
print(f"{'='*60}")
