import os
import sys
import logging
import json
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from torch.utils.data import DataLoader

from src.core.constants import DEFAULT_MODEL_PATH, NVIDIA_TO_MODEL_MAP
from src.core.label_mapper import LabelMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log", mode="w"),
    ],
)
logger = logging.getLogger("NvidiaEval")

MODEL_PATH = DEFAULT_MODEL_PATH
OUTPUT_DIR = "./plots/evaluation_nvidia"
BATCH_SIZE = 16
MAX_SAMPLES = 1000
CONTEXT_LENGTH = 512


@dataclass
class EvalConfig:
    model_path: str
    output_dir: str
    max_samples: Optional[int]
    context_length: int
    batch_size: int


class NvidiaEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self._setup_env()

        logger.info(f"Loading model resources from: {config.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.model_path
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

        self.mapper = LabelMapper(self.model.config.id2label)
        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_env(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        # Suppress extensive huggingface logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

    def load_data(self):
        logger.info("Loading NVIDIA/Nemotron-PII dataset...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train")
        if self.config.max_samples:
            logger.info(f"Subsampling dataset to {self.config.max_samples} samples.")
            ds = ds.select(range(min(len(ds), self.config.max_samples)))
        return ds

    def process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Aligns text spans to tokens. Intended for dataset.map()."""
        text = example["text"]
        spans = example["spans"]

        # Parse spans safely
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except json.JSONDecodeError:
                try:
                    spans = ast.literal_eval(spans)
                except (ValueError, SyntaxError):
                    spans = []

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.context_length,
            return_offsets_mapping=True,
            is_split_into_words=False,
        )

        offset_mapping = tokenized["offset_mapping"]
        input_ids = tokenized["input_ids"]
        labels = [self.mapper.model_label2id.get("O", 0)] * len(input_ids)

        # Filter valid spans
        valid_spans = []
        if isinstance(spans, list):
            for s in spans:
                if isinstance(s, dict) and "label" in s and "start" in s and "end" in s:
                    valid_spans.append(s)
        valid_spans.sort(key=lambda x: x["start"])

        # Span alignment logic
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end:  # Special tokens
                labels[idx] = -100
                continue

            for span in valid_spans:
                s_start, s_end, s_label = span["start"], span["end"], span["label"]

                if token_end <= s_start or token_start >= s_end:
                    continue

                is_start = token_start == s_start

                # Heuristic: If we are inside a span but not at the exact start,
                # check if the previous token had a different entity type to force B- tag if needed.
                if not is_start and idx > 0:
                    prev_label_id = labels[idx - 1]
                    # -100 check ensures we don't look at special tokens
                    if prev_label_id != -100:
                        prev_str = self.mapper.model_id2label.get(prev_label_id, "O")
                        target_base = NVIDIA_TO_MODEL_MAP.get(s_label, "O")
                        if target_base != "O" and target_base not in prev_str:
                            is_start = True

                labels[idx] = self.mapper.get_token_label_id(s_label, is_start)
                break

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    def evaluate(self):
        raw_dataset = self.load_data()

        logger.info("Aligning labels (Pre-processing)...")
        # Map dataset to add labels column
        processed_dataset = raw_dataset.map(
            self.process_example,
            remove_columns=raw_dataset.column_names,
            desc="Aligning labels",
        )

        # Use DataCollator for dynamic padding (more efficient than max_length padding everywhere)
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        dataloader = DataLoader(
            processed_dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )

        true_labels = []
        pred_labels = []

        logger.info(f"Starting inference on {len(processed_dataset)} samples...")

        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            # labels are needed here only for masking purposes later, but batch contains them
            batch_labels = batch["labels"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=-1)

            # Masking special tokens (-100)
            active_mask = batch_labels != -100

            # Flatten and filter
            active_labels = batch_labels[active_mask].cpu().numpy()
            active_preds = preds[active_mask].cpu().numpy()

            true_labels.extend(active_labels)
            pred_labels.extend(active_preds)

        self._generate_report(true_labels, pred_labels)

    def _generate_report(self, true_ids: List[int], pred_ids: List[int]):
        logger.info("Generating performance report...")

        id2label = self.model.config.id2label
        y_true = [id2label[i] for i in true_ids]
        y_pred = [id2label[i] for i in pred_ids]

        # Identify labels excluding 'O'
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        labels_no_o = [l for l in unique_labels if l != "O"]

        # 1. Classification Report (Per-class)
        cls_report = classification_report(
            y_true, y_pred, labels=labels_no_o, digits=4, zero_division=0
        )
        logger.info("\n" + cls_report)

        # 2. Global Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        global_metrics = {
            "global_accuracy": accuracy,
            "macro_precision": precision_m,
            "macro_recall": recall_m,
            "macro_f1": f1_m,
            "weighted_precision": precision_w,
            "weighted_recall": recall_w,
            "weighted_f1": f1_w,
        }

        logger.info(f"Global Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1_m:.4f} | Weighted F1: {f1_w:.4f}")

        self._save_results(cls_report, global_metrics)

        logger.info("--- Mapping Coverage Stats ---")
        sorted_stats = sorted(
            self.mapper.map_counter.items(), key=lambda x: x[1], reverse=True
        )
        for k, v in sorted_stats:
            if v > 0:
                target = NVIDIA_TO_MODEL_MAP.get(k, "O")
                logger.info(f"{k:<25} -> {target:<15}: {v}")

        self._plot_cm(y_true, y_pred)

    def _save_results(self, cls_report: str, metrics: Dict[str, float]):
        report_path = os.path.join(self.config.output_dir, "evaluation_report.txt")
        json_path = os.path.join(self.config.output_dir, "metrics.json")

        try:
            with open(report_path, "w") as f:
                f.write("--- Classification Report ---\n")
                f.write(cls_report)
                f.write("\n\n--- Global Metrics ---\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.4f}\n")

            with open(json_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Reports saved to {self.config.output_dir}")
        except IOError as e:
            logger.error(f"Failed to save reports: {e}")

    def _plot_cm(self, y_true: List[str], y_pred: List[str]):
        from collections import Counter

        # Filter sparse classes for cleaner visualization
        counts = Counter(y_true)
        relevant_labels = [k for k, v in counts.items() if v > 10 and k != "O"]
        relevant_labels = sorted(list(set(relevant_labels)))

        if not relevant_labels:
            logger.warning("Not enough data to plot a meaningful confusion matrix.")
            return

        cm = confusion_matrix(y_true, y_pred, labels=relevant_labels)

        # Normalize
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            xticklabels=relevant_labels,
            yticklabels=relevant_labels,
            cmap="Blues",
        )
        plt.title("Confusion Matrix (Normalized)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        save_path = os.path.join(self.config.output_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    config = EvalConfig(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        max_samples=MAX_SAMPLES,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
    )

    evaluator = NvidiaEvaluator(config)
    evaluator.evaluate()
