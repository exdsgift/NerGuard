"""
CoNLL-2003 Evaluation for NerGuard.

Evaluates the model on the standard CoNLL-2003 NER benchmark to enable
direct comparison with published results from LinkNER, GPT-NER, and
other NER systems.

NerGuard is a PII detection system, so only PER and LOC entities overlap
with CoNLL-2003's {PER, LOC, ORG, MISC} schema. ORG and MISC are mapped
to O (outside) since they are not PII types.

Metrics:
  - Token-level F1/P/R (sklearn) for internal consistency
  - Entity-level F1/P/R (seqeval) for comparison with published benchmarks
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report as sklearn_report
from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    CONLL_ID_TO_LABEL,
    CONLL_TO_UNIFIED,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NerGuard prediction -> unified label
NERGUARD_TO_UNIFIED = {
    "GIVENNAME": "PER",
    "SURNAME": "PER",
    "TITLE": "PER",
    "CITY": "LOC",
    "STREET": "LOC",
    "ZIPCODE": "LOC",
    "BUILDINGNUM": "LOC",
    "O": "O",
}


@dataclass
class CoNLLResult:
    """Results for CoNLL-2003 evaluation."""
    split: str
    n_samples: int
    # Token-level metrics
    token_f1_macro: float = 0.0
    token_f1_per_class: Dict[str, float] = field(default_factory=dict)
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_report: str = ""
    # Entity-level metrics (seqeval - standard for NER benchmarks)
    entity_f1_macro: float = 0.0
    entity_f1_micro: float = 0.0
    entity_f1_per_class: Dict[str, float] = field(default_factory=dict)
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_report: str = ""
    # Coverage analysis
    entity_counts: Dict[str, int] = field(default_factory=dict)
    covered_entities: int = 0
    total_entities: int = 0


class CoNLLEvaluator:
    """
    Evaluator for CoNLL-2003 standard NER benchmark.

    Enables direct comparison with LinkNER, GPT-NER, and other published
    NER systems by evaluating on the same dataset with entity-level F1.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_samples: Optional[int] = None,
        output_dir: str = "./plots/conll_results",
    ):
        self.model_path = model_path
        self.max_samples = max_samples
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"Model loaded with {len(self.id2label)} labels")

    def _load_conll(self, split: str = "test") -> Any:
        """Load CoNLL-2003 dataset."""
        logger.info(f"Loading CoNLL-2003 {split} split...")
        dataset = load_dataset("lhoestq/conll2003", split=split)
        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def _conll_id_to_bio(self, tag_id: int) -> str:
        """Convert CoNLL tag ID to BIO label string."""
        return CONLL_ID_TO_LABEL.get(tag_id, "O")

    def _conll_bio_to_unified_bio(self, conll_bio: str) -> str:
        """
        Convert CoNLL BIO label to unified BIO label for NerGuard comparison.

        Examples:
            B-PER -> B-PER (kept as is, overlaps with NerGuard PERSON)
            I-LOC -> I-LOC (kept as is, overlaps with NerGuard LOCATION)
            B-ORG -> O     (organizations not in PII schema)
            B-MISC -> O    (miscellaneous not in PII schema)
        """
        if conll_bio == "O":
            return "O"
        prefix, entity_type = conll_bio.split("-", 1)
        unified = CONLL_TO_UNIFIED.get(entity_type, "O")
        if unified == "O":
            return "O"
        return f"{prefix}-{unified}"

    def _nerguard_pred_to_unified_bio(self, pred_label: str) -> str:
        """
        Convert NerGuard prediction to unified BIO label.

        Examples:
            B-GIVENNAME -> B-PER
            I-SURNAME   -> I-PER
            B-CITY      -> B-LOC
            I-STREET    -> I-LOC
            B-EMAIL     -> O  (not a CoNLL entity type)
            O           -> O
        """
        if pred_label == "O":
            return "O"
        if "-" not in pred_label:
            return "O"
        prefix, entity_type = pred_label.split("-", 1)
        unified = NERGUARD_TO_UNIFIED.get(entity_type.upper(), None)
        if unified is None or unified == "O":
            return "O"
        return f"{prefix}-{unified}"

    def _predict_sample(self, tokens: List[str]) -> List[str]:
        """
        Predict NerGuard labels for a list of tokens, returning unified BIO labels.
        """
        text = " ".join(tokens)

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits[0], dim=-1).cpu().numpy()

        # Align predictions with original tokens
        pred_labels = []
        char_pos = 0

        for orig_token in tokens:
            found = False
            for i, (start, end) in enumerate(offset_mapping):
                if start <= char_pos < end:
                    pred_id = predictions[i]
                    pred_label = self.id2label[pred_id]
                    unified = self._nerguard_pred_to_unified_bio(pred_label)
                    pred_labels.append(unified)
                    found = True
                    break

            if not found:
                pred_labels.append("O")

            char_pos += len(orig_token) + 1  # +1 for space

        return pred_labels

    def evaluate(self, split: str = "test") -> CoNLLResult:
        """Run evaluation on CoNLL-2003."""
        dataset = self._load_conll(split)

        # Collect per-sentence BIO sequences for seqeval (entity-level)
        all_true_seqs: List[List[str]] = []
        all_pred_seqs: List[List[str]] = []

        # Flat lists for token-level metrics
        all_true_flat: List[str] = []
        all_pred_flat: List[str] = []

        # Entity counting
        entity_counts = defaultdict(int)

        for sample in tqdm(dataset, desc="CoNLL-2003 eval"):
            tokens = sample["tokens"]
            ner_tags = sample["ner_tags"]

            # Convert CoNLL tags to unified BIO
            true_bio = [
                self._conll_bio_to_unified_bio(self._conll_id_to_bio(tag))
                for tag in ner_tags
            ]

            # Get predictions
            pred_bio = self._predict_sample(tokens)

            # Align lengths
            min_len = min(len(true_bio), len(pred_bio))
            true_bio = true_bio[:min_len]
            pred_bio = pred_bio[:min_len]

            all_true_seqs.append(true_bio)
            all_pred_seqs.append(pred_bio)
            all_true_flat.extend(true_bio)
            all_pred_flat.extend(pred_bio)

            # Count entities in ground truth
            for tag in true_bio:
                if tag.startswith("B-"):
                    entity_type = tag.split("-", 1)[1]
                    entity_counts[entity_type] += 1

        # Count original CoNLL entities (including ORG/MISC mapped to O)
        total_original_entities = 0
        covered_entities = 0
        for sample in dataset:
            for tag_id in sample["ner_tags"]:
                bio = self._conll_id_to_bio(tag_id)
                if bio.startswith("B-"):
                    total_original_entities += 1
                    ent_type = bio.split("-", 1)[1]
                    if CONLL_TO_UNIFIED.get(ent_type, "O") != "O":
                        covered_entities += 1

        # Token-level metrics (sklearn)
        labels_token = sorted([l for l in set(all_true_flat) if l != "O"])
        token_report_str = ""
        token_f1_macro = 0.0
        token_precision = 0.0
        token_recall = 0.0
        token_f1_per_class = {}

        if labels_token:
            token_report_dict = sklearn_report(
                all_true_flat, all_pred_flat,
                labels=labels_token, output_dict=True, zero_division=0,
            )
            token_report_str = sklearn_report(
                all_true_flat, all_pred_flat,
                labels=labels_token, digits=4, zero_division=0,
            )
            token_f1_macro = token_report_dict["macro avg"]["f1-score"]
            token_precision = token_report_dict["macro avg"]["precision"]
            token_recall = token_report_dict["macro avg"]["recall"]
            for label in labels_token:
                if label in token_report_dict:
                    token_f1_per_class[label] = token_report_dict[label]["f1-score"]

        # Entity-level metrics (seqeval - standard for NER benchmarks)
        entity_report_str = seqeval_report(
            all_true_seqs, all_pred_seqs, digits=4, zero_division=0,
        )
        entity_f1_macro = seqeval_f1(
            all_true_seqs, all_pred_seqs, average="macro", zero_division=0,
        )
        entity_f1_micro = seqeval_f1(
            all_true_seqs, all_pred_seqs, average="micro", zero_division=0,
        )
        entity_precision = seqeval_precision(
            all_true_seqs, all_pred_seqs, average="macro", zero_division=0,
        )
        entity_recall = seqeval_recall(
            all_true_seqs, all_pred_seqs, average="macro", zero_division=0,
        )

        # Per-class entity-level F1
        entity_f1_per_class = {}
        for ent_type in ["PER", "LOC"]:
            try:
                ent_f1 = seqeval_f1(
                    all_true_seqs, all_pred_seqs,
                    average=None, zero_division=0,
                )
                # seqeval returns per-class in sorted order
                entity_report_dict = seqeval_report(
                    all_true_seqs, all_pred_seqs,
                    output_dict=True, zero_division=0,
                )
                if ent_type in entity_report_dict:
                    entity_f1_per_class[ent_type] = entity_report_dict[ent_type]["f1-score"]
            except Exception:
                pass

        return CoNLLResult(
            split=split,
            n_samples=len(dataset),
            token_f1_macro=token_f1_macro,
            token_f1_per_class=token_f1_per_class,
            token_precision=token_precision,
            token_recall=token_recall,
            token_report=token_report_str,
            entity_f1_macro=entity_f1_macro,
            entity_f1_micro=entity_f1_micro,
            entity_f1_per_class=entity_f1_per_class,
            entity_precision=entity_precision,
            entity_recall=entity_recall,
            entity_report=entity_report_str,
            entity_counts=dict(entity_counts),
            covered_entities=covered_entities,
            total_entities=total_original_entities,
        )

    def generate_report(self, result: CoNLLResult) -> str:
        """Generate comprehensive evaluation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("CoNLL-2003 EVALUATION REPORT")
        lines.append(f"Model: NerGuard (mDeBERTa-v3-base fine-tuned on AI4Privacy)")
        lines.append(f"Dataset: CoNLL-2003 ({result.split} split, N={result.n_samples})")
        lines.append(f"Covered entity types: PER, LOC (mapped from NerGuard PII labels)")
        lines.append(f"Excluded entity types: ORG, MISC (not in PII schema)")
        lines.append("=" * 80)
        lines.append("")

        # Coverage analysis
        pct = 100 * result.covered_entities / result.total_entities if result.total_entities > 0 else 0
        lines.append("COVERAGE ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Total CoNLL entities: {result.total_entities}")
        lines.append(f"Covered by NerGuard (PER+LOC): {result.covered_entities} ({pct:.1f}%)")
        lines.append(f"Not covered (ORG+MISC): {result.total_entities - result.covered_entities}")
        lines.append("")

        # Entity counts
        lines.append("ENTITY DISTRIBUTION (unified labels)")
        lines.append("-" * 40)
        for ent_type, count in sorted(result.entity_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {ent_type}: {count}")
        lines.append("")

        # Entity-level metrics (standard for NER benchmarks)
        lines.append("ENTITY-LEVEL METRICS (seqeval - standard for NER benchmarks)")
        lines.append("-" * 40)
        lines.append(f"F1 Macro:  {result.entity_f1_macro:.4f}")
        lines.append(f"F1 Micro:  {result.entity_f1_micro:.4f}")
        lines.append(f"Precision: {result.entity_precision:.4f}")
        lines.append(f"Recall:    {result.entity_recall:.4f}")
        lines.append("")
        lines.append("Per-class entity F1:")
        for cls, f1 in sorted(result.entity_f1_per_class.items()):
            lines.append(f"  {cls}: {f1:.4f}")
        lines.append("")
        lines.append("Full entity-level classification report:")
        lines.append(result.entity_report)
        lines.append("")

        # Token-level metrics
        lines.append("TOKEN-LEVEL METRICS (sklearn)")
        lines.append("-" * 40)
        lines.append(f"F1 Macro:  {result.token_f1_macro:.4f}")
        lines.append(f"Precision: {result.token_precision:.4f}")
        lines.append(f"Recall:    {result.token_recall:.4f}")
        lines.append("")
        lines.append("Full token-level classification report:")
        lines.append(result.token_report)
        lines.append("")

        # Comparison context
        lines.append("COMPARISON WITH PUBLISHED BENCHMARKS")
        lines.append("-" * 40)
        lines.append("Note: NerGuard is a PII detector, not a general NER system.")
        lines.append("Only PER and LOC entities are comparable. ORG and MISC are excluded.")
        lines.append("")
        lines.append("System                | Entity F1 (overall) | PER F1  | LOC F1  | Notes")
        lines.append("-" * 90)
        per_f1 = result.entity_f1_per_class.get("PER", 0)
        loc_f1 = result.entity_f1_per_class.get("LOC", 0)
        lines.append(
            f"NerGuard (this work)  | {result.entity_f1_micro:.4f}              "
            f"| {per_f1:.4f}  | {loc_f1:.4f}  | PER+LOC only"
        )
        lines.append(
            f"LinkNER (WWW 2024)    | ~0.93 (full CoNLL)       "
            f"| N/A     | N/A     | All 4 entity types"
        )
        lines.append(
            f"GPT-NER (NAACL 2025)  | 0.8315 (full CoNLL)      "
            f"| N/A     | N/A     | LLM-only, strict match"
        )
        lines.append("")

        report_text = "\n".join(lines)

        # Save report
        report_path = os.path.join(self.output_dir, "conll_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Report saved to {report_path}")

        # Save JSON results
        json_path = os.path.join(self.output_dir, "conll_results.json")
        json_data = {
            "dataset": "CoNLL-2003",
            "split": result.split,
            "n_samples": result.n_samples,
            "entity_level": {
                "f1_macro": result.entity_f1_macro,
                "f1_micro": result.entity_f1_micro,
                "precision": result.entity_precision,
                "recall": result.entity_recall,
                "per_class_f1": result.entity_f1_per_class,
            },
            "token_level": {
                "f1_macro": result.token_f1_macro,
                "precision": result.token_precision,
                "recall": result.token_recall,
                "per_class_f1": result.token_f1_per_class,
            },
            "coverage": {
                "total_entities": result.total_entities,
                "covered_entities": result.covered_entities,
                "entity_counts": result.entity_counts,
            },
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

        return report_text


def main():
    """Run CoNLL-2003 evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="CoNLL-2003 NerGuard evaluation")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples (default: full test set)",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots/conll_results",
        help="Output directory",
    )
    args = parser.parse_args()

    evaluator = CoNLLEvaluator(
        model_path=args.model_path,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )

    result = evaluator.evaluate(split="test")
    report = evaluator.generate_report(result)
    print(report)


if __name__ == "__main__":
    main()
