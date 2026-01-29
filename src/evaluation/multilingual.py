"""
Multilingual Evaluation for NerGuard.

Evaluates the model on WikiNeural dataset across multiple European languages
to assess cross-lingual transfer capabilities.

This addresses the thesis extension requirement for multilingual evaluation,
testing on English, Italian, Spanish, German, French, Portuguese, Dutch, and Polish.

Dataset: Babelscape/wikineural
Labels: PER (1-2), ORG (3-4), LOC (5-6), MISC (7-8), O (0)
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.core.constants import DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# WikiNeural tag ID to label string mapping
# WikiNeural uses: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
WIKINEURAL_ID_TO_LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}

# WikiNeural to NerGuard simplified label mapping
# We map to unified categories for comparison
WIKINEURAL_TO_UNIFIED = {
    "O": "O",
    "B-PER": "PERSON",
    "I-PER": "PERSON",
    "B-LOC": "LOCATION",
    "I-LOC": "LOCATION",
    "B-ORG": "O",  # Organizations not in our PII schema
    "I-ORG": "O",
    "B-MISC": "O",  # Misc not in our PII schema
    "I-MISC": "O",
}

# Languages to evaluate (European languages with different characteristics)
LANGUAGES = {
    "en": {"name": "English", "difficulty": "baseline", "family": "Germanic"},
    "it": {"name": "Italian", "difficulty": "medium", "family": "Romance"},
    "es": {"name": "Spanish", "difficulty": "medium", "family": "Romance"},
    "de": {"name": "German", "difficulty": "high", "family": "Germanic"},
    "fr": {"name": "French", "difficulty": "medium", "family": "Romance"},
    "pt": {"name": "Portuguese", "difficulty": "medium", "family": "Romance"},
    "nl": {"name": "Dutch", "difficulty": "high", "family": "Germanic"},
    "pl": {"name": "Polish", "difficulty": "high", "family": "Slavic"},
}


@dataclass
class LanguageResult:
    """Results for a single language evaluation."""
    language: str
    language_name: str
    family: str
    n_samples: int
    f1_macro: float
    f1_weighted: float
    precision: float
    recall: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)


@dataclass
class MultilingualResult:
    """Aggregated results across all languages."""
    model_name: str
    language_results: List[LanguageResult] = field(default_factory=list)

    def get_summary_df(self) -> Dict[str, List]:
        """Get summary as dictionary for DataFrame creation."""
        return {
            "Language": [r.language for r in self.language_results],
            "Name": [r.language_name for r in self.language_results],
            "Family": [r.family for r in self.language_results],
            "Samples": [r.n_samples for r in self.language_results],
            "F1 Macro": [r.f1_macro for r in self.language_results],
            "F1 Weighted": [r.f1_weighted for r in self.language_results],
            "Precision": [r.precision for r in self.language_results],
            "Recall": [r.recall for r in self.language_results],
        }


class MultilingualEvaluator:
    """
    Evaluator for cross-lingual NER transfer assessment.

    Tests the mDeBERTa-based NerGuard model on WikiNeural dataset
    across multiple European languages.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        languages: Optional[List[str]] = None,
        max_samples_per_lang: int = 1000,
        output_dir: str = "./plots/multilingual_results",
    ):
        """
        Initialize the multilingual evaluator.

        Args:
            model_path: Path to the trained model
            languages: List of language codes to evaluate (default: all)
            max_samples_per_lang: Maximum samples per language
            output_dir: Directory for output files
        """
        self.model_path = model_path
        self.languages = languages or list(LANGUAGES.keys())
        self.max_samples = max_samples_per_lang
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

    def _load_wikineural(self, lang: str) -> Any:
        """Load WikiNeural dataset for a specific language."""
        logger.info(f"Loading WikiNeural for {lang}...")
        try:
            # WikiNeural uses language-specific split names like test_en, test_it, etc.
            split_name = f"test_{lang}"
            dataset = load_dataset("Babelscape/wikineural", split=split_name)

            if len(dataset) > self.max_samples:
                indices = np.random.choice(len(dataset), self.max_samples, replace=False)
                dataset = dataset.select(indices)
            logger.info(f"  Loaded {len(dataset)} samples for {lang}")
            return dataset
        except Exception as e:
            logger.error(f"  Failed to load {lang}: {e}")
            return None

    def _wikineural_id_to_label(self, tag_id: int) -> str:
        """Convert WikiNeural tag ID to label string."""
        return WIKINEURAL_ID_TO_LABEL.get(tag_id, "O")

    def _predict_sample(self, tokens: List[str]) -> List[str]:
        """Predict labels for a list of tokens."""
        # Join tokens and tokenize
        text = " ".join(tokens)

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].numpy()

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        # Align predictions with original tokens
        pred_labels = []
        char_pos = 0

        for orig_token in tokens:
            # Find the subword token that covers this original token
            found = False
            for i, (start, end) in enumerate(offset_mapping):
                if start <= char_pos < end:
                    pred_id = predictions[i]
                    pred_label = self.id2label[pred_id]
                    # Map to unified category
                    clean_label = pred_label.replace("B-", "").replace("I-", "").upper()
                    if clean_label in ["GIVENNAME", "SURNAME", "TITLE"]:
                        pred_labels.append("PERSON")
                    elif clean_label in ["CITY", "STREET", "ZIPCODE", "BUILDINGNUM"]:
                        pred_labels.append("LOCATION")
                    elif clean_label == "O":
                        pred_labels.append("O")
                    else:
                        pred_labels.append("O")  # Other PII types not in WikiNeural
                    found = True
                    break

            if not found:
                pred_labels.append("O")

            char_pos += len(orig_token) + 1  # +1 for space

        return pred_labels

    def evaluate_language(self, lang: str) -> Optional[LanguageResult]:
        """Evaluate on a single language."""
        lang_info = LANGUAGES.get(lang, {"name": lang, "difficulty": "unknown", "family": "unknown"})
        logger.info(f"Evaluating {lang_info['name']} ({lang})...")

        dataset = self._load_wikineural(lang)
        if dataset is None:
            return None

        y_true = []
        y_pred = []
        errors = []

        for sample in tqdm(dataset, desc=f"  {lang}", leave=False):
            tokens = sample["tokens"]
            ner_tags = sample["ner_tags"]

            # Convert WikiNeural tags to unified labels
            true_labels = [
                WIKINEURAL_TO_UNIFIED.get(self._wikineural_id_to_label(tag), "O")
                for tag in ner_tags
            ]

            # Get predictions
            pred_labels = self._predict_sample(tokens)

            # Align lengths
            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]

            y_true.extend(true_labels)
            y_pred.extend(pred_labels)

            # Track errors for analysis
            for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
                if t != p and t != "O":
                    errors.append({
                        "token": tokens[i] if i < len(tokens) else "",
                        "true": t,
                        "pred": p,
                    })

        # Compute metrics
        labels = [l for l in set(y_true) if l != "O"]
        if not labels:
            logger.warning(f"No entity labels found for {lang}")
            labels = ["PERSON", "LOCATION"]  # Default labels

        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        per_class_f1 = {
            label: report[label]["f1-score"]
            for label in labels
            if label in report
        }

        return LanguageResult(
            language=lang,
            language_name=lang_info["name"],
            family=lang_info["family"],
            n_samples=len(dataset),
            f1_macro=report["macro avg"]["f1-score"],
            f1_weighted=report["weighted avg"]["f1-score"],
            precision=report["macro avg"]["precision"],
            recall=report["macro avg"]["recall"],
            per_class_f1=per_class_f1,
            errors=errors[:100],  # Keep top 100 errors for analysis
        )

    def run_evaluation(self) -> MultilingualResult:
        """Run evaluation across all configured languages."""
        logger.info(f"Starting multilingual evaluation on {len(self.languages)} languages")

        result = MultilingualResult(model_name="NerGuard")

        for lang in self.languages:
            lang_result = self.evaluate_language(lang)
            if lang_result:
                result.language_results.append(lang_result)
                logger.info(
                    f"  {lang}: F1={lang_result.f1_macro:.3f}, "
                    f"P={lang_result.precision:.3f}, R={lang_result.recall:.3f}"
                )

        return result

    def generate_report(self, result: MultilingualResult) -> str:
        """Generate comprehensive multilingual evaluation report."""
        lines = []
        lines.append("=" * 80)
        lines.append("MULTILINGUAL EVALUATION REPORT")
        lines.append(f"Model: {result.model_name}")
        lines.append(f"Dataset: Babelscape/wikineural")
        lines.append(f"Languages: {len(result.language_results)}")
        lines.append("=" * 80)
        lines.append("")

        # Summary table
        lines.append("PERFORMANCE BY LANGUAGE")
        lines.append("-" * 80)
        lines.append(f"{'Lang':<6} {'Name':<12} {'Family':<10} {'Samples':<8} {'F1 Macro':<10} {'Precision':<10} {'Recall':<10}")
        lines.append("-" * 80)

        for r in sorted(result.language_results, key=lambda x: x.f1_macro, reverse=True):
            lines.append(
                f"{r.language:<6} {r.language_name:<12} {r.family:<10} {r.n_samples:<8} "
                f"{r.f1_macro:.3f}      {r.precision:.3f}      {r.recall:.3f}"
            )

        lines.append("")

        # Cross-lingual analysis
        lines.append("CROSS-LINGUAL ANALYSIS")
        lines.append("-" * 40)

        # Group by language family
        by_family = defaultdict(list)
        for r in result.language_results:
            by_family[r.family].append(r)

        for family, results in sorted(by_family.items()):
            avg_f1 = np.mean([r.f1_macro for r in results])
            lines.append(f"{family}: Avg F1 = {avg_f1:.3f} ({len(results)} languages)")

        lines.append("")

        # English baseline comparison
        en_result = next((r for r in result.language_results if r.language == "en"), None)
        if en_result:
            lines.append("TRANSFER DEGRADATION (vs English baseline)")
            lines.append("-" * 40)
            for r in result.language_results:
                if r.language != "en":
                    degradation = en_result.f1_macro - r.f1_macro
                    pct = 100 * degradation / en_result.f1_macro if en_result.f1_macro > 0 else 0
                    lines.append(f"{r.language_name:<12}: {degradation:+.3f} ({pct:+.1f}%)")

        lines.append("")

        # Per-class breakdown for key languages
        lines.append("PER-CLASS F1 BY LANGUAGE")
        lines.append("-" * 40)
        for r in result.language_results:
            if r.per_class_f1:
                class_str = ", ".join([f"{k}={v:.2f}" for k, v in r.per_class_f1.items()])
                lines.append(f"{r.language}: {class_str}")

        report_text = "\n".join(lines)

        # Save report
        report_path = os.path.join(self.output_dir, "multilingual_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Report saved to {report_path}")

        # Save JSON results
        json_path = os.path.join(self.output_dir, "multilingual_results.json")
        json_data = {
            "model_name": result.model_name,
            "dataset": "Babelscape/wikineural",
            "languages": [
                {
                    "code": r.language,
                    "name": r.language_name,
                    "family": r.family,
                    "samples": r.n_samples,
                    "f1_macro": r.f1_macro,
                    "f1_weighted": r.f1_weighted,
                    "precision": r.precision,
                    "recall": r.recall,
                    "per_class_f1": r.per_class_f1,
                }
                for r in result.language_results
            ],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

        return report_text


def main():
    """Run multilingual evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Multilingual NerGuard evaluation")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Max samples per language",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots/multilingual_results",
        help="Output directory",
    )
    args = parser.parse_args()

    evaluator = MultilingualEvaluator(
        model_path=args.model_path,
        languages=args.languages,
        max_samples_per_lang=args.max_samples,
        output_dir=args.output_dir,
    )

    result = evaluator.run_evaluation()
    report = evaluator.generate_report(result)
    print(report)


if __name__ == "__main__":
    main()
