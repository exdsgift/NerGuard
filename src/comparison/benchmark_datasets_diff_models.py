import os
import sys
import logging
import json
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict


try:
    import seaborn as sns

    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid", {"axes.grid": False})
except ImportError:
    pass

try:
    from src.pipeline.optimize_llmrouting import OptimizedLLMRouter
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.pipeline.optimize_llmrouting import OptimizedLLMRouter

from src.utils.visual import PlottingMixin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation_comparison.log", mode="w"),
    ],
)
logger = logging.getLogger("HybridEval")


@dataclass
class EvalConfig:
    model_path: str = "./models/mdeberta-pii-safe/final"
    output_dir: str = "./evaluation_comparison"
    max_samples: Optional[int] = 1000
    context_length: int = 512
    THRESHOLD_ENTROPY: float = 0.583
    THRESHOLD_CONF: float = 0.787


NVIDIA_TO_MODEL_MAP = {
    "first_name": "GIVENNAME",
    "last_name": "SURNAME",
    "middle_name": "GIVENNAME",
    "name": "GIVENNAME",
    "user_name": "GIVENNAME",
    "email": "EMAIL",
    "phone_number": "TELEPHONENUM",
    "cell_phone": "TELEPHONENUM",
    "fax_number": "TELEPHONENUM",
    "city": "CITY",
    "zipcode": "ZIPCODE",
    "postcode": "ZIPCODE",
    "ssn": "SOCIALNUM",
    "social_security_number": "SOCIALNUM",
    "tax_id": "TAXNUM",
    "driver_license": "DRIVERLICENSENUM",
    "credit_debit_card": "CREDITCARDNUMBER",
    "date": "DATE",
    "date_of_birth": "DATE",
    "time": "TIME",
    "age": "AGE",
    "gender": "GENDER",
    "street_address": "STREET",
    "company_name": "O",
    "organization": "O",
    "url": "O",
    "ip_address": "O",
    "country": "O",
    "state": "O",
    "national_id": "O",
    "passport_number": "O",
}


@dataclass
class HybridResult:
    true_ids: List[int] = field(default_factory=list)
    baseline_preds: List[int] = field(default_factory=list)
    hybrid_preds: List[int] = field(default_factory=list)
    llm_calls: int = 0
    llm_corrections: int = 0
    llm_wrong_corrections: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    entropy_scores: List[float] = field(default_factory=list)
    llm_interventions: List[Dict] = field(default_factory=list)
    per_class_stats: Dict[str, Dict] = field(
        default_factory=lambda: defaultdict(
            lambda: {
                "total": 0,
                "baseline_correct": 0,
                "hybrid_correct": 0,
                "llm_called": 0,
                "llm_helped": 0,
                "llm_hurt": 0,
            }
        )
    )


class LabelMapper:
    def __init__(self, model_id2label: Dict[int, str]):
        self.model_id2label = model_id2label
        self.model_label2id = {v: k for k, v in model_id2label.items()}

    def get_token_label_id(self, nv_label: str, is_start: bool) -> int:
        target_base = NVIDIA_TO_MODEL_MAP.get(nv_label, "O")
        if target_base == "O":
            return self.model_label2id.get("O", 0)
        prefix = "B-" if is_start else "I-"
        full_label = f"{prefix}{target_base}"
        return self.model_label2id.get(full_label, self.model_label2id.get("O", 0))


class HybridEvaluator(PlottingMixin):
    def __init__(
        self, config: EvalConfig, llm_router: Optional[OptimizedLLMRouter] = None
    ):
        self.config = config
        self.llm_router = llm_router
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_resources()
        self._setup_plotting()
        os.makedirs(config.output_dir, exist_ok=True)

    def _init_resources(self):
        logger.info(f"Loading resources from: {self.config.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_path
            )
            self.model.to(self.device)
            self.model.eval()
            self.mapper = LabelMapper(self.model.config.id2label)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            sys.exit(1)

    def _setup_plotting(self):
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def load_and_preprocess(self):
        logger.info("Loading NVIDIA/Nemotron-PII dataset...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train")
        if self.config.max_samples:
            ds = ds.select(range(min(len(ds), self.config.max_samples)))
        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _tokenize_and_align(self, text: str, spans: List[Dict]) -> Dict:
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.context_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = tokenized["offset_mapping"][0].numpy()
        labels = [self.mapper.model_label2id.get("O", 0)] * len(offset_mapping)

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                labels[idx] = -100
                continue

            for span in spans:
                if end <= span["start"] or start >= span["end"]:
                    continue

                is_start = start == span["start"]
                if not is_start and idx > 0 and labels[idx - 1] != -100:
                    prev_label = self.mapper.model_id2label.get(labels[idx - 1], "O")
                    curr_base = NVIDIA_TO_MODEL_MAP.get(span["label"], "O")
                    if curr_base not in prev_label:
                        is_start = True

                labels[idx] = self.mapper.get_token_label_id(span["label"], is_start)
                break

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "offset_mapping": offset_mapping,
            "labels": torch.tensor([labels]),
            "text": text,
        }

    @staticmethod
    def _parse_spans(spans: Any) -> List[Dict]:
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except:
                try:
                    spans = ast.literal_eval(spans)
                except:
                    return []
        if isinstance(spans, list):
            valid = [
                s
                for s in spans
                if isinstance(s, dict) and {"label", "start", "end"} <= s.keys()
            ]
            return sorted(valid, key=lambda x: x["start"])
        return []

    def evaluate_comparison(self, dataset):
        logger.info(f"Starting Comparative Inference on {len(dataset)} samples...")
        logger.info(
            f"Using thresholds - Confidence: {self.config.THRESHOLD_CONF}, Entropy: {self.config.THRESHOLD_ENTROPY}"
        )

        res = HybridResult()

        for i, example in tqdm(
            enumerate(dataset), total=len(dataset), desc="Processing"
        ):
            spans = self._parse_spans(example["spans"])
            full_text = example["text"]

            processed = self._tokenize_and_align(full_text, spans)
            input_ids = processed["input_ids"].to(self.device)
            mask = processed["attention_mask"].to(self.device)
            labels = processed["labels"].to(self.device)
            offsets = processed["offset_mapping"]

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=mask)
                logits = outputs.logits

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                confidence, preds = torch.max(probs, dim=-1)

            preds_flat = preds[0].cpu().numpy()
            conf_flat = confidence[0].cpu().numpy()
            entropy_flat = entropy[0].cpu().numpy()
            labels_flat = labels[0].cpu().numpy()

            hybrid_preds_flat = preds_flat.copy()

            if self.llm_router:
                for idx, (token_id, pred_id, conf, ent) in enumerate(
                    zip(input_ids[0], preds_flat, conf_flat, entropy_flat)
                ):
                    if labels_flat[idx] == -100:
                        continue

                    true_label = self.mapper.model_id2label[labels_flat[idx]]
                    pred_label = self.mapper.model_id2label[pred_id]

                    res.per_class_stats[true_label]["total"] += 1
                    if pred_id == labels_flat[idx]:
                        res.per_class_stats[true_label]["baseline_correct"] += 1

                    res.confidence_scores.append(float(conf))
                    res.entropy_scores.append(float(ent))

                    is_uncertain = (conf < self.config.THRESHOLD_CONF) and (
                        ent > self.config.THRESHOLD_ENTROPY
                    )

                    if is_uncertain and pred_label != "O" and true_label != "O":
                        start_char, end_char = offsets[idx]
                        if start_char == end_char:
                            continue

                        token_text = full_text[start_char:end_char]
                        current_pred_label = self.mapper.model_id2label[pred_id]
                        prev_label = (
                            self.mapper.model_id2label[hybrid_preds_flat[idx - 1]]
                            if idx > 0
                            else "O"
                        )

                        res.llm_calls += 1
                        res.per_class_stats[true_label]["llm_called"] += 1

                        llm_out = self.llm_router.disambiguate(
                            target_token=token_text,
                            full_text=full_text,
                            char_start=start_char,
                            char_end=end_char,
                            current_pred=current_pred_label,
                            prev_label=prev_label,
                        )

                        corrected_label = llm_out.get("corrected_label", "O")
                        corrected_id = self.mapper.model_label2id.get(
                            corrected_label, self.mapper.model_label2id.get("O")
                        )

                        intervention = {
                            "token": token_text,
                            "true_label": true_label,
                            "baseline_pred": pred_label,
                            "llm_pred": corrected_label,
                            "confidence": float(conf),
                            "entropy": float(ent),
                            "was_correct_before": pred_id == labels_flat[idx],
                            "is_correct_after": corrected_id == labels_flat[idx],
                        }
                        res.llm_interventions.append(intervention)

                        if corrected_id != pred_id:
                            hybrid_preds_flat[idx] = corrected_id

                            if (
                                corrected_id == labels_flat[idx]
                                and pred_id != labels_flat[idx]
                            ):
                                res.llm_corrections += 1
                                res.per_class_stats[true_label]["llm_helped"] += 1
                            elif (
                                corrected_id != labels_flat[idx]
                                and pred_id == labels_flat[idx]
                            ):
                                res.llm_wrong_corrections += 1
                                res.per_class_stats[true_label]["llm_hurt"] += 1

                    if hybrid_preds_flat[idx] == labels_flat[idx]:
                        res.per_class_stats[true_label]["hybrid_correct"] += 1

            active_mask = labels_flat != -100
            res.true_ids.extend(labels_flat[active_mask])
            res.baseline_preds.extend(preds_flat[active_mask])
            res.hybrid_preds.extend(hybrid_preds_flat[active_mask])

        logger.info(
            f"Evaluation complete. LLM calls: {res.llm_calls}, Helpful: {res.llm_corrections}, Harmful: {res.llm_wrong_corrections}"
        )
        self._generate_comparison_report(res)

    def _generate_comparison_report(self, res: HybridResult):
        logger.info("Generating comprehensive comparison report...")

        id2label = self.model.config.id2label
        y_true = [id2label[i] for i in res.true_ids]
        y_base = [id2label[i] for i in res.baseline_preds]
        y_hyb = [id2label[i] for i in res.hybrid_preds]

        labels = sorted(list({l for l in y_true if l != "O"}))

        base_report = classification_report(
            y_true, y_base, labels=labels, zero_division=0
        )
        hyb_report = classification_report(
            y_true, y_hyb, labels=labels, zero_division=0
        )

        # Report testuale
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BASELINE MODEL (DeBERTa Only)")
        report_lines.append("=" * 80)
        report_lines.append(base_report)

        report_lines.append("\n" + "=" * 80)
        report_lines.append(f"HYBRID MODEL (DeBERTa + LLM Router)")
        report_lines.append(
            f"Logic: (Conf < {self.config.THRESHOLD_CONF}) OR (Entropy > {self.config.THRESHOLD_ENTROPY})"
        )
        report_lines.append(f"LLM Calls: {res.llm_calls}")
        report_lines.append(f"Helpful Corrections: {res.llm_corrections}")
        report_lines.append(f"Harmful Corrections: {res.llm_wrong_corrections}")
        report_lines.append(
            f"Net Improvement: {res.llm_corrections - res.llm_wrong_corrections}"
        )
        report_lines.append("=" * 80)
        report_lines.append(hyb_report)

        # Analisi per-class
        report_lines.append("\n" + "=" * 80)
        report_lines.append("PER-CLASS LLM IMPACT ANALYSIS")
        report_lines.append("=" * 80)

        for label in sorted(res.per_class_stats.keys()):
            stats = res.per_class_stats[label]
            if stats["total"] == 0:
                continue

            baseline_acc = stats["baseline_correct"] / stats["total"] * 100
            hybrid_acc = stats["hybrid_correct"] / stats["total"] * 100
            improvement = hybrid_acc - baseline_acc

            report_lines.append(f"\n{label}:")
            report_lines.append(f"  Total instances: {stats['total']}")
            report_lines.append(f"  Baseline accuracy: {baseline_acc:.1f}%")
            report_lines.append(f"  Hybrid accuracy: {hybrid_acc:.1f}%")
            report_lines.append(f"  Improvement: {improvement:+.1f}%")
            report_lines.append(f"  LLM interventions: {stats['llm_called']}")
            report_lines.append(
                f"  Helped: {stats['llm_helped']}, Hurt: {stats['llm_hurt']}"
            )

        report_text = "\n".join(report_lines)

        # Log e salva
        for line in report_lines:
            logger.info(line)

        with open(
            os.path.join(self.config.output_dir, "comparison_report.txt"), "w"
        ) as f:
            f.write(report_text)

        # Genera tutti i grafici
        self._plot_confusion_matrices(y_true, y_base, y_hyb, labels)
        self._plot_per_class_improvements(res)
        self._plot_confidence_entropy_analysis(res)
        self._plot_llm_impact_analysis(res)
        self._plot_error_analysis(y_true, y_base, y_hyb)

        logger.info(f"All reports and visualizations saved to {self.config.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from dotenv import load_dotenv

    load_dotenv()

    config = EvalConfig(
        model_path="./models/mdeberta-pii-safe/final",
        output_dir="./evaluation_comparison",
        max_samples=1000,
    )

    router = None
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Initializing OpenAI Router...")
        router = OptimizedLLMRouter(
            source="openai", model="gpt-4o-mini", enable_cache=True
        )
    else:
        logger.warning("No OPENAI_API_KEY found. Trying Ollama...")
        try:
            router = OptimizedLLMRouter(
                source="ollama", ollama_model="qwen2.5:3b", enable_cache=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama router: {e}")

    if router is None:
        logger.warning("Running in baseline-only mode (no LLM router available)")

    evaluator = HybridEvaluator(config, llm_router=router)
    dataset = evaluator.load_and_preprocess()
    evaluator.evaluate_comparison(dataset)
