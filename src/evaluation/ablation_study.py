"""
Ablation Study for NerGuard Hybrid System.

This module implements systematic ablation studies to analyze:
1. Uncertainty measures: Entropy-only vs Confidence-only vs Combined
2. Threshold sensitivity: Performance across threshold ranges
3. Routing strategies: No selective vs Selective vs I-blocking

These studies address the thesis requirement for rigorous experimental analysis.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from collections import defaultdict

from src.inference.llm_router import LLMRouter
from src.inference.entity_router import EntitySpecificRouter
from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    NVIDIA_TO_MODEL_MAP,
)
from src.core.label_mapper import LabelMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    config_name: str
    entropy_threshold: Optional[float]
    confidence_threshold: Optional[float]
    use_entropy: bool
    use_confidence: bool
    selective_routing: bool
    block_continuation: bool

    # Metrics
    f1_macro: float
    f1_weighted: float
    precision: float
    recall: float

    # LLM stats
    tokens_routed: int
    llm_calls: int
    helpful: int
    harmful: int
    net_improvement: int

    # Efficiency
    routing_rate: float  # % of tokens routed


@dataclass
class AblationStudy:
    """Container for ablation study results."""
    study_name: str
    results: List[AblationResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "study_name": self.study_name,
            "results": [
                {
                    "config_name": r.config_name,
                    "entropy_threshold": r.entropy_threshold,
                    "confidence_threshold": r.confidence_threshold,
                    "use_entropy": r.use_entropy,
                    "use_confidence": r.use_confidence,
                    "selective_routing": r.selective_routing,
                    "block_continuation": r.block_continuation,
                    "f1_macro": r.f1_macro,
                    "f1_weighted": r.f1_weighted,
                    "precision": r.precision,
                    "recall": r.recall,
                    "tokens_routed": r.tokens_routed,
                    "llm_calls": r.llm_calls,
                    "helpful": r.helpful,
                    "harmful": r.harmful,
                    "net_improvement": r.net_improvement,
                    "routing_rate": r.routing_rate,
                }
                for r in self.results
            ]
        }


class AblationRunner:
    """
    Runs systematic ablation studies on the hybrid NER system.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        output_dir: str = "./plots/ablation_study",
        max_samples: int = 500,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.max_samples = max_samples

        os.makedirs(output_dir, exist_ok=True)

        # Load model once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.mapper = LabelMapper(self.model.config.id2label)

        # Initialize LLM router (shared across configs)
        self.llm_router = self._init_llm_router()

        # Load dataset once
        self.dataset = self._load_dataset()

    def _init_llm_router(self) -> Optional[LLMRouter]:
        """Initialize LLM router from environment."""
        from dotenv import load_dotenv
        load_dotenv()

        if os.getenv("OPENAI_API_KEY"):
            logger.info("Initializing OpenAI Router...")
            return LLMRouter(
                source="openai", model="gpt-4o-mini", enable_cache=True
            )
        else:
            logger.warning("No OPENAI_API_KEY found. Trying Ollama...")
            try:
                return LLMRouter(
                    source="ollama", ollama_model="qwen2.5:3b", enable_cache=True
                )
            except Exception as e:
                logger.error(f"Failed to initialize router: {e}")
                return None

    def _load_dataset(self):
        """Load NVIDIA dataset."""
        logger.info("Loading NVIDIA/Nemotron-PII dataset...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train")
        if self.max_samples:
            ds = ds.select(range(min(len(ds), self.max_samples)))
        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _parse_spans(self, spans_str: str) -> List[Dict]:
        """Parse spans from dataset."""
        import ast
        try:
            if isinstance(spans_str, list):
                return spans_str
            return ast.literal_eval(spans_str) if spans_str else []
        except:
            return []

    def _tokenize_and_align(self, text: str, spans: List[Dict]) -> Dict:
        """Tokenize and align labels."""
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
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

                prefix = "B-" if is_start else "I-"
                base_label = NVIDIA_TO_MODEL_MAP.get(span["label"], "O")
                if base_label != "O":
                    full_label = f"{prefix}{base_label}"
                    if full_label in self.mapper.model_label2id:
                        labels[idx] = self.mapper.model_label2id[full_label]
                break

        tokenized["labels"] = torch.tensor([labels])
        return tokenized

    def _run_single_config(
        self,
        config_name: str,
        entropy_threshold: float,
        confidence_threshold: float,
        use_entropy: bool,
        use_confidence: bool,
        selective_routing: bool,
        block_continuation: bool,
    ) -> AblationResult:
        """Run evaluation with a specific configuration."""

        # Set effective thresholds
        # If not using entropy, set threshold to 0 (always passes entropy check)
        # If not using confidence, set threshold to 1 (always passes confidence check)
        effective_entropy = entropy_threshold if use_entropy else 0.0
        effective_confidence = confidence_threshold if use_confidence else 1.0

        # Create entity router with these settings
        entity_router = EntitySpecificRouter(
            entropy_threshold=effective_entropy,
            confidence_threshold=effective_confidence,
            enable_selective=selective_routing,
            block_continuation_tokens=block_continuation,
        )

        # Tracking
        true_ids = []
        baseline_preds = []
        hybrid_preds = []
        llm_calls = 0
        helpful = 0
        harmful = 0
        total_tokens = 0
        routed_tokens = 0

        for example in tqdm(self.dataset, desc=f"  {config_name}", leave=False):
            spans = self._parse_spans(example["spans"])
            full_text = example["text"]

            processed = self._tokenize_and_align(full_text, spans)
            input_ids = processed["input_ids"].to(self.device)
            mask = processed["attention_mask"].to(self.device)
            labels = processed["labels"].to(self.device)
            offsets = processed["offset_mapping"][0].numpy()

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

                    total_tokens += 1
                    true_label = self.mapper.model_id2label[labels_flat[idx]]
                    pred_label = self.mapper.model_id2label[pred_id]

                    # Check if we should route to LLM
                    should_route = entity_router.should_route(
                        predicted_label=pred_label,
                        confidence=float(conf),
                        entropy=float(ent),
                    )

                    if should_route:
                        routed_tokens += 1
                        start, end = offsets[idx]

                        # Get previous label for context
                        if idx > 0 and labels_flat[idx-1] != -100:
                            prev_label = self.mapper.model_id2label[int(hybrid_preds_flat[idx-1])]
                        else:
                            prev_label = "O"

                        # Call LLM for disambiguation
                        result = self.llm_router.disambiguate(
                            target_token=full_text[start:end],
                            full_text=full_text,
                            char_start=int(start),
                            char_end=int(end),
                            current_pred=pred_label,
                            prev_label=prev_label,
                        )
                        llm_calls += 1

                        llm_pred = result.get("corrected_label")
                        if llm_pred and llm_pred != pred_label:
                            if llm_pred in self.mapper.model_label2id:
                                new_id = self.mapper.model_label2id[llm_pred]
                                hybrid_preds_flat[idx] = new_id

                                # Check if it helped or hurt
                                was_correct = (pred_id == labels_flat[idx])
                                is_correct = (new_id == labels_flat[idx])

                                if not was_correct and is_correct:
                                    helpful += 1
                                elif was_correct and not is_correct:
                                    harmful += 1

            active_mask = labels_flat != -100
            true_ids.extend(labels_flat[active_mask])
            baseline_preds.extend(preds_flat[active_mask])
            hybrid_preds.extend(hybrid_preds_flat[active_mask])

        # Calculate metrics
        id2label = self.model.config.id2label
        y_true = [id2label[i] for i in true_ids]
        y_hyb = [id2label[i] for i in hybrid_preds]

        labels = sorted(list({l for l in y_true if l != "O"}))

        report = classification_report(
            y_true, y_hyb, labels=labels, output_dict=True, zero_division=0
        )

        routing_rate = routed_tokens / total_tokens if total_tokens > 0 else 0.0

        return AblationResult(
            config_name=config_name,
            entropy_threshold=entropy_threshold if use_entropy else None,
            confidence_threshold=confidence_threshold if use_confidence else None,
            use_entropy=use_entropy,
            use_confidence=use_confidence,
            selective_routing=selective_routing,
            block_continuation=block_continuation,
            f1_macro=report.get("macro avg", {}).get("f1-score", 0.0),
            f1_weighted=report.get("weighted avg", {}).get("f1-score", 0.0),
            precision=report.get("weighted avg", {}).get("precision", 0.0),
            recall=report.get("weighted avg", {}).get("recall", 0.0),
            tokens_routed=routed_tokens,
            llm_calls=llm_calls,
            helpful=helpful,
            harmful=harmful,
            net_improvement=helpful - harmful,
            routing_rate=routing_rate,
        )

    def run_uncertainty_ablation(self) -> AblationStudy:
        """
        Ablation Study 1: Uncertainty Measures

        Compares:
        - Entropy-only: Route based only on high entropy
        - Confidence-only: Route based only on low confidence
        - Combined: Route based on both (current approach)
        """
        logger.info("=" * 60)
        logger.info("ABLATION STUDY 1: Uncertainty Measures")
        logger.info("=" * 60)

        study = AblationStudy(study_name="uncertainty_measures")

        configs = [
            # (name, use_entropy, use_confidence)
            ("Entropy Only", True, False),
            ("Confidence Only", False, True),
            ("Combined (E+C)", True, True),
        ]

        for name, use_entropy, use_confidence in configs:
            logger.info(f"\nRunning config: {name}")

            result = self._run_single_config(
                config_name=name,
                entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                use_entropy=use_entropy,
                use_confidence=use_confidence,
                selective_routing=True,
                block_continuation=True,
            )
            study.results.append(result)

            logger.info(f"  Net improvement: {result.net_improvement}")
            logger.info(f"  Routing rate: {result.routing_rate:.2%}")
            logger.info(f"  Helpful/Harmful: {result.helpful}/{result.harmful}")

        return study

    def run_threshold_sensitivity(self) -> AblationStudy:
        """
        Ablation Study 2: Threshold Sensitivity

        Analyzes how performance varies with different threshold values.
        Tests a grid of entropy and confidence thresholds.
        """
        logger.info("=" * 60)
        logger.info("ABLATION STUDY 2: Threshold Sensitivity")
        logger.info("=" * 60)

        study = AblationStudy(study_name="threshold_sensitivity")

        # Test key threshold combinations
        test_configs = [
            # (entropy, confidence, name)
            (0.3, 0.6, "Low (E=0.3, C=0.6)"),
            (0.5, 0.7, "Medium (E=0.5, C=0.7)"),
            (0.583, 0.787, "Default (E=0.583, C=0.787)"),
            (0.7, 0.85, "High (E=0.7, C=0.85)"),
            (0.9, 0.95, "Very High (E=0.9, C=0.95)"),
        ]

        for entropy_t, confidence_t, name in test_configs:
            logger.info(f"\nRunning config: {name}")

            result = self._run_single_config(
                config_name=name,
                entropy_threshold=entropy_t,
                confidence_threshold=confidence_t,
                use_entropy=True,
                use_confidence=True,
                selective_routing=True,
                block_continuation=True,
            )
            study.results.append(result)

            logger.info(f"  Net improvement: {result.net_improvement}")
            logger.info(f"  Tokens routed: {result.tokens_routed}")

        return study

    def run_routing_strategy_ablation(self) -> AblationStudy:
        """
        Ablation Study 3: Routing Strategies

        Compares:
        - No selective routing (route all above threshold)
        - Selective routing (entity-specific)
        - Selective + I-blocking (current best)
        """
        logger.info("=" * 60)
        logger.info("ABLATION STUDY 3: Routing Strategies")
        logger.info("=" * 60)

        study = AblationStudy(study_name="routing_strategies")

        configs = [
            # (name, selective_routing, block_continuation)
            ("No Selective", False, False),
            ("Selective Only", True, False),
            ("Selective + I-Block", True, True),
        ]

        for name, selective, block_cont in configs:
            logger.info(f"\nRunning config: {name}")

            result = self._run_single_config(
                config_name=name,
                entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                use_entropy=True,
                use_confidence=True,
                selective_routing=selective,
                block_continuation=block_cont,
            )
            study.results.append(result)

            logger.info(f"  Net improvement: {result.net_improvement}")
            logger.info(f"  Helpful/Harmful: {result.helpful}/{result.harmful}")

        return study

    def generate_report(self, studies: List[AblationStudy]) -> str:
        """Generate comprehensive ablation study report."""
        lines = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("NerGuard Hybrid System Analysis")
        lines.append("=" * 80)
        lines.append(f"\nModel: {self.model_path}")
        lines.append(f"Samples: {self.max_samples}")
        lines.append("")

        for study in studies:
            lines.append(f"\n{'='*60}")
            lines.append(f"Study: {study.study_name.upper().replace('_', ' ')}")
            lines.append("=" * 60)
            lines.append("")

            # Header
            lines.append(f"{'Config':<30} {'Net Imp':>8} {'Help':>6} {'Harm':>6} {'Route%':>8} {'F1-W':>6}")
            lines.append("-" * 70)

            for r in study.results:
                lines.append(
                    f"{r.config_name:<30} {r.net_improvement:>+8} "
                    f"{r.helpful:>6} {r.harmful:>6} "
                    f"{r.routing_rate:>7.2%} {r.f1_weighted:>6.3f}"
                )

            lines.append("")

            # Best config
            if study.results:
                best = max(study.results, key=lambda x: x.net_improvement)
                lines.append(f"Best config: {best.config_name}")
                lines.append(f"  Net improvement: {best.net_improvement}")
                lines.append(f"  Efficiency (help/harm ratio): {best.helpful}/{best.harmful}")
            lines.append("")

        # Summary recommendations
        lines.append("\n" + "=" * 60)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 60)

        # Find best overall config
        all_results = [r for s in studies for r in s.results]
        if all_results:
            best_overall = max(all_results, key=lambda x: x.net_improvement)

            lines.append(f"\nBest overall configuration:")
            lines.append(f"  Name: {best_overall.config_name}")
            if best_overall.entropy_threshold is not None:
                lines.append(f"  Entropy threshold: {best_overall.entropy_threshold}")
            if best_overall.confidence_threshold is not None:
                lines.append(f"  Confidence threshold: {best_overall.confidence_threshold}")
            lines.append(f"  Selective routing: {best_overall.selective_routing}")
            lines.append(f"  Block I- tokens: {best_overall.block_continuation}")
            lines.append(f"  Net improvement: {best_overall.net_improvement}")
            lines.append(f"  Routing rate: {best_overall.routing_rate:.2%}")

        report_text = "\n".join(lines)

        # Save report
        report_path = os.path.join(self.output_dir, "ablation_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        logger.info(f"Report saved to {report_path}")

        # Save JSON
        json_data = {"studies": [s.to_dict() for s in studies]}
        json_path = os.path.join(self.output_dir, "ablation_results.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON saved to {json_path}")

        return report_text

    def run_all(self) -> List[AblationStudy]:
        """Run all ablation studies."""
        studies = []

        # Study 1: Uncertainty measures
        studies.append(self.run_uncertainty_ablation())

        # Study 2: Threshold sensitivity
        studies.append(self.run_threshold_sensitivity())

        # Study 3: Routing strategies
        studies.append(self.run_routing_strategy_ablation())

        # Generate report
        report = self.generate_report(studies)
        print(report)

        return studies


def main():
    """Run ablation studies."""
    import argparse

    parser = argparse.ArgumentParser(description="NerGuard Ablation Studies")
    parser.add_argument(
        "--model-path",
        default="./models/mdeberta-pii-safe/final",
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-dir",
        default="./plots/ablation_study",
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max samples for evaluation (use smaller for faster testing)",
    )
    parser.add_argument(
        "--study",
        choices=["uncertainty", "threshold", "routing", "all"],
        default="all",
        help="Which study to run",
    )
    args = parser.parse_args()

    runner = AblationRunner(
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    if args.study == "uncertainty":
        study = runner.run_uncertainty_ablation()
        runner.generate_report([study])
    elif args.study == "threshold":
        study = runner.run_threshold_sensitivity()
        runner.generate_report([study])
    elif args.study == "routing":
        study = runner.run_routing_strategy_ablation()
        runner.generate_report([study])
    else:
        runner.run_all()


if __name__ == "__main__":
    main()
