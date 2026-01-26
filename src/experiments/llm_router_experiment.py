"""
LLM Router Experiment Script for NerGuard.

Systematically evaluates different OpenAI models and prompts to find
the optimal configuration for LLM-based disambiguation of uncertain NER predictions.

Usage:
    python -m src.experiments.llm_router_experiment --n-samples 100
"""

import os
import json
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dotenv import load_dotenv

from src.inference.llm_router import LLMRouter
from src.inference.prompts import PROMPTS
import src.inference.prompts as prompts_module
from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    NVIDIA_TO_MODEL_MAP,
)
from src.core.label_mapper import LabelMapper
from src.visualization.style import set_publication_style

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LLMExperiment")


@dataclass
class ExperimentConfig:
    """Configuration for LLM Router experiment."""
    n_samples: int = 100
    model_path: str = DEFAULT_MODEL_PATH
    output_dir: str = "./experiments/llm_router"
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    context_length: int = 512

    models_to_test: List[str] = field(default_factory=lambda: [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-4-turbo",
    ])
    prompts_to_test: List[str] = field(default_factory=lambda: [
        "V11", "V9", "V_EXP"
    ])


@dataclass
class TokenResult:
    """Result for a single token intervention."""
    sample_idx: int
    token_idx: int
    token_text: str
    context: str
    ground_truth: str
    model_prediction: str
    llm_correction: str
    entropy: float
    confidence: float
    model_was_correct: bool
    llm_is_correct: bool
    was_helpful: bool
    was_harmful: bool
    reasoning: str
    latency_ms: float


@dataclass
class ExperimentRun:
    """Results from a single model+prompt configuration."""
    model_name: str
    prompt_version: str
    total_tokens_processed: int = 0
    total_llm_calls: int = 0
    helpful_corrections: int = 0
    harmful_corrections: int = 0
    neutral_changes: int = 0
    total_latency_ms: float = 0.0
    per_class_stats: Dict[str, Dict] = field(default_factory=lambda: defaultdict(
        lambda: {"total": 0, "helpful": 0, "harmful": 0, "neutral": 0}
    ))
    token_results: List[TokenResult] = field(default_factory=list)

    @property
    def net_improvement(self) -> int:
        return self.helpful_corrections - self.harmful_corrections

    @property
    def correction_accuracy(self) -> float:
        total = self.helpful_corrections + self.harmful_corrections
        return self.helpful_corrections / total if total > 0 else 0.0

    @property
    def harm_rate(self) -> float:
        return self.harmful_corrections / self.total_llm_calls if self.total_llm_calls > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_llm_calls if self.total_llm_calls > 0 else 0.0


class LLMRouterExperiment:
    """
    Systematic experiment to compare LLM models and prompts for NER disambiguation.
    """

    # Estimated cost per 1K tokens (input + output combined estimate)
    MODEL_COSTS = {
        "gpt-4o-mini": 0.00030,
        "gpt-4o": 0.0075,
        "gpt-3.5-turbo": 0.001,
        "gpt-4-turbo": 0.015,
    }

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(config.output_dir, exist_ok=True)
        self._init_model()
        set_publication_style()

    def _init_model(self):
        """Initialize the base NER model."""
        logger.info(f"Loading model from: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.mapper = LabelMapper(self.model.config.id2label)
        logger.info(f"Model loaded on {self.device}")

    def _load_dataset(self) -> Any:
        """Load samples from NVIDIA dataset."""
        logger.info("Loading NVIDIA/Nemotron-PII dataset...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train")
        ds = ds.select(range(min(len(ds), self.config.n_samples)))
        logger.info(f"Loaded {len(ds)} samples")
        return ds

    def _parse_spans(self, spans: Any) -> List[Dict]:
        """Parse spans from various formats."""
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except:
                try:
                    import ast
                    spans = ast.literal_eval(spans)
                except:
                    return []
        if isinstance(spans, list):
            valid = [s for s in spans if isinstance(s, dict) and {"label", "start", "end"} <= s.keys()]
            return sorted(valid, key=lambda x: x["start"])
        return []

    def _tokenize_and_align(self, text: str, spans: List[Dict]) -> Dict:
        """Tokenize text and align with span labels."""
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

    def _set_prompt(self, version: str):
        """Dynamically set the prompt version."""
        if version not in PROMPTS:
            raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPTS.keys())}")
        prompts_module.PROMPT = PROMPTS[version]
        logger.info(f"Set prompt to {version}")

    def _run_single_configuration(
        self,
        dataset: Any,
        model_name: str,
        prompt_version: str,
    ) -> ExperimentRun:
        """Run experiment with a specific model+prompt configuration."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: Model={model_name}, Prompt={prompt_version}")
        logger.info(f"{'='*60}")

        # Set prompt before creating router
        self._set_prompt(prompt_version)

        # Create fresh router with caching DISABLED for fair comparison
        router = LLMRouter(
            source="openai",
            model=model_name,
            enable_cache=False,
        )

        result = ExperimentRun(model_name=model_name, prompt_version=prompt_version)

        for sample_idx, example in tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc=f"{model_name}/{prompt_version}"
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

            # Process each token
            for idx, (token_id, pred_id, conf, ent) in enumerate(
                zip(input_ids[0], preds_flat, conf_flat, entropy_flat)
            ):
                if labels_flat[idx] == -100:
                    continue

                true_label = self.mapper.model_id2label[labels_flat[idx]]
                pred_label = self.mapper.model_id2label[pred_id]

                # Check if uncertain (triggers LLM)
                is_uncertain = (conf < self.config.confidence_threshold) and (
                    ent > self.config.entropy_threshold
                )

                if not is_uncertain:
                    continue

                # Get token info
                start_char, end_char = offsets[idx]
                if start_char == end_char:
                    continue

                token_text = full_text[start_char:end_char]
                prev_label = self.mapper.model_id2label[preds_flat[idx - 1]] if idx > 0 else "O"

                result.total_llm_calls += 1

                # Call LLM
                start_time = time.time()
                llm_out = router.disambiguate(
                    target_token=token_text,
                    full_text=full_text,
                    char_start=start_char,
                    char_end=end_char,
                    current_pred=pred_label,
                    prev_label=prev_label,
                )
                latency = (time.time() - start_time) * 1000
                result.total_latency_ms += latency

                corrected_label = llm_out.get("corrected_label", pred_label)
                corrected_id = self.mapper.model_label2id.get(
                    corrected_label, self.mapper.model_label2id.get("O")
                )

                model_was_correct = pred_id == labels_flat[idx]
                llm_is_correct = corrected_id == labels_flat[idx]

                was_helpful = not model_was_correct and llm_is_correct
                was_harmful = model_was_correct and not llm_is_correct

                if was_helpful:
                    result.helpful_corrections += 1
                    result.per_class_stats[true_label]["helpful"] += 1
                elif was_harmful:
                    result.harmful_corrections += 1
                    result.per_class_stats[true_label]["harmful"] += 1
                elif corrected_id != pred_id:
                    result.neutral_changes += 1
                    result.per_class_stats[true_label]["neutral"] += 1

                result.per_class_stats[true_label]["total"] += 1

                # Record token result
                token_result = TokenResult(
                    sample_idx=sample_idx,
                    token_idx=idx,
                    token_text=token_text,
                    context=full_text[max(0, start_char-50):min(len(full_text), end_char+50)],
                    ground_truth=true_label,
                    model_prediction=pred_label,
                    llm_correction=corrected_label,
                    entropy=float(ent),
                    confidence=float(conf),
                    model_was_correct=model_was_correct,
                    llm_is_correct=llm_is_correct,
                    was_helpful=was_helpful,
                    was_harmful=was_harmful,
                    reasoning=llm_out.get("reasoning", ""),
                    latency_ms=latency,
                )
                result.token_results.append(token_result)

        self._log_run_summary(result)
        return result

    def _log_run_summary(self, result: ExperimentRun):
        """Log summary for a single run."""
        logger.info(f"\nResults for {result.model_name} + {result.prompt_version}:")
        logger.info(f"  LLM Calls: {result.total_llm_calls}")
        logger.info(f"  Helpful: {result.helpful_corrections}")
        logger.info(f"  Harmful: {result.harmful_corrections}")
        logger.info(f"  Net Improvement: {result.net_improvement}")
        logger.info(f"  Correction Accuracy: {result.correction_accuracy:.2%}")
        logger.info(f"  Harm Rate: {result.harm_rate:.2%}")
        logger.info(f"  Avg Latency: {result.avg_latency_ms:.1f}ms")

    def _select_best_model(self, results: Dict[str, ExperimentRun]) -> str:
        """Select best model based on net improvement."""
        best_model = max(results.keys(), key=lambda m: results[m].net_improvement)
        logger.info(f"\nBest model: {best_model} (net improvement: {results[best_model].net_improvement})")
        return best_model

    def _estimate_cost(self, result: ExperimentRun) -> float:
        """Estimate API cost for a run."""
        avg_tokens_per_call = 250  # Approximate
        cost_per_1k = self.MODEL_COSTS.get(result.model_name, 0.001)
        return (result.total_llm_calls * avg_tokens_per_call / 1000) * cost_per_1k

    def run(self):
        """Execute the full two-phase experiment."""
        dataset = self._load_dataset()

        # Phase 1: Model Comparison
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: MODEL COMPARISON (using V11 prompt)")
        logger.info("="*80)

        model_results = {}
        for model_name in self.config.models_to_test:
            try:
                result = self._run_single_configuration(dataset, model_name, "V11")
                model_results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to test {model_name}: {e}")
                continue

        if not model_results:
            logger.error("No models successfully tested!")
            return

        best_model = self._select_best_model(model_results)

        # Phase 2: Prompt Comparison
        logger.info("\n" + "="*80)
        logger.info(f"PHASE 2: PROMPT COMPARISON (using {best_model})")
        logger.info("="*80)

        prompt_results = {}
        for prompt_version in self.config.prompts_to_test:
            if prompt_version == "V11" and best_model in model_results:
                # Reuse Phase 1 result for V11
                prompt_results["V11"] = model_results[best_model]
                logger.info(f"Reusing V11 results from Phase 1")
                continue

            try:
                result = self._run_single_configuration(dataset, best_model, prompt_version)
                prompt_results[prompt_version] = result
            except Exception as e:
                logger.error(f"Failed to test prompt {prompt_version}: {e}")
                continue

        # Generate visualizations and reports
        self._generate_visualizations(model_results, prompt_results)
        self._generate_report(model_results, prompt_results, best_model)
        self._analyze_failures(model_results, prompt_results)
        self._save_results(model_results, prompt_results)

        logger.info(f"\nExperiment complete! Results saved to: {self.config.output_dir}")

    def _generate_visualizations(
        self,
        model_results: Dict[str, ExperimentRun],
        prompt_results: Dict[str, ExperimentRun],
    ):
        """Generate all visualizations."""
        self._plot_model_comparison(model_results)
        self._plot_prompt_comparison(prompt_results)
        self._plot_error_analysis(model_results, prompt_results)
        self._plot_per_class_impact(model_results, prompt_results)

    def _plot_model_comparison(self, results: Dict[str, ExperimentRun]):
        """Create model comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        models = list(results.keys())

        # 1. Net Improvement
        net_imps = [results[m].net_improvement for m in models]
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in net_imps]
        axes[0, 0].bar(models, net_imps, color=colors, edgecolor="black", linewidth=0.5)
        axes[0, 0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        axes[0, 0].set_title("Net Improvement (Helpful - Harmful)", fontweight="bold")
        axes[0, 0].set_ylabel("Count")
        for i, v in enumerate(net_imps):
            axes[0, 0].text(i, v + 0.5, str(v), ha="center", fontsize=10)

        # 2. Correction Accuracy
        acc = [results[m].correction_accuracy * 100 for m in models]
        axes[0, 1].bar(models, acc, color="#1f77b4", edgecolor="black", linewidth=0.5)
        axes[0, 1].set_title("Correction Accuracy (%)", fontweight="bold")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_ylim(0, 100)
        for i, v in enumerate(acc):
            axes[0, 1].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

        # 3. Harm Rate
        harm = [results[m].harm_rate * 100 for m in models]
        axes[1, 0].bar(models, harm, color="#d62728", edgecolor="black", linewidth=0.5)
        axes[1, 0].set_title("Harm Rate (%)", fontweight="bold")
        axes[1, 0].set_ylabel("Rate (%)")
        axes[1, 0].set_ylim(0, max(harm) * 1.3 if harm else 50)
        for i, v in enumerate(harm):
            axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

        # 4. Cost per Improvement
        costs = [self._estimate_cost(results[m]) for m in models]
        cost_per_imp = [c / max(results[m].net_improvement, 1) for m, c in zip(models, costs)]
        axes[1, 1].bar(models, cost_per_imp, color="#ff7f0e", edgecolor="black", linewidth=0.5)
        axes[1, 1].set_title("Est. Cost per Net Improvement ($)", fontweight="bold")
        axes[1, 1].set_ylabel("$ per improvement")
        for i, v in enumerate(cost_per_imp):
            axes[1, 1].text(i, v + 0.01, f"${v:.3f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "model_comparison.png"), dpi=150)
        plt.close()
        logger.info("Saved: model_comparison.png")

    def _plot_prompt_comparison(self, results: Dict[str, ExperimentRun]):
        """Create prompt comparison visualization."""
        if len(results) < 2:
            logger.warning("Not enough prompt results for comparison plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        prompts = list(results.keys())

        # Same structure as model comparison
        net_imps = [results[p].net_improvement for p in prompts]
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in net_imps]
        axes[0, 0].bar(prompts, net_imps, color=colors, edgecolor="black", linewidth=0.5)
        axes[0, 0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        axes[0, 0].set_title("Net Improvement (Helpful - Harmful)", fontweight="bold")
        axes[0, 0].set_ylabel("Count")
        for i, v in enumerate(net_imps):
            axes[0, 0].text(i, v + 0.5, str(v), ha="center", fontsize=10)

        acc = [results[p].correction_accuracy * 100 for p in prompts]
        axes[0, 1].bar(prompts, acc, color="#1f77b4", edgecolor="black", linewidth=0.5)
        axes[0, 1].set_title("Correction Accuracy (%)", fontweight="bold")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_ylim(0, 100)
        for i, v in enumerate(acc):
            axes[0, 1].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

        harm = [results[p].harm_rate * 100 for p in prompts]
        axes[1, 0].bar(prompts, harm, color="#d62728", edgecolor="black", linewidth=0.5)
        axes[1, 0].set_title("Harm Rate (%)", fontweight="bold")
        axes[1, 0].set_ylabel("Rate (%)")
        axes[1, 0].set_ylim(0, max(harm) * 1.3 if harm else 50)
        for i, v in enumerate(harm):
            axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

        # Avg latency
        latency = [results[p].avg_latency_ms for p in prompts]
        axes[1, 1].bar(prompts, latency, color="#9467bd", edgecolor="black", linewidth=0.5)
        axes[1, 1].set_title("Avg Latency (ms)", fontweight="bold")
        axes[1, 1].set_ylabel("Milliseconds")
        for i, v in enumerate(latency):
            axes[1, 1].text(i, v + 20, f"{v:.0f}", ha="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "prompt_comparison.png"), dpi=150)
        plt.close()
        logger.info("Saved: prompt_comparison.png")

    def _plot_error_analysis(
        self,
        model_results: Dict[str, ExperimentRun],
        prompt_results: Dict[str, ExperimentRun],
    ):
        """Analyze and visualize why LLM fails."""
        # Collect all harmful corrections
        all_harmful = []
        all_helpful = []

        for run in list(model_results.values()) + list(prompt_results.values()):
            all_harmful.extend([t for t in run.token_results if t.was_harmful])
            all_helpful.extend([t for t in run.token_results if t.was_helpful])

        if not all_harmful and not all_helpful:
            logger.warning("No interventions to analyze")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Over vs Under classification
        over_class = sum(1 for t in all_harmful if t.model_prediction == "O" and t.llm_correction != "O")
        under_class = sum(1 for t in all_harmful if t.model_prediction != "O" and t.llm_correction == "O")
        wrong_type = sum(1 for t in all_harmful if t.model_prediction != "O" and t.llm_correction != "O"
                        and t.model_prediction != t.llm_correction)

        categories = ["Over-classification\n(O→PII)", "Under-classification\n(PII→O)", "Wrong Entity Type"]
        values = [over_class, under_class, wrong_type]
        colors = ["#ff7f0e", "#1f77b4", "#9467bd"]

        axes[0, 0].bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
        axes[0, 0].set_title("Harmful Correction Types", fontweight="bold")
        axes[0, 0].set_ylabel("Count")
        for i, v in enumerate(values):
            axes[0, 0].text(i, v + 0.5, str(v), ha="center", fontsize=10)

        # 2. Most harmed entity classes
        harmed_classes = Counter(t.ground_truth for t in all_harmful)
        top_harmed = harmed_classes.most_common(8)
        if top_harmed:
            classes, counts = zip(*top_harmed)
            axes[0, 1].barh(classes, counts, color="#d62728", edgecolor="black", linewidth=0.5)
            axes[0, 1].set_title("Most Harmed Entity Classes", fontweight="bold")
            axes[0, 1].set_xlabel("Harmful Corrections")
            axes[0, 1].invert_yaxis()

        # 3. Entropy/Confidence scatter for helpful vs harmful
        if all_helpful:
            axes[1, 0].scatter(
                [t.confidence for t in all_helpful],
                [t.entropy for t in all_helpful],
                c="#2ca02c", alpha=0.6, label=f"Helpful ({len(all_helpful)})", s=30
            )
        if all_harmful:
            axes[1, 0].scatter(
                [t.confidence for t in all_harmful],
                [t.entropy for t in all_harmful],
                c="#d62728", alpha=0.6, label=f"Harmful ({len(all_harmful)})", s=30
            )
        axes[1, 0].set_xlabel("Confidence")
        axes[1, 0].set_ylabel("Entropy")
        axes[1, 0].set_title("Intervention Outcomes by Uncertainty", fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].axvline(x=self.config.confidence_threshold, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].axhline(y=self.config.entropy_threshold, color="gray", linestyle="--", alpha=0.5)

        # 4. Common wrong corrections
        wrong_corrections = Counter(
            f"{t.ground_truth}→{t.llm_correction}" for t in all_harmful
        )
        top_wrong = wrong_corrections.most_common(8)
        if top_wrong:
            patterns, counts = zip(*top_wrong)
            axes[1, 1].barh(patterns, counts, color="#d62728", edgecolor="black", linewidth=0.5)
            axes[1, 1].set_title("Most Common Wrong Corrections", fontweight="bold")
            axes[1, 1].set_xlabel("Count")
            axes[1, 1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "error_analysis.png"), dpi=150)
        plt.close()
        logger.info("Saved: error_analysis.png")

    def _plot_per_class_impact(
        self,
        model_results: Dict[str, ExperimentRun],
        prompt_results: Dict[str, ExperimentRun],
    ):
        """Plot per-class LLM impact."""
        # Use the best result for per-class analysis
        all_results = {**model_results, **prompt_results}
        if not all_results:
            return

        best_key = max(all_results.keys(), key=lambda k: all_results[k].net_improvement)
        result = all_results[best_key]

        # Calculate net impact per class
        class_impact = {}
        for label, stats in result.per_class_stats.items():
            if stats["total"] >= 3:  # Filter out rare classes
                class_impact[label] = stats["helpful"] - stats["harmful"]

        if not class_impact:
            return

        # Sort by impact
        sorted_items = sorted(class_impact.items(), key=lambda x: x[1])
        labels, impacts = zip(*sorted_items)

        fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))

        colors = ["#2ca02c" if x > 0 else "#d62728" if x < 0 else "#7f7f7f" for x in impacts]
        bars = ax.barh(labels, impacts, color=colors, edgecolor="black", linewidth=0.5)

        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Net Impact (Helpful - Harmful)")
        ax.set_title(f"Per-Class LLM Impact ({best_key})", fontweight="bold")

        # Add value labels
        for bar, impact in zip(bars, impacts):
            width = bar.get_width()
            offset = 0.3 if width >= 0 else -0.3
            ax.text(width + offset, bar.get_y() + bar.get_height()/2,
                   f"{impact:+d}", va="center", ha="left" if width >= 0 else "right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "per_class_impact.png"), dpi=150)
        plt.close()
        logger.info("Saved: per_class_impact.png")

    def _generate_report(
        self,
        model_results: Dict[str, ExperimentRun],
        prompt_results: Dict[str, ExperimentRun],
        best_model: str,
    ):
        """Generate text report with recommendations."""
        lines = []
        lines.append("=" * 80)
        lines.append("LLM ROUTER EXPERIMENT REPORT")
        lines.append("=" * 80)
        lines.append(f"\nSamples tested: {self.config.n_samples}")
        lines.append(f"Entropy threshold: {self.config.entropy_threshold}")
        lines.append(f"Confidence threshold: {self.config.confidence_threshold}")

        # Phase 1 Summary
        lines.append("\n" + "=" * 80)
        lines.append("PHASE 1: MODEL COMPARISON (V11 Prompt)")
        lines.append("=" * 80)

        for model, result in sorted(model_results.items(), key=lambda x: -x[1].net_improvement):
            cost = self._estimate_cost(result)
            lines.append(f"\n{model}:")
            lines.append(f"  LLM Calls: {result.total_llm_calls}")
            lines.append(f"  Helpful: {result.helpful_corrections}")
            lines.append(f"  Harmful: {result.harmful_corrections}")
            lines.append(f"  Net Improvement: {result.net_improvement}")
            lines.append(f"  Correction Accuracy: {result.correction_accuracy:.2%}")
            lines.append(f"  Harm Rate: {result.harm_rate:.2%}")
            lines.append(f"  Avg Latency: {result.avg_latency_ms:.1f}ms")
            lines.append(f"  Est. Cost: ${cost:.4f}")

        lines.append(f"\n>>> BEST MODEL: {best_model}")

        # Phase 2 Summary
        lines.append("\n" + "=" * 80)
        lines.append(f"PHASE 2: PROMPT COMPARISON ({best_model})")
        lines.append("=" * 80)

        best_prompt = None
        best_net = float("-inf")

        for prompt, result in sorted(prompt_results.items(), key=lambda x: -x[1].net_improvement):
            lines.append(f"\n{prompt}:")
            lines.append(f"  LLM Calls: {result.total_llm_calls}")
            lines.append(f"  Helpful: {result.helpful_corrections}")
            lines.append(f"  Harmful: {result.harmful_corrections}")
            lines.append(f"  Net Improvement: {result.net_improvement}")
            lines.append(f"  Correction Accuracy: {result.correction_accuracy:.2%}")
            lines.append(f"  Harm Rate: {result.harm_rate:.2%}")

            if result.net_improvement > best_net:
                best_net = result.net_improvement
                best_prompt = prompt

        lines.append(f"\n>>> BEST PROMPT: {best_prompt}")

        # Recommendations
        lines.append("\n" + "=" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append(f"\nOptimal Configuration: {best_model} + {best_prompt}")

        if best_net <= 0:
            lines.append("\nWARNING: LLM routing is not providing net benefit!")
            lines.append("Consider:")
            lines.append("  1. Adjusting entropy/confidence thresholds")
            lines.append("  2. Filtering out certain entity classes from LLM routing")
            lines.append("  3. Improving the prompt with more specific instructions")

        report_text = "\n".join(lines)

        with open(os.path.join(self.config.output_dir, "experiment_report.txt"), "w") as f:
            f.write(report_text)

        logger.info("Saved: experiment_report.txt")
        print("\n" + report_text)

    def _analyze_failures(
        self,
        model_results: Dict[str, ExperimentRun],
        prompt_results: Dict[str, ExperimentRun],
    ):
        """Detailed analysis of failure patterns."""
        all_harmful = []
        for run in list(model_results.values()) + list(prompt_results.values()):
            all_harmful.extend([t for t in run.token_results if t.was_harmful])

        if not all_harmful:
            return

        lines = []
        lines.append("=" * 80)
        lines.append("FAILURE ANALYSIS")
        lines.append("=" * 80)

        lines.append(f"\nTotal harmful corrections analyzed: {len(all_harmful)}")

        # Pattern analysis
        over_class = sum(1 for t in all_harmful if t.model_prediction == "O")
        under_class = sum(1 for t in all_harmful if t.llm_correction == "O")

        lines.append(f"\nFailure Patterns:")
        lines.append(f"  Over-classification (O→PII): {over_class} ({over_class/len(all_harmful)*100:.1f}%)")
        lines.append(f"  Under-classification (PII→O): {under_class} ({under_class/len(all_harmful)*100:.1f}%)")
        lines.append(f"  Wrong entity type: {len(all_harmful) - over_class - under_class}")

        # Most affected classes
        harmed_classes = Counter(t.ground_truth for t in all_harmful)
        lines.append(f"\nMost Affected Classes:")
        for cls, count in harmed_classes.most_common(5):
            lines.append(f"  {cls}: {count}")

        # Sample harmful cases
        lines.append(f"\nSample Harmful Cases:")
        for t in all_harmful[:10]:
            lines.append(f"\n  Token: '{t.token_text}'")
            lines.append(f"  Ground Truth: {t.ground_truth}")
            lines.append(f"  Model Pred: {t.model_prediction} (correct: {t.model_was_correct})")
            lines.append(f"  LLM Correction: {t.llm_correction}")
            lines.append(f"  Reasoning: {t.reasoning[:100]}...")

        with open(os.path.join(self.config.output_dir, "failure_analysis.txt"), "w") as f:
            f.write("\n".join(lines))

        logger.info("Saved: failure_analysis.txt")

    def _save_results(
        self,
        model_results: Dict[str, ExperimentRun],
        prompt_results: Dict[str, ExperimentRun],
    ):
        """Save results to JSON."""
        def run_to_dict(run: ExperimentRun) -> Dict:
            return {
                "model_name": run.model_name,
                "prompt_version": run.prompt_version,
                "total_llm_calls": run.total_llm_calls,
                "helpful_corrections": run.helpful_corrections,
                "harmful_corrections": run.harmful_corrections,
                "neutral_changes": run.neutral_changes,
                "net_improvement": run.net_improvement,
                "correction_accuracy": run.correction_accuracy,
                "harm_rate": run.harm_rate,
                "avg_latency_ms": run.avg_latency_ms,
                "estimated_cost_usd": self._estimate_cost(run),
                "per_class_stats": dict(run.per_class_stats),
            }

        results = {
            "config": {
                "n_samples": self.config.n_samples,
                "entropy_threshold": self.config.entropy_threshold,
                "confidence_threshold": self.config.confidence_threshold,
            },
            "model_comparison": {k: run_to_dict(v) for k, v in model_results.items()},
            "prompt_comparison": {k: run_to_dict(v) for k, v in prompt_results.items()},
        }

        with open(os.path.join(self.config.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Saved: results.json")


def main():
    parser = argparse.ArgumentParser(description="LLM Router Experiment")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--models", nargs="+",
                       default=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
                       help="OpenAI models to test")
    parser.add_argument("--prompts", nargs="+", default=["V11", "V9", "V_EXP"],
                       help="Prompt versions to test")
    parser.add_argument("--output-dir", default="./experiments/llm_router",
                       help="Output directory for results")

    args = parser.parse_args()

    config = ExperimentConfig(
        n_samples=args.n_samples,
        models_to_test=args.models,
        prompts_to_test=args.prompts,
        output_dir=args.output_dir,
    )

    experiment = LLMRouterExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
