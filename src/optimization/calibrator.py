"""Task-agnostic threshold calibrator for NerGuard routing.

Given a loaded NER model and calibration samples with gold labels,
finds optimal entropy/confidence thresholds via grid search over
the routing decision boundary. The calibrator maximises the F-beta
score of error detection: "does the uncertainty flag correctly
identify tokens where the base model is wrong?"

This is model- and task-agnostic: it works with any HuggingFace
token-classification model and any BIO-labelled calibration set.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.benchmark.datasets.base import BenchmarkSample
from src.core.route_config import RouteConfig

logger = logging.getLogger(__name__)

MAX_LENGTH = 512


@dataclass
class CalibrationResult:
    entropy_threshold: float
    confidence_threshold: float
    routing_precision: float
    routing_recall: float
    routing_f_score: float
    intervention_rate: float
    n_tokens: int
    n_errors: int

    def to_dict(self) -> Dict:
        return asdict(self)


class ThresholdCalibrator:
    """Find optimal routing thresholds for any NER model + dataset.

    The calibrator runs base-model inference on a held-out calibration
    set, collects per-word entropy and confidence, then grid-searches
    for the (entropy_threshold, confidence_threshold) pair that best
    identifies model errors (tokens where pred != gold).

    Usage::

        calibrator = ThresholdCalibrator()
        result, config = calibrator.calibrate(
            model=model, tokenizer=tokenizer, id2label=id2label,
            device=device, samples=cal_samples,
            base_config=get_biomedical_route_config(),
        )
        # config has optimised thresholds, ready for EntitySpecificRouter
    """

    def __init__(self, grid_resolution: int = 25, beta: float = 0.5):
        """
        Args:
            grid_resolution: Number of points per axis in the grid search.
            beta: Beta parameter for F-beta score. beta < 1 favours
                precision (fewer false routing calls), beta > 1 favours
                recall (catch more errors). Default 0.5 balances toward
                precision to minimise unnecessary LLM calls.
        """
        self.grid_resolution = grid_resolution
        self.beta = beta

    def calibrate(
        self,
        model: torch.nn.Module,
        tokenizer,
        id2label: Dict[int, str],
        device: torch.device,
        samples: List[BenchmarkSample],
        base_config: Optional[RouteConfig] = None,
    ) -> Tuple[CalibrationResult, RouteConfig]:
        """Run calibration and return optimised RouteConfig.

        Args:
            model: Loaded HuggingFace token-classification model (eval mode).
            tokenizer: Matching tokenizer.
            id2label: Model's id-to-label mapping.
            device: Torch device the model is on.
            samples: Calibration samples with gold BIO labels.
            base_config: Base RouteConfig whose non-threshold fields
                (routable_entities, blocked_entities, etc.) are preserved.
                If None, creates a permissive default config.

        Returns:
            Tuple of (CalibrationResult with stats, optimised RouteConfig).
        """
        entropies, confidences, is_errors = self._collect_stats(
            model, tokenizer, id2label, device, samples
        )

        logger.info(
            f"Calibration: {len(entropies)} tokens, "
            f"{np.sum(is_errors)} errors ({np.mean(is_errors):.2%}), "
            f"entropy \u03bc={np.mean(entropies):.3f} \u03c3={np.std(entropies):.3f}, "
            f"confidence \u03bc={np.mean(confidences):.3f} \u03c3={np.std(confidences):.3f}"
        )

        result = self._grid_search(entropies, confidences, is_errors)

        logger.info(
            f"Optimal thresholds: entropy>{result.entropy_threshold:.3f}, "
            f"confidence<{result.confidence_threshold:.3f} "
            f"(F{self.beta}={result.routing_f_score:.3f}, "
            f"intervention={result.intervention_rate:.1%})"
        )

        config = self._build_route_config(result, base_config)
        return result, config

    def _collect_stats(
        self,
        model: torch.nn.Module,
        tokenizer,
        id2label: Dict[int, str],
        device: torch.device,
        samples: List[BenchmarkSample],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and collect per-word entropy, confidence, is_error."""
        all_entropies = []
        all_confidences = []
        all_is_error = []

        for sample in samples:
            encoding = tokenizer(
                sample.text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            offset_mapping = encoding["offset_mapping"][0].tolist()

            with torch.no_grad():
                logits = model(input_ids, attention_mask=attention_mask).logits[0]

            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
            conf, pred_ids = torch.max(probs, dim=-1)

            pred_ids = pred_ids.cpu().tolist()
            entropy_vals = entropy.cpu().tolist()
            conf_vals = conf.cpu().tolist()

            subword_preds = [id2label.get(pid, "O") for pid in pred_ids]

            for word_idx, (w_start, w_end) in enumerate(sample.token_spans):
                if word_idx >= len(sample.labels):
                    break
                gold = sample.labels[word_idx]

                # Find first overlapping subword for this word
                for sw_idx, (sw_start, sw_end) in enumerate(offset_mapping):
                    if sw_start == sw_end == 0:
                        continue
                    if sw_end <= w_start or sw_start >= w_end:
                        continue

                    pred = subword_preds[sw_idx]

                    # Compare at entity level (ignore B-/I- prefix differences)
                    pred_ent = pred.split("-", 1)[1] if "-" in pred else pred
                    gold_ent = gold.split("-", 1)[1] if "-" in gold else gold

                    all_entropies.append(entropy_vals[sw_idx])
                    all_confidences.append(conf_vals[sw_idx])
                    all_is_error.append(pred_ent != gold_ent)
                    break

        return (
            np.array(all_entropies),
            np.array(all_confidences),
            np.array(all_is_error, dtype=bool),
        )

    def _grid_search(
        self,
        entropies: np.ndarray,
        confidences: np.ndarray,
        is_errors: np.ndarray,
    ) -> CalibrationResult:
        """Grid search over entropy/confidence thresholds."""
        n_total = len(is_errors)
        n_errors = int(np.sum(is_errors))

        if n_errors == 0:
            logger.warning("No errors in calibration set; using default thresholds")
            return CalibrationResult(
                entropy_threshold=0.5,
                confidence_threshold=0.8,
                routing_precision=0.0,
                routing_recall=0.0,
                routing_f_score=0.0,
                intervention_rate=0.0,
                n_tokens=n_total,
                n_errors=0,
            )

        # Adaptive grid ranges based on data distribution
        ent_lo = max(0.01, float(np.percentile(entropies, 40)))
        ent_hi = min(4.0, float(np.percentile(entropies, 99.5)))
        conf_lo = max(0.3, float(np.percentile(confidences, 0.5)))
        conf_hi = min(0.999, float(np.percentile(confidences, 60)))

        ent_grid = np.linspace(ent_lo, ent_hi, self.grid_resolution)
        conf_grid = np.linspace(conf_lo, conf_hi, self.grid_resolution)

        best_score = -1.0
        best_result = None
        beta_sq = self.beta ** 2

        for ent_th in ent_grid:
            for conf_th in conf_grid:
                trigger = (entropies > ent_th) & (confidences < conf_th)
                n_triggers = np.sum(trigger)

                if n_triggers == 0:
                    continue

                tp = np.sum(trigger & is_errors)
                precision = float(tp / n_triggers)
                recall = float(tp / n_errors)

                if precision + recall == 0:
                    continue

                f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

                if f_score > best_score:
                    best_score = f_score
                    best_result = CalibrationResult(
                        entropy_threshold=round(float(ent_th), 4),
                        confidence_threshold=round(float(conf_th), 4),
                        routing_precision=round(precision, 4),
                        routing_recall=round(recall, 4),
                        routing_f_score=round(float(f_score), 4),
                        intervention_rate=round(float(n_triggers / n_total), 4),
                        n_tokens=n_total,
                        n_errors=n_errors,
                    )

        if best_result is None:
            logger.warning("Grid search found no valid threshold; using defaults")
            return CalibrationResult(
                entropy_threshold=0.5,
                confidence_threshold=0.8,
                routing_precision=0.0,
                routing_recall=0.0,
                routing_f_score=0.0,
                intervention_rate=0.0,
                n_tokens=n_total,
                n_errors=n_errors,
            )

        return best_result

    @staticmethod
    def _build_route_config(
        result: CalibrationResult,
        base_config: Optional[RouteConfig],
    ) -> RouteConfig:
        """Create a RouteConfig with calibrated thresholds."""
        calibrated_thresholds = {
            "DEFAULT": {
                "entropy": result.entropy_threshold,
                "confidence": result.confidence_threshold,
            }
        }

        if base_config is not None:
            return RouteConfig(
                entropy_threshold=result.entropy_threshold,
                confidence_threshold=result.confidence_threshold,
                routable_entities=base_config.routable_entities,
                blocked_entities=base_config.blocked_entities,
                routable_i_entities=base_config.routable_i_entities,
                entity_thresholds=calibrated_thresholds,
                enable_selective=base_config.enable_selective,
                block_continuation_tokens=base_config.block_continuation_tokens,
                o_entropy_multiplier=base_config.o_entropy_multiplier,
            )

        return RouteConfig(
            entropy_threshold=result.entropy_threshold,
            confidence_threshold=result.confidence_threshold,
            entity_thresholds=calibrated_thresholds,
        )
