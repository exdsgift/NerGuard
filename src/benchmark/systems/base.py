"""Base classes for system wrappers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SystemPrediction:
    """Prediction for a single sample."""

    labels: List[str]  # BIO labels using system's NATIVE label names
    latency_ms: float
    metadata: Optional[Dict] = None  # Optional routing/system metadata


@dataclass
class DeferredPrediction:
    """NER-only prediction with pending LLM routing requests.

    Used in batch mode: Pass 1 produces these, Pass 2 resolves LLM calls,
    Pass 3 generates final SystemPrediction.
    """

    sample_idx: int
    text: str
    tokens: List[str]
    token_spans: List[Tuple[int, int]]
    subword_preds: List[str]
    offset_mapping: list
    pending_spans: list  # EntitySpan objects needing LLM routing
    ner_latency_ms: float
    conf_vals: Optional[List[float]] = None  # per-subword confidence values


class SystemWrapper(ABC):
    """Abstract base class for NER system wrappers."""

    @abstractmethod
    def name(self) -> str:
        """Return the system name (e.g., 'NerGuard Base')."""
        ...

    @abstractmethod
    def native_labels(self) -> Set[str]:
        """Return set of entity class labels this system produces, WITHOUT BIO prefix."""
        ...

    @abstractmethod
    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        """Run inference on a single sample.

        Args:
            text: Raw input text.
            tokens: Word-level tokens.
            token_spans: Character (start, end) offsets per token.

        Returns:
            SystemPrediction with BIO labels aligned to tokens.
        """
        ...

    def setup(self) -> None:
        """One-time setup: load model, allocate resources."""
        pass

    def teardown(self) -> None:
        """Cleanup: unload model, free memory."""
        pass

    def calibrate_thresholds(self, samples: list) -> None:
        """Calibrate routing thresholds on held-out samples.

        Must be called after setup(). Only meaningful for hybrid systems
        that use uncertainty-based routing. Default is no-op.
        """
        pass

    def describe(self) -> Dict:
        """Return system metadata for documentation."""
        return {
            "name": self.name(),
            "native_labels": sorted(self.native_labels()),
            "n_labels": len(self.native_labels()),
        }
