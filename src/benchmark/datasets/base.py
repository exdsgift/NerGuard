"""Base classes for dataset adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class BenchmarkSample:
    """A single benchmark sample with word-level tokenization and native labels."""

    text: str
    tokens: List[str]
    token_spans: List[Tuple[int, int]]  # (char_start, char_end) per token
    labels: List[str]  # BIO labels using dataset's NATIVE label names
    sample_id: str
    language: str = "en"
    metadata: Dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @abstractmethod
    def name(self) -> str:
        """Return the canonical dataset name (e.g., 'ai4privacy')."""
        ...

    @abstractmethod
    def native_labels(self) -> Set[str]:
        """Return set of native entity class labels WITHOUT BIO prefix."""
        ...

    @abstractmethod
    def load(
        self,
        max_samples: int = 0,
        seed: int = 42,
        **kwargs,
    ) -> List[BenchmarkSample]:
        """Load and return benchmark samples with native labels.

        Args:
            max_samples: Maximum samples to load (0 = all).
            seed: Random seed for reproducible sampling.

        Returns:
            List of BenchmarkSample with native dataset labels.
        """
        ...

    def describe(self) -> Dict:
        """Return dataset metadata for documentation."""
        return {
            "name": self.name(),
            "native_labels": sorted(self.native_labels()),
            "n_labels": len(self.native_labels()),
        }
