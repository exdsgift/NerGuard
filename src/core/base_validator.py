"""
Abstract validation strategy for NerGuard.

Validators provide domain-specific structural checks that complement
the uncertainty-based routing. For PII this means regex patterns with
checksums (Luhn, IBAN); for biomedical NER it could be knowledge-base
lookups; for other tasks, a simple passthrough.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PreScanMatch:
    """A validated match found during pre-scan."""

    char_start: int
    char_end: int
    entity_class: str
    matched_text: str


class ValidationStrategy(ABC):
    """Abstract base for domain-specific entity validators.

    Subclasses implement three optional capabilities:
    1. can_skip_llm: Confirm a prediction structurally (skip LLM call).
    2. correct_predictions: Post-process token predictions (promote/override).
    3. validate_predictions: Precision filter (demote unconfirmed predictions).
    4. find_hints: Pre-scan text for structural patterns.
    """

    @abstractmethod
    def can_skip_llm(
        self,
        entity_class: str,
        text: str,
        char_start: int,
        char_end: int,
    ) -> bool:
        """Return True if structural validation confirms the entity, skipping LLM."""
        ...

    @abstractmethod
    def correct_predictions(
        self,
        text: str,
        offset_mapping: np.ndarray,
        preds: np.ndarray,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        correct_partial: bool = False,
    ) -> np.ndarray:
        """Post-process predictions: promote O tokens where structural match found."""
        ...

    @abstractmethod
    def validate_predictions(
        self,
        text: str,
        offset_mapping: np.ndarray,
        preds: np.ndarray,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        entities_to_validate: Optional[set] = None,
    ) -> np.ndarray:
        """Precision filter: demote predictions that fail structural validation."""
        ...

    def find_hints(self, text: str) -> List[PreScanMatch]:
        """Pre-scan text for structural patterns (optional)."""
        return []

    def get_stats(self) -> Dict[str, int]:
        """Return validation statistics."""
        return {}

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        pass


class PassthroughValidator(ValidationStrategy):
    """No-op validator for tasks without structural validation rules.

    All methods return the input unchanged. Use this when no domain-specific
    regex or structural checks are available (e.g., standard NER).
    """

    def can_skip_llm(self, entity_class, text, char_start, char_end):
        return False

    def correct_predictions(self, text, offset_mapping, preds, id2label, label2id, correct_partial=False):
        return preds

    def validate_predictions(self, text, offset_mapping, preds, id2label, label2id, entities_to_validate=None):
        return preds
