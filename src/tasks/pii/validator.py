"""
PII validation strategy — wraps the existing RegexValidator as a ValidationStrategy.

This provides backward compatibility: the existing RegexValidator with its Luhn,
IBAN, SSN checksums and PII-specific patterns is exposed through the generic
ValidationStrategy interface.
"""

from typing import Dict, List, Optional

import numpy as np

from src.core.base_validator import PreScanMatch, ValidationStrategy
from src.inference.regex_validator import RegexValidator


class PIIValidator(ValidationStrategy):
    """PII-specific validator using regex patterns with checksum validation.

    Wraps the existing RegexValidator to conform to the ValidationStrategy interface.
    """

    def __init__(self, regex_validator: Optional[RegexValidator] = None):
        self._validator = regex_validator or RegexValidator()

    def can_skip_llm(self, entity_class, text, char_start, char_end):
        return self._validator.can_skip_llm(entity_class, text, char_start, char_end)

    def correct_predictions(self, text, offset_mapping, preds, id2label, label2id, correct_partial=False):
        return self._validator.correct_predictions(
            text, offset_mapping, preds, id2label, label2id, correct_partial
        )

    def validate_predictions(self, text, offset_mapping, preds, id2label, label2id, entities_to_validate=None):
        return self._validator.validate_predictions(
            text, offset_mapping, preds, id2label, label2id, entities_to_validate
        )

    def find_hints(self, text: str) -> List[PreScanMatch]:
        raw_hints = self._validator.find_regex_hints(text)
        return [
            PreScanMatch(
                char_start=start,
                char_end=end,
                entity_class=entity_class,
                matched_text=text[start:end],
            )
            for start, end, entity_class in raw_hints
        ]

    def get_stats(self):
        return self._validator.get_stats()

    def reset_stats(self):
        self._validator.reset_stats()
