"""
Label mapping utilities for NerGuard.

This module provides the LabelMapper class for converting between different
label schemas (e.g., NVIDIA PII dataset labels to NerGuard model labels).
"""

from typing import Dict, Optional
from collections import defaultdict

from src.core.constants import NVIDIA_TO_MODEL_MAP


class LabelMapper:
    """
    Handles mapping between external dataset labels and model-specific labels.

    This class is used primarily for evaluation on external datasets that use
    different label schemas than the NerGuard model.

    Example:
        >>> mapper = LabelMapper(model.config.id2label)
        >>> label_id = mapper.get_token_label_id("first_name", is_start=True)
        >>> # Returns the ID for "B-GIVENNAME"
    """

    def __init__(
        self,
        model_id2label: Dict[int, str],
        external_to_model_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the LabelMapper.

        Args:
            model_id2label: Mapping from model label IDs to label names
            external_to_model_map: Optional custom mapping from external labels
                                   to model base labels. Defaults to NVIDIA_TO_MODEL_MAP.
        """
        self.model_id2label = model_id2label
        self.model_label2id = {v: k for k, v in model_id2label.items()}
        self.external_map = external_to_model_map or NVIDIA_TO_MODEL_MAP

        # Statistics tracking
        self.map_counter: Dict[str, int] = defaultdict(int)
        self.unmapped_labels: Dict[str, int] = defaultdict(int)

    def get_token_label_id(self, external_label: str, is_start: bool) -> int:
        """
        Convert an external label to a model label ID.

        Args:
            external_label: Label from the external dataset
            is_start: Whether this is the first token of the entity (B- prefix)

        Returns:
            Model label ID corresponding to the external label
        """
        # Get the base model label (without B-/I- prefix)
        target_base = self.external_map.get(external_label, "O")

        # Track mapping statistics
        self.map_counter[external_label] += 1

        # Handle "O" (Outside) labels
        if target_base == "O":
            return self.model_label2id.get("O", 0)

        # Add BIO prefix
        prefix = "B-" if is_start else "I-"
        full_label = f"{prefix}{target_base}"

        # Get the ID, fallback to "O" if label not in model
        label_id = self.model_label2id.get(full_label)
        if label_id is None:
            self.unmapped_labels[full_label] += 1
            return self.model_label2id.get("O", 0)

        return label_id

    def get_model_label(self, external_label: str, is_start: bool) -> str:
        """
        Convert an external label to a model label string.

        Args:
            external_label: Label from the external dataset
            is_start: Whether this is the first token of the entity

        Returns:
            Model label string (e.g., "B-GIVENNAME")
        """
        label_id = self.get_token_label_id(external_label, is_start)
        return self.model_id2label.get(label_id, "O")

    def get_statistics(self) -> Dict:
        """
        Get mapping statistics.

        Returns:
            Dictionary containing:
                - map_counter: Count of each external label mapped
                - unmapped_labels: Labels that couldn't be mapped
                - total_mapped: Total number of labels mapped
        """
        return {
            "map_counter": dict(self.map_counter),
            "unmapped_labels": dict(self.unmapped_labels),
            "total_mapped": sum(self.map_counter.values()),
        }

    def reset_statistics(self):
        """Reset the mapping statistics counters."""
        self.map_counter.clear()
        self.unmapped_labels.clear()

    def __repr__(self) -> str:
        return (
            f"LabelMapper(model_labels={len(self.model_id2label)}, "
            f"external_mappings={len(self.external_map)})"
        )
