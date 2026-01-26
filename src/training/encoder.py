"""
PII Encoder - Model factory for NerGuard.

This module provides the PIIEncoder class for initializing and configuring
the mDeBERTa-based token classification model.
"""

import logging
from typing import Dict

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.core.constants import DEFAULT_BASE_MODEL

logger = logging.getLogger(__name__)


class PIIEncoder:
    """
    Factory class for creating and configuring PII detection models.

    This class handles:
    - Loading the base mDeBERTa tokenizer
    - Initializing the token classification model with custom labels
    - Optional backbone freezing for transfer learning

    Example:
        >>> encoder = PIIEncoder()
        >>> tokenizer = encoder.get_tokenizer()
        >>> model = encoder.get_model(
        ...     num_labels=43,
        ...     id2label=id2label,
        ...     label2id=label2id,
        ... )
    """

    def __init__(self, model_name: str = DEFAULT_BASE_MODEL):
        """
        Initialize the encoder factory.

        Args:
            model_name: HuggingFace model name or path (default: microsoft/mdeberta-v3-base)
        """
        self.model_name = model_name
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer.

        We use the fast tokenizer with add_prefix_space for better subword handling.

        Returns:
            Loaded tokenizer instance

        Raises:
            OSError: If tokenizer cannot be loaded
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                add_prefix_space=True,
            )
            logger.debug(f"Loaded tokenizer from {self.model_name}")
            return tokenizer
        except OSError as e:
            raise OSError(
                f"Error loading tokenizer for {self.model_name}. "
                f"Make sure the model exists and dependencies are installed.\n{e}"
            )

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer instance."""
        return self.tokenizer

    def get_model(
        self,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
    ) -> PreTrainedModel:
        """
        Initialize and return a configured token classification model.

        Args:
            num_labels: Total number of PII classes
            id2label: Mapping from label ID to label name
            label2id: Mapping from label name to label ID
            dropout_rate: Dropout probability for the classification head
            freeze_backbone: If True, freeze encoder weights (only head is trainable)

        Returns:
            Configured AutoModelForTokenClassification instance
        """
        # Create model configuration
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )

        # Load pretrained model with custom config
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            config=config,
            ignore_mismatched_sizes=True,  # Useful when reloading checkpoints with different heads
        )

        if freeze_backbone:
            self._freeze_layers(model)

        logger.info(f"Initialized model from {self.model_name} with {num_labels} labels")
        return model

    def _freeze_layers(self, model: PreTrainedModel) -> None:
        """
        Freeze the encoder backbone, leaving only the classification head trainable.

        This is useful for transfer learning when you want to adapt the model
        to a new domain without modifying the pretrained representations.

        Args:
            model: The model to freeze
        """
        # mDeBERTa uses 'deberta' as the encoder attribute
        if hasattr(model, "deberta"):
            for param in model.deberta.parameters():
                param.requires_grad = False
            logger.info(f"Backbone {self.model_name} frozen. Only classification head will be trained.")
        else:
            logger.warning("Unable to find 'deberta' module for freezing. Model may have different structure.")


# Convenience function for quick model loading
def create_pii_model(
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    model_name: str = DEFAULT_BASE_MODEL,
    dropout_rate: float = 0.1,
) -> tuple:
    """
    Create a PII model and tokenizer in one call.

    Args:
        num_labels: Number of labels
        id2label: ID to label mapping
        label2id: Label to ID mapping
        model_name: Base model name
        dropout_rate: Dropout rate

    Returns:
        Tuple of (model, tokenizer)
    """
    encoder = PIIEncoder(model_name)
    model = encoder.get_model(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        dropout_rate=dropout_rate,
    )
    return model, encoder.get_tokenizer()
