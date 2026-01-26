"""
Metrics computation utilities for NerGuard.

This module provides functions for computing entropy, confidence scores,
and other metrics used in the hybrid inference pipeline.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def compute_entropy_confidence(
    logits: torch.Tensor,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute entropy, confidence scores, and predictions from logits.

    This function calculates Shannon entropy as a measure of uncertainty
    and confidence as the maximum probability.

    Args:
        logits: Raw model output logits of shape (..., num_classes)
        dim: Dimension along which to compute softmax (default: -1)

    Returns:
        Tuple of (entropy, confidence, predictions):
            - entropy: Shannon entropy for each position
            - confidence: Maximum probability for each position
            - predictions: Predicted class IDs

    Example:
        >>> logits = model(input_ids).logits  # Shape: (batch, seq_len, num_classes)
        >>> entropy, confidence, preds = compute_entropy_confidence(logits)
    """
    # Compute probabilities
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=dim)

    # Confidence and predictions
    confidence, predictions = torch.max(probs, dim=dim)

    return entropy, confidence, predictions


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Shannon entropy from logits.

    Args:
        logits: Raw model output logits
        dim: Dimension along which to compute

    Returns:
        Entropy values for each position
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    return -torch.sum(probs * log_probs, dim=dim)


def compute_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute confidence (max probability) from logits.

    Args:
        logits: Raw model output logits
        dim: Dimension along which to compute

    Returns:
        Confidence values for each position
    """
    probs = F.softmax(logits, dim=dim)
    return torch.max(probs, dim=dim)[0]


def should_trigger_llm(
    entropy: float,
    confidence: float,
    entropy_threshold: float,
    confidence_threshold: float,
) -> bool:
    """
    Determine if LLM disambiguation should be triggered based on uncertainty.

    The LLM is called when the model is uncertain, which is indicated by:
    - High entropy (> threshold): Multiple classes have similar probabilities
    - Low confidence (< threshold): The top prediction is not confident

    Args:
        entropy: Entropy value for the token
        confidence: Confidence value for the token
        entropy_threshold: Minimum entropy to trigger LLM
        confidence_threshold: Maximum confidence to trigger LLM

    Returns:
        True if LLM should be called, False otherwise
    """
    return entropy > entropy_threshold and confidence < confidence_threshold


def normalize_entropy(
    entropy: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Normalize entropy to [0, 1] range.

    Maximum entropy for a uniform distribution over n classes is log(n).

    Args:
        entropy: Raw entropy values
        num_classes: Number of classes in the classification

    Returns:
        Normalized entropy in [0, 1]
    """
    import math
    max_entropy = math.log(num_classes)
    return entropy / max_entropy
