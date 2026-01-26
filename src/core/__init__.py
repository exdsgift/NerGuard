"""
Core module - Shared utilities and constants for NerGuard.

This module contains:
- constants: Configuration values, label mappings, thresholds
- model_loader: Unified model/tokenizer loading functions
- label_mapper: Label mapping between different datasets
- metrics: Entropy, confidence, and evaluation metrics computation
"""

from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_CONTEXT_LENGTH,
    OVERLAP,
    STRIDE,
    VALID_LABELS,
    NVIDIA_TO_MODEL_MAP,
)
from src.core.model_loader import (
    load_model,
    load_tokenizer,
    load_model_and_tokenizer,
    load_labels,
)
from src.core.label_mapper import LabelMapper
from src.core.metrics import compute_entropy_confidence

__all__ = [
    # Constants
    "DEFAULT_MODEL_PATH",
    "DEFAULT_DATA_PATH",
    "DEFAULT_ENTROPY_THRESHOLD",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MAX_CONTEXT_LENGTH",
    "OVERLAP",
    "STRIDE",
    "VALID_LABELS",
    "NVIDIA_TO_MODEL_MAP",
    # Model loader
    "load_model",
    "load_tokenizer",
    "load_model_and_tokenizer",
    "load_labels",
    # Label mapper
    "LabelMapper",
    # Metrics
    "compute_entropy_confidence",
]
