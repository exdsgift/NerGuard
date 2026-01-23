"""
Model loading utilities for NerGuard.

This module provides unified functions for loading models, tokenizers, and label mappings,
eliminating code duplication across the codebase.
"""

import json
import os
import logging
from typing import Dict, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from src.core.constants import DEFAULT_MODEL_PATH, DEFAULT_LABEL_PATH

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_path: str = DEFAULT_MODEL_PATH,
    use_fast: bool = True,
    add_prefix_space: bool = True,
) -> PreTrainedTokenizer:
    """
    Load a tokenizer from a pretrained model path.

    Args:
        model_path: Path to the pretrained model or HuggingFace model name
        use_fast: Whether to use the fast tokenizer implementation
        add_prefix_space: Whether to add a space before the first token

    Returns:
        PreTrainedTokenizer instance

    Raises:
        OSError: If the tokenizer cannot be loaded
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=use_fast,
            add_prefix_space=add_prefix_space,
        )
        logger.debug(f"Loaded tokenizer from {model_path}")
        return tokenizer
    except OSError as e:
        logger.error(f"Failed to load tokenizer from {model_path}: {e}")
        raise


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> PreTrainedModel:
    """
    Load a model from a pretrained path.

    Args:
        model_path: Path to the pretrained model or HuggingFace model name
        device: Device to load the model on ('cuda', 'cpu', or None for auto)
        eval_mode: Whether to set the model to evaluation mode

    Returns:
        PreTrainedModel instance

    Raises:
        OSError: If the model cannot be loaded
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model = model.to(device)

        if eval_mode:
            model.eval()

        logger.debug(f"Loaded model from {model_path} on {device}")
        return model
    except OSError as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def load_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    eval_mode: bool = True,
    use_fast: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load both model and tokenizer from a pretrained path.

    This is a convenience function that combines load_model and load_tokenizer.

    Args:
        model_path: Path to the pretrained model or HuggingFace model name
        device: Device to load the model on ('cuda', 'cpu', or None for auto)
        eval_mode: Whether to set the model to evaluation mode
        use_fast: Whether to use the fast tokenizer implementation

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = load_tokenizer(model_path, use_fast=use_fast)
    model = load_model(model_path, device=device, eval_mode=eval_mode)
    return model, tokenizer


def load_labels(
    label_path: str = DEFAULT_LABEL_PATH,
    as_id2label: bool = True,
) -> Dict:
    """
    Load label mappings from a JSON file.

    Args:
        label_path: Path to the JSON file containing label mappings
        as_id2label: If True, returns {int: str} mapping. If False, returns {str: int}

    Returns:
        Dictionary mapping label IDs to label names (or vice versa)

    Raises:
        FileNotFoundError: If the label file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    if not os.path.exists(label_path):
        logger.error(f"Label file not found: {label_path}")
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if as_id2label:
        # Convert string keys to integers
        return {int(k): v for k, v in data.items()}
    else:
        # Invert the mapping
        return {v: int(k) for k, v in data.items()}


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Returns:
        torch.device for CUDA if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.debug("Using CPU device")
    return device
