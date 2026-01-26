"""
Training module for NerGuard.

This module provides:
- PIIEncoder: Model factory for mDeBERTa-based token classification
- Training utilities and configurations
- Validation pipeline
"""

from src.training.encoder import PIIEncoder

__all__ = [
    "PIIEncoder",
]
