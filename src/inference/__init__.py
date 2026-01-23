"""
Inference module for NerGuard.

This module provides:
- LLMRouter: Intelligent routing to LLM for uncertain predictions
- LLMCache: In-memory caching for LLM responses
- PIITester: Complete inference pipeline for PII detection
- Prompt templates for LLM disambiguation
"""

from src.inference.llm_router import LLMRouter, LLMCache
from src.inference.prompts import (
    PROMPT,
    PROMPT_V5,
    VALID_LABELS_STR,
)

__all__ = [
    "LLMRouter",
    "LLMCache",
    "PROMPT",
    "PROMPT_V5",
    "VALID_LABELS_STR",
]
