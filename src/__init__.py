"""
NerGuard - Hybrid NER for PII Detection

A production-ready toolkit for detecting and protecting Personally Identifiable Information (PII)
using a hybrid approach combining transformer models (mDeBERTa v3) with LLM-based disambiguation.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nerguard")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "NerGuard Team"
