"""
Utility module for NerGuard.

This module contains general-purpose utilities:
- io: File I/O operations (ensure_dir, JSON handling)
- logging_config: Unified logging configuration
- colors: ANSI color codes for terminal output
- samples: Sample PII texts for testing
- calibration: Debug and calibration utilities
"""

from src.utils.io import ensure_dir, load_json, save_json
from src.utils.logging_config import setup_logging, get_logger
from src.utils.colors import Colors

__all__ = [
    "ensure_dir",
    "load_json",
    "save_json",
    "setup_logging",
    "get_logger",
    "Colors",
]
