"""
Logging configuration for NerGuard.

This module provides unified logging setup used throughout the project.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logging(
    name: str = "NerGuard",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = DEFAULT_FORMAT,
    console_output: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger with consistent settings.

    Args:
        name: Logger name (default: "NerGuard")
        level: Logging level (default: INFO)
        log_file: Optional path to a log file
        format_string: Log message format
        console_output: Whether to output to console (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(format_string)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience function for getting child loggers.

    Args:
        name: Logger name (will be prefixed with "NerGuard.")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"NerGuard.{name}")


def set_log_level(level: int, logger_name: str = "NerGuard") -> None:
    """
    Set the logging level for a logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        logger_name: Name of the logger to configure
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def suppress_external_loggers() -> None:
    """
    Suppress verbose logging from external libraries.

    This is useful for reducing noise from libraries like transformers.
    """
    # Suppress transformers logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Suppress other verbose loggers
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
