"""
File I/O utilities for NerGuard.

This module provides common file operations used throughout the project.
"""

import json
import os
from typing import Any, Dict, Optional, Union
from pathlib import Path


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory if it doesn't exist.

    Args:
        path: Path to the directory to create

    Returns:
        Path object for the created/existing directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file
        encoding: File encoding (default: utf-8)

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding=encoding) as f:
        return json.load(f)


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        path: Path to save the file
        indent: JSON indentation level (default: 2)
        encoding: File encoding (default: utf-8)
        ensure_ascii: If False, allows non-ASCII characters (default: False)
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_text(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> str:
    """
    Read text content from a file.

    Args:
        path: Path to the file
        encoding: File encoding (default: utf-8)

    Returns:
        File content as string
    """
    path = Path(path)
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write_text(
    content: str,
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> None:
    """
    Write text content to a file.

    Args:
        content: Text content to write
        path: Path to save the file
        encoding: File encoding (default: utf-8)
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding=encoding) as f:
        f.write(content)


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if a file exists.

    Args:
        path: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.

    Args:
        path: Path to the file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.stat().st_size


def format_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"
