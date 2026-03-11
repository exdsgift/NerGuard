"""Data models for NerGuard RAG redaction."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RedactResult:
    """Result of a RAG-optimized redaction.

    Attributes:
        text: Redacted text with typed placeholders (e.g. [NAME], [EMAIL]).
        entities: List of detected entity dicts, each with keys:
            label, text, start, end, confidence, source.
        mapping: Dict mapping each placeholder instance to its original value,
            keyed as "<LABEL>_<index>" (e.g. "NAME_0", "EMAIL_0").
            Useful for auditing or selective de-redaction.
    """
    text: str
    entities: List[Dict] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)
