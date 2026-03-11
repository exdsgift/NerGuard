"""nerguard-rag compatibility shim.

The RAG API now lives in the main nerguard package (src/rag/).
This module re-exports everything for backwards compatibility.

Preferred import (after pip install nerguard):

    from src.rag import nerguard, RedactResult
"""

from src.rag.redactor import nerguard
from src.rag.models import RedactResult

__all__ = ["nerguard", "RedactResult"]
