"""NerGuard RAG — PII redaction with typed placeholders.

Quick start::

    from src.rag import NerGuard

    ng = NerGuard()
    result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")

    print(result.text)
    # "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"

    print(result.mapping)
    # {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}
"""

from src.rag.redactor import NerGuard
from src.rag.models import RedactResult

__all__ = ["NerGuard", "RedactResult"]
