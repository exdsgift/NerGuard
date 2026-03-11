"""
NerGuard — Entropy-gated hybrid NER for privacy-compliant PII detection.

Primary use case: PII anonymization layer before inserting data into RAG pipelines.

Quick start::

    from nerguard import Redactor

    ng = Redactor()
    result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")

    print(result.text)
    # "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"

    print(result.mapping)
    # {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}

    print(result.entities)
    # [{"label": "GIVENNAME", "text": "John", "confidence": 0.998, ...}, ...]

With LLM routing (higher accuracy on ambiguous spans)::

    ng = Redactor(llm_routing=True, llm_source="ollama", llm_model="qwen2.5:7b")
    result = ng.redact("Call me at 078-05-1120")

With generic placeholders (maximum token compression for LLM context windows)::

    ng = Redactor(typed=False)
    result = ng.redact("John Smith, SSN 078-05-1120")
    # "Hi, I'm [PII] [PII]. SSN [PII]"

Batch redaction::

    results = ng.redact_batch(["text one", "text two"])
"""

from src.rag.redactor import nerguard as Redactor
from src.rag.models import RedactResult

__version__ = "1.0.0"
__all__ = ["Redactor", "RedactResult"]
