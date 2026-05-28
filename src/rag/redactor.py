"""RAG-optimized PII redactor using typed placeholders."""

from typing import Dict, List

from src.rag.models import RedactResult

# Normalized short labels for RAG-friendly placeholders.
# Based on PPDL / PrivacyMind conventions: typed placeholders preserve
# semantic context without exposing the underlying value.
_LABEL_MAP: Dict[str, str] = {
    "GIVENNAME": "NAME",
    "SURNAME": "NAME",
    "TITLE": "TITLE",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "SOCIALNUM": "SSN",
    "CREDITCARDNUMBER": "CC",
    "PASSPORTNUM": "PASSPORT",
    "IDCARDNUM": "ID",
    "DRIVERLICENSENUM": "DL",
    "TAXNUM": "TAX",
    "IBAN": "IBAN",
    "STREET": "ADDR",
    "BUILDINGNUM": "ADDR",
    "CITY": "CITY",
    "ZIPCODE": "ZIP",
    "DATE": "DATE",
    "TIME": "TIME",
    "AGE": "AGE",
    "SEX": "DEMO",
    "GENDER": "DEMO",
}


def _normalize_label(label: str) -> str:
    return _LABEL_MAP.get(label, label)


class NerGuard:
    """RAG-optimized PII redactor.

    Uses NerGuard's full hybrid pipeline, then replaces block characters (█████)
    with compact typed placeholders ([NAME], [EMAIL], etc.) that preserve
    semantic context for downstream LLMs while minimizing token usage.

    Args:
        model_path: Local path or HuggingFace Hub ID for the NER model.
            Auto-downloads from HuggingFace if not found locally.
        llm_routing: Enable entropy-gated LLM routing for uncertain spans.
        llm_source: LLM backend — "openai" or "ollama".
        llm_model: LLM model name (e.g. "gpt-4o", "qwen2.5:7b").
        typed: If True (default), use typed placeholders like [NAME], [EMAIL].
            If False, use the generic [PII] marker for maximum compression.

    Example::

        from nerguard import Redactor

        ng = Redactor()
        result = ng.redact("Hi, I'm John Smith. Email: john@acme.com")
        print(result.text)
        # "Hi, I'm [NAME] [NAME]. Email: [EMAIL]"
        print(result.mapping)
        # {"NAME_0": "John", "NAME_1": "Smith", "EMAIL_0": "john@acme.com"}
    """

    def __init__(
        self,
        model_path: str = "./models/mdeberta-pii-safe/final",
        llm_routing: bool = False,
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        typed: bool = True,
        api_key: str = None,
    ) -> None:
        self.model_path = model_path
        self.llm_routing = llm_routing
        self.llm_source = llm_source
        self.llm_model = llm_model
        self.typed = typed
        self.api_key = api_key
        self._pipeline = None  # lazy-loaded on first call
        self._model = None     # cached NER model
        self._tokenizer = None # cached tokenizer

    def _load_pipeline(self) -> None:
        """Lazy-load model, tokenizer, and pipeline — runs only once per instance."""
        if self._pipeline is None:
            from src.core.pipeline import redact_pipeline
            from src.core.model_loader import load_model_and_tokenizer, get_device
            device = get_device()
            self._model, self._tokenizer = load_model_and_tokenizer(
                self.model_path, device=str(device), eval_mode=True
            )
            self._pipeline = redact_pipeline

    def redact(self, text: str) -> RedactResult:
        """Detect and redact PII from text using typed placeholders.

        Args:
            text: Input text to redact.

        Returns:
            RedactResult with redacted text, entity list, and value mapping.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text exceeds 100,000 characters.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return RedactResult(text=text, entities=[], mapping={})
        if len(text) > 100_000:
            raise ValueError(f"text exceeds 100,000 characters ({len(text)})")

        self._load_pipeline()
        entities, _ = self._pipeline(
            text=text,
            model_path=self.model_path,
            llm_routing=self.llm_routing,
            llm_source=self.llm_source,
            llm_model=self.llm_model,
            model=self._model,
            tokenizer=self._tokenizer,
            api_key=self.api_key,
        )
        return self._build_result(text, entities)

    def redact_batch(self, texts: List[str]) -> List[RedactResult]:
        """Redact a list of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of RedactResult, one per input text.
        """
        self._load_pipeline()
        return [self.redact(t) for t in texts]

    def _build_result(self, original_text: str, entities: List[Dict]) -> RedactResult:
        """Replace entity spans with typed placeholders and build mapping."""
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        redacted = original_text
        for entity in sorted_entities:
            short = _normalize_label(entity["label"]) if self.typed else "PII"
            redacted = redacted[:entity["start"]] + f"[{short}]" + redacted[entity["end"]:]

        mapping = self._reindex_mapping(entities)
        return RedactResult(text=redacted, entities=entities, mapping=mapping)

    def _reindex_mapping(self, entities: List[Dict]) -> Dict[str, str]:
        """Build mapping in forward (left-to-right) order with stable indices."""
        label_counters: Dict[str, int] = {}
        mapping: Dict[str, str] = {}
        for entity in sorted(entities, key=lambda e: e["start"]):
            short = _normalize_label(entity["label"]) if self.typed else "PII"
            idx = label_counters.get(short, 0)
            label_counters[short] = idx + 1
            mapping[f"{short}_{idx}"] = entity["text"]
        return mapping
