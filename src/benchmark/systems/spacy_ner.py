"""spaCy NER system wrapper."""

import logging
import time
from typing import Dict, List, Set, Tuple

from src.benchmark.alignment import CharSpan, align_spans_to_tokens
from src.benchmark.systems.base import SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not installed — spaCy wrapper unavailable")

SPACY_MODEL = "en_core_web_trf"
SPACY_FALLBACK_MODEL = "en_core_web_lg"


class SpacyWrapper(SystemWrapper):
    def __init__(self):
        self.nlp = None
        self._model_name = None

    def name(self) -> str:
        model = self._model_name or SPACY_MODEL
        return f"spaCy ({model})"

    def native_labels(self) -> Set[str]:
        # spaCy OntoNotes labels
        return {
            "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
            "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
            "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL",
        }

    def setup(self) -> None:
        if not SPACY_AVAILABLE:
            raise RuntimeError("spacy not installed")

        # Try trf model first, fall back to lg
        for model_name in [SPACY_MODEL, SPACY_FALLBACK_MODEL]:
            try:
                if not spacy.util.is_package(model_name):
                    logger.info(f"Downloading spaCy model: {model_name}")
                    spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
                self._model_name = model_name
                logger.info(f"spaCy ready with model: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")

        raise RuntimeError(f"Could not load any spaCy model ({SPACY_MODEL}, {SPACY_FALLBACK_MODEL})")

    def teardown(self) -> None:
        self.nlp = None
        logger.info("spaCy teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        t0 = time.time()

        doc = self.nlp(text)

        entity_spans = [
            CharSpan(
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                text=ent.text,
            )
            for ent in doc.ents
        ]

        labels = align_spans_to_tokens(entity_spans, tokens, token_spans)

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=labels, latency_ms=latency_ms)
