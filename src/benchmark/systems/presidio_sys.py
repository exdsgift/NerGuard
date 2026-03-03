"""Microsoft Presidio system wrapper."""

import logging
import time
from typing import Dict, List, Set, Tuple

from src.benchmark.alignment import CharSpan, align_spans_to_tokens
from src.benchmark.systems.base import SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("presidio-analyzer not installed — Presidio wrapper unavailable")


class PresidioWrapper(SystemWrapper):
    def __init__(self):
        self.analyzer = None

    def name(self) -> str:
        return "Presidio"

    def native_labels(self) -> Set[str]:
        # Presidio's default recognized entities
        return {
            "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
            "CRYPTO", "DATE_TIME", "DOMAIN_NAME", "IBAN_CODE",
            "IP_ADDRESS", "LOCATION", "MEDICAL_LICENSE", "NRP",
            "SG_NRIC_FIN", "UK_NHS", "URL",
            "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN",
            "US_PASSPORT", "US_SSN",
            "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
        }

    def setup(self) -> None:
        if not PRESIDIO_AVAILABLE:
            raise RuntimeError("presidio-analyzer not installed")
        logger.info("Initializing Presidio AnalyzerEngine")
        self.analyzer = AnalyzerEngine()
        logger.info("Presidio ready")

    def teardown(self) -> None:
        self.analyzer = None
        logger.info("Presidio teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        t0 = time.time()

        results = self.analyzer.analyze(text=text, language="en")

        entity_spans = [
            CharSpan(
                label=r.entity_type,
                start=r.start,
                end=r.end,
                text=text[r.start:r.end],
            )
            for r in results
        ]

        labels = align_spans_to_tokens(entity_spans, tokens, token_spans)

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=labels, latency_ms=latency_ms)
