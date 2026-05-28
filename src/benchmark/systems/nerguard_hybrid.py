"""NerGuard Hybrid system wrapper — mDeBERTa + OpenAI LLM routing + regex validation."""

import logging
import time
from typing import Dict, List, Set, Tuple

from src.benchmark.systems.base import SystemPrediction, SystemWrapper
from src.core.pipeline_class import HybridPipeline

logger = logging.getLogger(__name__)


class NerGuardHybrid(SystemWrapper):
    def __init__(
        self,
        model_path: str = "./models/mdeberta-pii-safe/final",
        device: str = "auto",
        llm_source: str = "openai",
        llm_model: str = "gpt-4o",
        span_prompt_version: str = "V14_SPAN",
    ):
        self.model_path = model_path
        self.device = device
        self.llm_source = llm_source
        self.llm_model = llm_model
        self.pipeline = None

    def name(self) -> str:
        return f"NerGuard Hybrid ({self.llm_model})"

    def native_labels(self) -> Set[str]:
        return {
            "AGE", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATE",
            "DRIVERLICENSENUM", "EMAIL", "GENDER", "GIVENNAME", "IDCARDNUM",
            "PASSPORTNUM", "SEX", "SOCIALNUM", "STREET", "SURNAME",
            "TAXNUM", "TELEPHONENUM", "TIME", "TITLE", "ZIPCODE",
        }

    def setup(self) -> None:
        logger.info(f"Loading NerGuard Hybrid via unified pipeline (LLM: {self.llm_source}/{self.llm_model})")
        
        # Determine device
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu" if self.device == "auto" else self.device

        self.pipeline = HybridPipeline(
            model_path=self.model_path,
            llm_routing=True,
            llm_source=self.llm_source,
            llm_model=self.llm_model,
            device=device_str
        )

    def teardown(self) -> None:
        self.pipeline = None

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        from src.benchmark.alignment import align_spans_to_tokens, CharSpan

        t0 = time.time()

        entities, _ = self.pipeline.process_text(text)
        
        # Convert entities to CharSpans
        char_spans = [
            CharSpan(label=e["label"], start=e["start"], end=e["end"], text=e["text"])
            for e in entities
        ]
        
        word_labels = align_spans_to_tokens(char_spans, tokens, token_spans)

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=word_labels, latency_ms=latency_ms)
