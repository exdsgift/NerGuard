"""dslim/bert-base-NER system wrapper — standard HuggingFace NER model."""

import logging
import time
from typing import Dict, List, Set, Tuple

from src.benchmark.alignment import CharSpan, align_spans_to_tokens
from src.benchmark.systems.base import SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

BERT_NER_MODEL = "dslim/bert-base-NER"


class BertNERWrapper(SystemWrapper):
    def __init__(self, device: str = "auto"):
        self.device_str = device
        self.pipeline = None

    def name(self) -> str:
        return "dslim/bert-base-NER"

    def native_labels(self) -> Set[str]:
        return {"PER", "LOC", "ORG", "MISC"}

    def setup(self) -> None:
        import torch
        from transformers import pipeline

        device_num = 0 if (self.device_str == "auto" and torch.cuda.is_available()) else -1
        if self.device_str == "cuda":
            device_num = 0
        elif self.device_str == "cpu":
            device_num = -1

        logger.info(f"Loading {BERT_NER_MODEL} pipeline")
        self.pipeline = pipeline(
            "ner",
            model=BERT_NER_MODEL,
            device=device_num,
            aggregation_strategy="none",
        )
        logger.info(f"bert-base-NER ready (device={device_num})")

    def teardown(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("bert-base-NER teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        t0 = time.time()

        # Run NER pipeline (returns subword-level entities)
        try:
            ner_results = self.pipeline(text)
        except Exception:
            labels = ["O"] * len(tokens)
            latency_ms = (time.time() - t0) * 1000
            return SystemPrediction(labels=labels, latency_ms=latency_ms)

        # Convert pipeline output to character spans
        # Pipeline with aggregation_strategy="none" returns per-token results
        # with 'entity', 'start', 'end' fields
        entity_spans = []
        current_span = None

        for item in ner_results:
            entity = item["entity"]  # e.g., "B-PER", "I-PER"
            start = item["start"]
            end = item["end"]

            if entity.startswith("B-"):
                # Save previous span if any
                if current_span is not None:
                    entity_spans.append(current_span)
                label = entity[2:]
                current_span = CharSpan(label=label, start=start, end=end, text=text[start:end])
            elif entity.startswith("I-") and current_span is not None:
                label = entity[2:]
                if label == current_span.label:
                    # Extend current span
                    current_span.end = end
                    current_span.text = text[current_span.start:end]
                else:
                    entity_spans.append(current_span)
                    current_span = CharSpan(label=label, start=start, end=end, text=text[start:end])
            elif entity.startswith("I-"):
                # I- without B-, treat as B-
                label = entity[2:]
                current_span = CharSpan(label=label, start=start, end=end, text=text[start:end])

        if current_span is not None:
            entity_spans.append(current_span)

        labels = align_spans_to_tokens(entity_spans, tokens, token_spans)

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=labels, latency_ms=latency_ms)
