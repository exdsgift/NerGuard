"""GLiNER zero-shot NER system wrapper."""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple

from src.benchmark.alignment import CharSpan, align_spans_to_tokens
from src.benchmark.systems.base import SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("gliner not installed — GLiNER wrapper unavailable")

GLINER_MODEL = "urchade/gliner_multi-v2.1"


class GLiNERWrapper(SystemWrapper):
    """GLiNER zero-shot NER.

    Since GLiNER accepts arbitrary labels, we pass the dataset's native labels
    as input. This gives full native overlap without any mapping.
    """

    def __init__(self, dataset_labels: Optional[Set[str]] = None, device: str = "auto"):
        self.device_str = device
        self.model = None
        self._dataset_labels = dataset_labels or set()

    def set_dataset_labels(self, labels: Set[str]) -> None:
        """Set the labels GLiNER should detect (from the current dataset)."""
        self._dataset_labels = labels

    def name(self) -> str:
        return "GLiNER"

    def native_labels(self) -> Set[str]:
        return self._dataset_labels.copy()

    def setup(self) -> None:
        if not GLINER_AVAILABLE:
            raise RuntimeError("gliner not installed")

        import torch
        logger.info(f"Loading GLiNER model from {GLINER_MODEL}")

        device = self.device_str
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = GLiNER.from_pretrained(GLINER_MODEL).to(device)
        logger.info(f"GLiNER ready on {device}")

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GLiNER teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        t0 = time.time()

        if not self._dataset_labels:
            labels = ["O"] * len(tokens)
            return SystemPrediction(labels=labels, latency_ms=0.0)

        # GLiNER expects lowercase label names
        label_list = [lbl.lower() for lbl in self._dataset_labels]
        label_map = {lbl.lower(): lbl for lbl in self._dataset_labels}

        preds = self.model.predict_entities(text, label_list, threshold=0.5)

        entity_spans = []
        for p in preds:
            original_label = label_map.get(p["label"], p["label"].upper())
            entity_spans.append(CharSpan(
                label=original_label,
                start=p["start"],
                end=p["end"],
                text=p.get("text", ""),
            ))

        labels = align_spans_to_tokens(entity_spans, tokens, token_spans)

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=labels, latency_ms=latency_ms)
