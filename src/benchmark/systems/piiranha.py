"""Piiranha system wrapper — iiiorg/piiranha-v1-detect-personal-information."""

import logging
import time
from typing import Dict, List, Set, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.benchmark.systems.base import SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

PIIRANHA_MODEL = "iiiorg/piiranha-v1-detect-personal-information"
PIIRANHA_MAX_LENGTH = 256  # Piiranha's documented context length


class PiiranhaWrapper(SystemWrapper):
    def __init__(self, device: str = "auto"):
        self.device_str = device
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self._native_labels = None

    def name(self) -> str:
        return "Piiranha"

    def native_labels(self) -> Set[str]:
        if self._native_labels is not None:
            return self._native_labels
        return {
            "ACCOUNTNUM", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER",
            "DATEOFBIRTH", "DRIVERLICENSENUM", "EMAIL", "GIVENNAME",
            "IDCARDNUM", "PASSWORD", "SOCIALNUM", "STREET", "SURNAME",
            "TAXNUM", "TELEPHONENUM", "USERNAME", "ZIPCODE",
        }

    def setup(self) -> None:
        logger.info(f"Loading Piiranha model from {PIIRANHA_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(PIIRANHA_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(PIIRANHA_MODEL)

        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        self._native_labels = set()
        for label in self.id2label.values():
            if label != "O":
                clean = label.replace("B-", "").replace("I-", "")
                self._native_labels.add(clean)

        logger.info(f"Piiranha ready on {self.device} ({len(self._native_labels)} entity types)")

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Piiranha teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        t0 = time.time()

        # Sliding window for texts longer than context
        all_subword_preds = {}
        text_len = len(text)
        window_start = 0
        stride = PIIRANHA_MAX_LENGTH - 30  # overlap for continuity

        while window_start < text_len:
            # Find the window text (approximate char-based chunking)
            chunk_text = text if text_len <= PIIRANHA_MAX_LENGTH * 5 else text  # full text, let tokenizer truncate

            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=PIIRANHA_MAX_LENGTH,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            offset_mapping = encoding["offset_mapping"][0].tolist()

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits[0]

            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

            for sw_idx, (sw_start, sw_end) in enumerate(offset_mapping):
                if sw_start == sw_end == 0:
                    continue
                if sw_idx not in all_subword_preds or sw_start not in all_subword_preds:
                    all_subword_preds[sw_start] = (sw_start, sw_end, pred_ids[sw_idx])

            break  # For now, single pass; sliding window for very long texts

        # Map subword predictions to word-level tokens
        word_labels = self._align_to_word_tokens(
            pred_ids, offset_mapping, tokens, token_spans
        )

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=word_labels, latency_ms=latency_ms)

    def _align_to_word_tokens(
        self,
        subword_preds: List[int],
        subword_offsets: List[Tuple[int, int]],
        word_tokens: List[str],
        word_spans: List[Tuple[int, int]],
    ) -> List[str]:
        """Map subword-level predictions to word-level BIO labels."""
        word_labels = ["O"] * len(word_tokens)

        for word_idx, (w_start, w_end) in enumerate(word_spans):
            for sw_idx, (sw_start, sw_end) in enumerate(subword_offsets):
                if sw_start == sw_end == 0:
                    continue
                if sw_end <= w_start or sw_start >= w_end:
                    continue
                pred_id = subword_preds[sw_idx]
                pred = self.id2label.get(pred_id, "O")
                if pred != "O":
                    word_labels[word_idx] = pred
                    break

        # BIO repair
        for i in range(len(word_labels)):
            if word_labels[i].startswith("I-"):
                entity_type = word_labels[i][2:]
                if i == 0:
                    word_labels[i] = f"B-{entity_type}"
                else:
                    prev = word_labels[i - 1]
                    prev_type = prev[2:] if prev.startswith(("B-", "I-")) else None
                    if prev_type != entity_type:
                        word_labels[i] = f"B-{entity_type}"

        return word_labels
