"""NerGuard Base system wrapper — mDeBERTa model only, no LLM routing."""

import logging
import time
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.benchmark.systems.base import SystemPrediction, SystemWrapper

logger = logging.getLogger(__name__)

MAX_LENGTH = 512
STRIDE = 382


class NerGuardBase(SystemWrapper):
    def __init__(self, model_path: str = "./models/mdeberta-pii-safe/final", device: str = "auto"):
        self.model_path = model_path
        self.device_str = device
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self._native_labels = None

    def name(self) -> str:
        return "NerGuard Base"

    def native_labels(self) -> Set[str]:
        if self._native_labels is not None:
            return self._native_labels
        # Fallback before setup
        return {
            "AGE", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATE",
            "DRIVERLICENSENUM", "EMAIL", "GENDER", "GIVENNAME", "IDCARDNUM",
            "PASSPORTNUM", "SEX", "SOCIALNUM", "STREET", "SURNAME",
            "TAXNUM", "TELEPHONENUM", "TIME", "TITLE", "ZIPCODE",
        }

    def setup(self) -> None:
        logger.info(f"Loading NerGuard model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)

        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        # Extract native labels from model config
        self._native_labels = set()
        for label in self.id2label.values():
            if label != "O":
                clean = label.replace("B-", "").replace("I-", "")
                self._native_labels.add(clean)

        logger.info(f"NerGuard Base ready on {self.device} ({len(self._native_labels)} entity types)")

    def teardown(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NerGuard Base teardown complete")

    def predict(
        self,
        text: str,
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
    ) -> SystemPrediction:
        t0 = time.time()

        # Tokenize with subword tokenizer
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[0]

        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        # Map subword predictions to word-level tokens via character overlap
        labels = self._align_to_word_tokens(pred_ids, offset_mapping, tokens, token_spans)

        latency_ms = (time.time() - t0) * 1000
        return SystemPrediction(labels=labels, latency_ms=latency_ms)

    def _align_to_word_tokens(
        self,
        subword_preds: List[int],
        subword_offsets: List[Tuple[int, int]],
        word_tokens: List[str],
        word_spans: List[Tuple[int, int]],
    ) -> List[str]:
        """Map subword-level predictions back to word-level BIO labels."""
        word_labels = ["O"] * len(word_tokens)

        for word_idx, (w_start, w_end) in enumerate(word_spans):
            # Find first subword token that overlaps this word
            best_pred = None
            for sw_idx, (sw_start, sw_end) in enumerate(subword_offsets):
                if sw_start == sw_end == 0:
                    continue  # Special token
                # Check overlap
                if sw_end <= w_start or sw_start >= w_end:
                    continue
                # Take prediction of first overlapping subword
                if best_pred is None:
                    pred_id = subword_preds[sw_idx]
                    best_pred = self.id2label.get(pred_id, "O")
                    break

            if best_pred is not None and best_pred != "O":
                word_labels[word_idx] = best_pred

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
