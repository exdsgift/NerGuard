"""NVIDIA Nemotron-PII dataset adapter.

Loads from HuggingFace `nvidia/Nemotron-PII` (train split — only available split).
Uses text + spans for character-level annotations → word-level BIO labels.
Labels are kept as NVIDIA's native names (first_name, email, etc.).
"""

import ast
import json
import logging
from typing import Dict, List, Set

from datasets import load_dataset

from src.benchmark.alignment import CharSpan, align_spans_to_tokens, word_tokenize
from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter

logger = logging.getLogger(__name__)

# All labels observed in the dataset
NVIDIA_ALL_LABELS = {
    "account_number", "age", "api_key", "bank_routing_number",
    "biometric_identifier", "blood_type", "certificate_license_number",
    "city", "company_name", "coordinate", "country", "county",
    "credit_debit_card", "customer_id", "cvv", "date", "date_of_birth",
    "date_time", "device_identifier", "education_level", "email",
    "employee_id", "employment_status", "fax_number", "first_name",
    "gender", "health_plan_beneficiary_number", "http_cookie",
    "ipv4", "ipv6", "language", "last_name", "license_plate",
    "mac_address", "medical_record_number", "occupation", "password",
    "phone_number", "pin", "political_view", "postcode",
    "race_ethnicity", "religious_belief", "sexuality", "ssn",
    "state", "street_address", "swift_bic", "tax_id", "time",
    "unique_id", "url", "user_name", "vehicle_identifier",
}


def _parse_spans(spans_raw) -> list:
    """Robustly parse spans from various formats."""
    if isinstance(spans_raw, list):
        return spans_raw
    if isinstance(spans_raw, str):
        try:
            return json.loads(spans_raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(spans_raw)
            except (ValueError, SyntaxError):
                return []
    return []


class NvidiaPIIAdapter(DatasetAdapter):
    def name(self) -> str:
        return "nvidia-pii"

    def native_labels(self) -> Set[str]:
        return NVIDIA_ALL_LABELS.copy()

    def load(self, max_samples: int = 0, seed: int = 42, **kwargs) -> List[BenchmarkSample]:
        logger.info("Loading NVIDIA/Nemotron-PII dataset from HuggingFace...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train")

        if max_samples > 0 and max_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_samples))
            logger.info(f"Subsampled to {max_samples} samples (seed={seed})")

        logger.info(f"Processing {len(ds)} samples...")
        samples = []

        for idx in range(len(ds)):
            example = ds[idx]
            text = example["text"]
            spans_raw = example.get("spans", [])
            sample_id = example.get("uid", str(idx))

            # Word-level tokenization
            tokens, token_spans = word_tokenize(text)
            if not tokens:
                continue

            # Parse spans
            spans = _parse_spans(spans_raw)
            entity_spans = []
            for span in spans:
                if not isinstance(span, dict):
                    continue
                label = span.get("label", "")
                start = span.get("start")
                end = span.get("end")
                if label and start is not None and end is not None:
                    entity_spans.append(CharSpan(
                        label=label,
                        start=start,
                        end=end,
                        text=span.get("text", ""),
                    ))

            # Align to word tokens
            labels = align_spans_to_tokens(entity_spans, tokens, token_spans)

            samples.append(BenchmarkSample(
                text=text,
                tokens=tokens,
                token_spans=token_spans,
                labels=labels,
                sample_id=sample_id,
                language="en",
            ))

        logger.info(f"Loaded {len(samples)} samples from NVIDIA/Nemotron-PII")
        return samples

    def describe(self) -> Dict:
        base = super().describe()
        base.update({
            "source": "nvidia/Nemotron-PII",
            "split": "train",
            "total_size": 100000,
        })
        return base
