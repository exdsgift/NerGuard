"""AI4Privacy 500k dataset adapter.

Loads from HuggingFace `ai4privacy/open-pii-masking-500k-ai4privacy` (validation split).
Uses source_text + privacy_mask for character-level annotations → word-level BIO labels.
"""

import logging
from typing import Dict, List, Set

from datasets import load_dataset

from src.benchmark.alignment import CharSpan, align_spans_to_tokens, word_tokenize
from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter

logger = logging.getLogger(__name__)

AI4PRIVACY_LABELS = {
    "AGE", "BUILDINGNUM", "CITY", "CREDITCARDNUMBER", "DATE",
    "DRIVERLICENSENUM", "EMAIL", "GENDER", "GIVENNAME", "IDCARDNUM",
    "PASSPORTNUM", "SEX", "SOCIALNUM", "STREET", "SURNAME",
    "TAXNUM", "TELEPHONENUM", "TIME", "TITLE", "ZIPCODE",
}


class AI4PrivacyAdapter(DatasetAdapter):
    def name(self) -> str:
        return "ai4privacy"

    def native_labels(self) -> Set[str]:
        return AI4PRIVACY_LABELS.copy()

    def load(self, max_samples: int = 0, seed: int = 42, **kwargs) -> List[BenchmarkSample]:
        logger.info("Loading AI4Privacy validation split from HuggingFace...")
        ds = load_dataset(
            "ai4privacy/open-pii-masking-500k-ai4privacy",
            split="validation",
        )

        if max_samples > 0 and max_samples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_samples))
            logger.info(f"Subsampled to {max_samples} samples (seed={seed})")

        logger.info(f"Processing {len(ds)} samples...")
        samples = []

        for idx, example in enumerate(ds):
            text = example["source_text"]
            privacy_mask = example.get("privacy_mask", [])
            language = example.get("language", "en")
            sample_id = example.get("uid", str(idx))

            # Word-level tokenization
            tokens, token_spans = word_tokenize(text)
            if not tokens:
                continue

            # Build character spans from privacy_mask
            entity_spans = []
            for mask in privacy_mask:
                label = mask.get("label", "")
                if label in AI4PRIVACY_LABELS:
                    entity_spans.append(CharSpan(
                        label=label,
                        start=mask["start"],
                        end=mask["end"],
                        text=mask.get("value", ""),
                    ))

            # Align to word tokens
            labels = align_spans_to_tokens(entity_spans, tokens, token_spans)

            samples.append(BenchmarkSample(
                text=text,
                tokens=tokens,
                token_spans=token_spans,
                labels=labels,
                sample_id=sample_id,
                language=language,
            ))

        logger.info(f"Loaded {len(samples)} samples from AI4Privacy")
        return samples

    def describe(self) -> Dict:
        base = super().describe()
        base.update({
            "source": "ai4privacy/open-pii-masking-500k-ai4privacy",
            "split": "validation",
            "total_size": 116077,
        })
        return base
