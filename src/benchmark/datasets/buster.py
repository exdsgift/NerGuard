"""BUSTER financial NER dataset adapter.

Loads BUSTER (Business Entity Recognition) from HuggingFace.
SEC filing documents (M&A) with 6 entity types in BIO scheme.
Uses 5-fold cross-validation splits; we use FOLD_5 as test by default.
"""

import logging
import random
from typing import Dict, List, Set

from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter
from src.tasks.financial.config import BUSTER_DATASET, BUSTER_ENTITY_CLASSES

logger = logging.getLogger(__name__)


class BUSTERAdapter(DatasetAdapter):
    def name(self) -> str:
        return "buster"

    def native_labels(self) -> Set[str]:
        return BUSTER_ENTITY_CLASSES

    def load(self, max_samples: int = 0, seed: int = 42, **kwargs) -> List[BenchmarkSample]:
        from datasets import load_dataset

        split = kwargs.get("split", "FOLD_5")

        logger.info(f"Loading BUSTER ({split}) from HuggingFace...")
        ds = load_dataset(BUSTER_DATASET, split=split)

        examples = []
        for row in ds:
            tokens = row["tokens"]
            labels = row["labels"]
            # Filter to samples that have at least one entity
            if any(l != "O" for l in labels):
                examples.append(row)

        logger.info(
            f"BUSTER: {len(ds)} total, {len(examples)} with entities "
            f"({100 * len(examples) / len(ds):.1f}%)"
        )

        if max_samples is not None and max_samples > 0 and max_samples < len(examples):
            rng = random.Random(seed)
            examples = rng.sample(examples, max_samples)
            logger.info(f"Subsampled to {max_samples} samples (seed={seed})")

        samples = []
        for idx, ex in enumerate(examples):
            tokens = ex["tokens"]
            labels = ex["labels"]

            # Build character spans from tokens (space-joined)
            token_spans = []
            offset = 0
            for token in tokens:
                start = offset
                end = start + len(token)
                token_spans.append((start, end))
                offset = end + 1

            text = " ".join(tokens)

            samples.append(BenchmarkSample(
                text=text,
                tokens=tokens,
                token_spans=token_spans,
                labels=labels,
                sample_id=str(ex.get("document_id", idx)),
                language="en",
            ))

        logger.info(f"Loaded {len(samples)} samples from BUSTER ({split})")
        return samples

    def describe(self) -> Dict:
        base = super().describe()
        base.update({
            "source": BUSTER_DATASET,
            "entity_types": "6 M&A entity types (parties, advisors, revenues)",
            "domain": "financial (SEC filings, M&A)",
        })
        return base
