"""FiNER-139 financial NER dataset adapter.

Loads FiNER-139 (139 XBRL entity types from SEC filings) from HuggingFace.
Sentence-level dataset with US-GAAP taxonomy tags in BIO scheme.
"""

import json
import logging
import os
import random
import zipfile
from typing import Dict, List, Set

from huggingface_hub import hf_hub_download

from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter
from src.tasks.financial.config import FINER_DATASET, FINER_ENTITY_CLASSES, FINER_ZIP

logger = logging.getLogger(__name__)

EXTRACT_DIR = "/tmp/finer139"


class FiNER139Adapter(DatasetAdapter):
    def name(self) -> str:
        return "finer-139"

    def native_labels(self) -> Set[str]:
        return FINER_ENTITY_CLASSES

    def load(self, max_samples: int = 0, seed: int = 42, **kwargs) -> List[BenchmarkSample]:
        split = kwargs.get("split", "test")
        jsonl_file = f"{split}.jsonl"
        jsonl_path = os.path.join(EXTRACT_DIR, jsonl_file)

        # Download and extract if needed
        if not os.path.exists(jsonl_path):
            logger.info(f"Downloading FiNER-139 from HuggingFace...")
            zip_path = hf_hub_download(FINER_DATASET, FINER_ZIP, repo_type="dataset")
            os.makedirs(EXTRACT_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(EXTRACT_DIR)
            logger.info(f"Extracted to {EXTRACT_DIR}")

        logger.info(f"Loading FiNER-139 ({split}) from {jsonl_path}...")
        examples = []
        with open(jsonl_path) as f:
            for line in f:
                examples.append(json.loads(line))

        # Filter to samples that have at least one entity (83% are O-only)
        examples_with_entities = [
            ex for ex in examples
            if any(t != "O" for t in ex["ner_tags"])
        ]
        logger.info(
            f"FiNER-139: {len(examples)} total, "
            f"{len(examples_with_entities)} with entities ({100*len(examples_with_entities)/len(examples):.1f}%)"
        )
        # Use entity-containing samples for evaluation
        examples = examples_with_entities

        if max_samples is not None and max_samples > 0 and max_samples < len(examples):
            rng = random.Random(seed)
            examples = rng.sample(examples, max_samples)
            logger.info(f"Subsampled to {max_samples} samples (seed={seed})")

        samples = []
        for idx, ex in enumerate(examples):
            tokens = ex["tokens"]
            ner_tags = ex["ner_tags"]  # Already string BIO labels

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
                labels=ner_tags,
                sample_id=str(ex.get("id", idx)),
                language="en",
            ))

        logger.info(f"Loaded {len(samples)} samples from FiNER-139 ({split})")
        return samples

    def describe(self) -> Dict:
        base = super().describe()
        base.update({
            "source": FINER_DATASET,
            "entity_types": "139 XBRL US-GAAP tags",
            "domain": "financial (SEC filings)",
        })
        return base
