"""BC5CDR biomedical NER dataset adapter.

Loads BC5CDR (BioCreative V Chemical Disease Relation) from HuggingFace.
Sentence-level dataset with Chemical and Disease entities in BIO scheme.
"""

import json
import logging
import random
from typing import Dict, List, Set

from huggingface_hub import hf_hub_download

from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter
from src.tasks.biomedical.config import BC5CDR_DATASET, BC5CDR_TAG_TO_LABEL

logger = logging.getLogger(__name__)


class BC5CDRAdapter(DatasetAdapter):
    def name(self) -> str:
        return "bc5cdr"

    def native_labels(self) -> Set[str]:
        return {"Chemical", "Disease"}

    def load(self, max_samples: int = 0, seed: int = 42, **kwargs) -> List[BenchmarkSample]:
        split = kwargs.get("split", "test")
        split_file = {"test": "test.json", "train": "train.json", "validation": "valid.json"}
        filename = split_file.get(split, "test.json")

        logger.info(f"Loading BC5CDR ({split}) from HuggingFace...")
        path = hf_hub_download(BC5CDR_DATASET, f"dataset/{filename}", repo_type="dataset")

        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))

        if max_samples is not None and max_samples > 0 and max_samples < len(examples):
            rng = random.Random(seed)
            examples = rng.sample(examples, max_samples)
            logger.info(f"Subsampled to {max_samples} samples (seed={seed})")

        samples = []
        for idx, ex in enumerate(examples):
            tokens = ex["tokens"]
            tag_ids = ex["tags"]

            # Convert integer tags to BIO string labels
            labels = [BC5CDR_TAG_TO_LABEL.get(t, "O") for t in tag_ids]

            # Build character spans from tokens (space-joined reconstruction)
            text_parts = []
            token_spans = []
            offset = 0
            for token in tokens:
                start = offset
                end = start + len(token)
                token_spans.append((start, end))
                text_parts.append(token)
                offset = end + 1  # +1 for space

            text = " ".join(text_parts)

            samples.append(BenchmarkSample(
                text=text,
                tokens=tokens,
                token_spans=token_spans,
                labels=labels,
                sample_id=str(idx),
                language="en",
            ))

        logger.info(f"Loaded {len(samples)} samples from BC5CDR ({split})")
        return samples

    def describe(self) -> Dict:
        base = super().describe()
        base.update({
            "source": BC5CDR_DATASET,
            "entity_types": "Chemical, Disease",
            "domain": "biomedical (PubMed abstracts)",
        })
        return base
