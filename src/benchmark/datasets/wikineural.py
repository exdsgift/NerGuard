"""WikiNeural multilingual dataset adapter.

Loads from HuggingFace `Babelscape/wikineural` (test splits per language).
Already word-tokenized with BIO-tagged labels.
"""

import logging
from typing import Dict, List, Set

from datasets import load_dataset

from src.benchmark.datasets.base import BenchmarkSample, DatasetAdapter

logger = logging.getLogger(__name__)

WIKINEURAL_LABELS = {"PER", "LOC", "ORG", "MISC"}

# WikiNeural uses integer tag IDs
WIKINEURAL_ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}

DEFAULT_LANGUAGES = ["en", "it", "es", "de", "fr", "pt", "nl", "pl"]


class WikiNeuralAdapter(DatasetAdapter):
    def __init__(self, languages: List[str] = None):
        self.languages = languages or DEFAULT_LANGUAGES

    def name(self) -> str:
        return "wikineural"

    def native_labels(self) -> Set[str]:
        return WIKINEURAL_LABELS.copy()

    def load(self, max_samples: int = 0, seed: int = 42, **kwargs) -> List[BenchmarkSample]:
        languages = kwargs.get("languages", self.languages)
        samples = []

        for lang in languages:
            split_name = f"test_{lang}"
            logger.info(f"Loading WikiNeural {split_name}...")

            try:
                ds = load_dataset("Babelscape/wikineural", split=split_name)
            except Exception as e:
                logger.warning(f"Failed to load {split_name}: {e}")
                continue

            if max_samples > 0 and max_samples < len(ds):
                ds = ds.shuffle(seed=seed).select(range(max_samples))

            for idx in range(len(ds)):
                example = ds[idx]
                tokens = example["tokens"]
                ner_tags = example["ner_tags"]

                if not tokens:
                    continue

                # Reconstruct text and character spans from tokens
                text_parts = []
                token_spans = []
                offset = 0
                for tok in tokens:
                    token_spans.append((offset, offset + len(tok)))
                    text_parts.append(tok)
                    offset += len(tok) + 1  # +1 for space

                text = " ".join(text_parts)

                # Convert integer tags to BIO string labels
                labels = []
                for tag_id in ner_tags:
                    labels.append(WIKINEURAL_ID2LABEL.get(tag_id, "O"))

                samples.append(BenchmarkSample(
                    text=text,
                    tokens=tokens,
                    token_spans=token_spans,
                    labels=labels,
                    sample_id=f"{lang}_{idx}",
                    language=lang,
                ))

            logger.info(f"  {lang}: {sum(1 for s in samples if s.language == lang)} samples")

        logger.info(f"Loaded {len(samples)} total samples from WikiNeural ({len(languages)} languages)")
        return samples

    def describe(self) -> Dict:
        base = super().describe()
        base.update({
            "source": "Babelscape/wikineural",
            "split": "test_{lang}",
            "languages": self.languages,
        })
        return base
