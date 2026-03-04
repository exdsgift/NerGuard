"""Benchmark configuration and CLI argument parsing."""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


ALL_SYSTEMS = [
    "nerguard-base",
    "nerguard-hybrid",
    "nerguard-hybrid-v2",
    "piiranha",
    "piiranha-hybrid",
    "presidio",
    "gliner",
    "spacy",
    "bert-ner",
]

ALL_DATASETS = [
    "ai4privacy",
    "nvidia-pii",
    "wikineural",
]

# Per-dataset default sample sizes
DATASET_DEFAULT_SAMPLES = {
    "ai4privacy": 5000,
    "nvidia-pii": 5000,
    "wikineural": 1000,  # per language
}

# Languages supported by NerGuard (trained on AI4Privacy) AND present in WikiNeural
ALL_LANGUAGES = ["en", "it", "es", "de", "fr", "nl"]


@dataclass
class BenchmarkConfig:
    systems: List[str] = field(default_factory=lambda: list(ALL_SYSTEMS))
    datasets: List[str] = field(default_factory=lambda: list(ALL_DATASETS))
    samples: int = 0  # 0 = use per-dataset defaults
    output_dir: str = "./experiments"
    batch_size: int = 16
    runs: int = 3
    seed: int = 42
    skip_install: bool = False
    semantic_alignment: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: list(ALL_LANGUAGES))
    model_path: str = "./models/mdeberta-pii-safe/final"
    llm_source: str = "openai"
    llm_model: str = "gpt-4o"
    device: str = "auto"
    batch_llm: bool = True  # Batch async LLM calls (fast for OpenAI, disable for local LLMs)
    batch_llm_concurrency: int = 50  # Max concurrent async LLM requests
    session_dir: Optional[str] = None  # Override timestamped session dir (for multi-model runs)
    span_prompt_version: str = "V14_SPAN"  # Span routing prompt version (V14_SPAN, V15_SPAN, V16_SPAN)

    def get_samples_for_dataset(self, dataset_name: str) -> int:
        """Return effective sample count: CLI override > per-dataset default."""
        if self.samples > 0:
            return self.samples
        return DATASET_DEFAULT_SAMPLES.get(dataset_name, 0)


def parse_args(argv: Optional[List[str]] = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="NER PII Benchmark — compare systems across datasets"
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="all",
        help=f"Comma-separated list of systems to test, or 'all'. Options: {', '.join(ALL_SYSTEMS)}",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=f"Comma-separated list of datasets, or 'all'. Options: {', '.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Max samples per dataset (0 = use per-dataset defaults: ai4privacy=full, nvidia=3000, wikineural=full)",
    )
    parser.add_argument("--output-dir", type=str, default="./experiments")
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Use a fixed session directory instead of ./experiments/{timestamp}. "
             "Useful to consolidate multiple model runs into a single summary.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--semantic-alignment",
        type=str,
        default=None,
        help="Path to semantic alignment JSON for Tier 2 evaluation",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="all",
        help="Comma-separated language codes for WikiNeural (default: all 8)",
    )
    parser.add_argument("--model-path", type=str, default="./models/mdeberta-pii-safe/final")
    parser.add_argument("--llm-source", type=str, default="openai", choices=["openai", "ollama"])
    parser.add_argument("--llm-model", type=str, default="gpt-4o")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--batch-llm",
        action="store_true",
        default=True,
        help="Batch LLM calls with async concurrency (default: True, fast for OpenAI)",
    )
    parser.add_argument(
        "--no-batch-llm",
        action="store_true",
        help="Disable LLM batching (use for local LLMs like Ollama)",
    )
    parser.add_argument(
        "--batch-llm-concurrency",
        type=int,
        default=50,
        help="Max concurrent async LLM requests (default: 50)",
    )
    parser.add_argument(
        "--span-prompt-version",
        type=str,
        default="V14_SPAN",
        choices=["V14_SPAN", "V15_SPAN", "V16_SPAN"],
        help="Span routing prompt version (default: V14_SPAN). V16_SPAN enables extended NVIDIA label set.",
    )

    args = parser.parse_args(argv)

    systems = ALL_SYSTEMS if args.systems == "all" else [s.strip() for s in args.systems.split(",")]
    datasets = ALL_DATASETS if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]
    languages = ALL_LANGUAGES if args.languages == "all" else [l.strip() for l in args.languages.split(",")]

    batch_llm = not args.no_batch_llm

    return BenchmarkConfig(
        systems=systems,
        datasets=datasets,
        samples=args.samples,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        runs=args.runs,
        seed=args.seed,
        skip_install=args.skip_install,
        semantic_alignment=args.semantic_alignment,
        languages=languages,
        model_path=args.model_path,
        llm_source=args.llm_source,
        llm_model=args.llm_model,
        device=args.device,
        batch_llm=batch_llm,
        batch_llm_concurrency=args.batch_llm_concurrency,
        session_dir=args.session_dir,
        span_prompt_version=args.span_prompt_version,
    )
