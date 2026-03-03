#!/usr/bin/env python
"""
Evaluation entry point for NerGuard.

Usage:
    python -m src.scripts.evaluate
    python -m src.scripts.evaluate --no-llm --sample-limit 500
"""

import argparse
from src.training.validator import evaluate
from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NerGuard model")

    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM routing (baseline only)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1000,
        help="Maximum samples to evaluate (0 for all)",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=DEFAULT_ENTROPY_THRESHOLD,
        help="Entropy threshold for LLM routing",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for LLM routing",
    )
    parser.add_argument(
        "--llm-source",
        choices=["openai", "ollama"],
        default=None,
        help="Explicitly choose LLM backend (default: auto-detect)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        llm_routing=not args.no_llm,
        sample_limit=args.sample_limit if args.sample_limit > 0 else None,
        entropy_threshold=args.entropy_threshold,
        confidence_threshold=args.confidence_threshold,
        llm_source=args.llm_source,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    main()
