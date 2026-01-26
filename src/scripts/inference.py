#!/usr/bin/env python
"""
Inference entry point for NerGuard.

Usage:
    python -m src.scripts.inference --text "Dear John Smith, your SSN is 555-01-4433"
    python -m src.scripts.inference --file input.txt --output results.json
    python -m src.scripts.inference --interactive
"""

import argparse
import json
import sys

from src.inference.tester import PIITester
from src.core.constants import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.utils.samples import SAMPLE_REPORT


def main():
    parser = argparse.ArgumentParser(description="Run NerGuard PII detection")

    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM routing",
    )
    parser.add_argument(
        "--llm-source",
        type=str,
        default="openai",
        choices=["openai", "ollama"],
        help="LLM backend",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Output redacted text instead of entities",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with demo sample text",
    )

    args = parser.parse_args()

    # Initialize tester
    print(f"Loading model from {args.model_path}...")
    tester = PIITester(
        model_path=args.model_path,
        llm_routing=args.llm,
        llm_source=args.llm_source,
        entropy_threshold=DEFAULT_ENTROPY_THRESHOLD,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    def process_text(text: str):
        """Process text and display results."""
        if args.redact:
            result = tester.redact_text(text)
            print("\n--- Redacted Text ---")
            print(result)
        else:
            entities = tester.get_entities(text)
            print(f"\n--- Found {len(entities)} entities ---")
            for e in entities:
                print(f"  [{e['label']}] \"{e['text']}\" (conf: {e['confidence']:.2%})")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(entities, f, indent=2)
                print(f"\nResults saved to {args.output}")

    # Determine input source
    if args.demo:
        print("\n--- Demo Mode ---")
        process_text(SAMPLE_REPORT)

    elif args.interactive:
        print("\n--- Interactive Mode (Ctrl+C to exit) ---")
        while True:
            try:
                text = input("\nEnter text: ").strip()
                if text:
                    process_text(text)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    elif args.text:
        process_text(args.text)

    elif args.file:
        with open(args.file, "r") as f:
            text = f.read()
        process_text(text)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
