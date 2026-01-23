#!/usr/bin/env python3
"""
NerGuard Demo Script

This script demonstrates how to use NerGuard for PII detection.
Supports loading models from local paths or Hugging Face Hub.

Usage:
    python -m src.scripts.demo
    python -m src.scripts.demo --model "your-username/nerguard-model"
    python -m src.scripts.demo --llm-routing --text "John Smith lives at 123 Main St."
"""

import argparse
import sys
from typing import Optional

from src.core.constants import DEFAULT_MODEL_PATH
from src.inference.tester import PIITester
from src.utils.samples import get_sample, list_samples
from src.utils.colors import Colors


def print_header():
    """Print demo header."""
    print(Colors.bold("\n" + "=" * 60))
    print(Colors.bold("  NerGuard - Personal Information Detection Demo"))
    print(Colors.bold("=" * 60 + "\n"))


def print_entities(entities: list, text: str):
    """Print detected entities in a formatted way."""
    if not entities:
        print(Colors.warning("  No PII entities detected.\n"))
        return

    print(Colors.success(f"  Found {len(entities)} PII entities:\n"))

    for entity in entities:
        label_color = Colors.OKCYAN
        print(f"  {label_color}[{entity['label']}]{Colors.ENDC} ", end="")
        print(f"'{Colors.bold(entity['text'])}' ", end="")
        print(f"(confidence: {entity['confidence']:.2%})")

    print()


def demo_basic(tester: PIITester, text: str):
    """Run basic entity extraction demo."""
    print(Colors.info("Basic Entity Extraction"))
    print("-" * 40)
    print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}\n")

    entities = tester.get_entities(text)
    print_entities(entities, text)


def demo_redaction(tester: PIITester, text: str):
    """Run text redaction demo."""
    print(Colors.info("Text Redaction"))
    print("-" * 40)
    print(f"Original:\n  {text[:200]}{'...' if len(text) > 200 else ''}\n")

    redacted = tester.redact_text(text)
    print(f"Redacted:\n  {redacted[:200]}{'...' if len(redacted) > 200 else ''}\n")


def demo_detailed(tester: PIITester, text: str):
    """Run detailed token analysis demo."""
    print(Colors.info("Detailed Token Analysis"))
    print("-" * 40)
    print(f"{'Token':<20} {'Label':<20} {'Conf':>8} {'Entropy':>8} {'Source':<15}")
    print("-" * 75)

    results = tester.analyze_text(text[:500], verbose=False)

    for r in results:
        if r["label"] != "O":
            label_str = Colors.OKGREEN + r["label"] + Colors.ENDC
        else:
            label_str = r["label"]

        token_display = r["token"][:18] if len(r["token"]) > 18 else r["token"]
        print(f"{token_display:<20} {label_str:<20} {r['confidence']:>8.4f} {r['entropy']:>8.4f} {r['source']:<15}")

    print()


def run_demo(
    model_path: str = DEFAULT_MODEL_PATH,
    text: Optional[str] = None,
    sample_name: str = "simple",
    llm_routing: bool = False,
    llm_source: str = "openai",
    detailed: bool = False,
):
    """
    Run the NerGuard demo.

    Args:
        model_path: Path to model (local or HuggingFace Hub ID)
        text: Custom text to analyze (uses sample if None)
        sample_name: Name of built-in sample to use
        llm_routing: Enable LLM disambiguation
        llm_source: LLM backend ("openai" or "ollama")
        detailed: Show detailed token analysis
    """
    print_header()

    # Model info
    print(Colors.info("Configuration"))
    print("-" * 40)
    print(f"  Model: {model_path}")
    print(f"  LLM Routing: {llm_routing}")
    if llm_routing:
        print(f"  LLM Source: {llm_source}")
    print()

    # Initialize tester
    print(Colors.info("Loading Model..."))
    try:
        tester = PIITester(
            model_path=model_path,
            llm_routing=llm_routing,
            llm_source=llm_source,
        )
        print(Colors.success("  Model loaded successfully!\n"))
    except Exception as e:
        print(Colors.error(f"  Failed to load model: {e}"))
        sys.exit(1)

    # Get text
    if text is None:
        text = get_sample(sample_name)
        print(f"Using sample: '{sample_name}'\n")

    # Run demos
    demo_basic(tester, text)
    demo_redaction(tester, text)

    if detailed:
        demo_detailed(tester, text)

    print(Colors.bold("=" * 60))
    print(Colors.success("Demo completed successfully!"))
    print(Colors.bold("=" * 60 + "\n"))


def main():
    parser = argparse.ArgumentParser(
        description="NerGuard PII Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.demo
  python -m src.scripts.demo --text "Contact John at john@email.com"
  python -m src.scripts.demo --model "username/model" --llm-routing
  python -m src.scripts.demo --sample report --detailed
        """,
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL_PATH,
        help="Model path or HuggingFace Hub ID (default: local model)",
    )
    parser.add_argument(
        "--text", "-t",
        default=None,
        help="Custom text to analyze",
    )
    parser.add_argument(
        "--sample", "-s",
        default="simple",
        choices=list_samples(),
        help="Built-in sample to use (default: simple)",
    )
    parser.add_argument(
        "--llm-routing",
        action="store_true",
        help="Enable LLM disambiguation for uncertain predictions",
    )
    parser.add_argument(
        "--llm-source",
        default="openai",
        choices=["openai", "ollama"],
        help="LLM backend to use (default: openai)",
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed token-level analysis",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available sample texts and exit",
    )

    args = parser.parse_args()

    if args.list_samples:
        print("\nAvailable samples:")
        for name in list_samples():
            sample = get_sample(name)
            preview = sample[:60] + "..." if len(sample) > 60 else sample
            print(f"  {name}: {preview}")
        print()
        return

    run_demo(
        model_path=args.model,
        text=args.text,
        sample_name=args.sample,
        llm_routing=args.llm_routing,
        llm_source=args.llm_source,
        detailed=args.detailed,
    )


if __name__ == "__main__":
    main()
