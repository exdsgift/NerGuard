#!/usr/bin/env python
"""
nerguard — PII detection and redaction CLI.

Usage:
    nerguard "Hi, I'm John Smith. Email: john@acme.com"
    nerguard "..." --rag
    nerguard "..." --json
    echo "John Smith" | nerguard --rag
    nerguard -f document.txt --rag --mapping
    nerguard "..." --llm --backend ollama --model qwen2.5:7b
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

from src.core.constants import DEFAULT_MODEL_PATH
from src.utils.colors import Colors

_VERSION = "1.0.0"

_EPILOG = """
examples:
  nerguard "Hi, I'm John Smith. Email: john@acme.com"
  nerguard "..." --rag
  nerguard "..." --rag --mapping
  nerguard "..." --json
  nerguard "..." --generic
  echo "John Smith" | nerguard --rag
  nerguard -f report.txt --rag
  nerguard "..." --llm --backend ollama --model qwen2.5:7b
  nerguard "..." --llm --backend openai --model gpt-4o-mini
"""


def _read_input(args) -> str:
    if args.text:
        return args.text
    if args.file:
        with open(args.file) as f:
            return f.read().strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None


def _print_human(text: str, entities: List[Dict], redacted: str) -> None:
    print(f"\n{Colors.BOLD}Input:{Colors.ENDC}")
    print(f'  "{text}"')

    print(f"\n{Colors.BOLD}Detected PII:{Colors.ENDC}")
    if not entities:
        print(f"  {Colors.DIM}(none){Colors.ENDC}")
    else:
        max_label = max(len(e["label"]) for e in entities)
        max_text = max(len(e["text"]) for e in entities)
        for e in entities:
            label = f"{Colors.OKCYAN}{e['label']:<{max_label}}{Colors.ENDC}"
            padding = max_text - len(e["text"])
            print(
                f"  {label} \u2192 \"{Colors.BOLD}{e['text']}{Colors.ENDC}\""
                f"{' ' * padding}    {Colors.DIM}[{e['source']:<16s} conf: {e['confidence']:.3f}]{Colors.ENDC}"
            )

    print(f"\n{Colors.BOLD}Redacted:{Colors.ENDC}")
    print(f'  "{redacted}"')
    print()


def _print_rag(text: str, result, show_mapping: bool) -> None:
    print(f'\nInput:    "{text}"')
    print(f'Redacted: "{result.text}"')
    if result.entities:
        print("\nEntities:")
        for e in result.entities:
            print(f"  [{e['label']}] {e['text']!r}  conf={e['confidence']:.3f}  src={e['source']}")
    if show_mapping and result.mapping:
        print("\nMapping:")
        for k, v in result.mapping.items():
            print(f"  {k}: {v!r}")
    print()


def _run_human(text, model_path, llm, backend, model):
    from src.scripts.redact import redact_pipeline
    entities, redacted = redact_pipeline(
        text=text, model_path=model_path,
        llm_routing=llm, llm_source=backend, llm_model=model,
    )
    _print_human(text, entities, redacted)


def _run_rag(text, model_path, llm, backend, model, typed, show_mapping):
    from src.rag.redactor import nerguard as NerGuardRAG
    result = NerGuardRAG(
        model_path=model_path, llm_routing=llm,
        llm_source=backend, llm_model=model, typed=typed,
    ).redact(text)
    _print_rag(text, result, show_mapping)


def _run_json(text, model_path, llm, backend, model):
    from src.scripts.redact import redact_pipeline
    entities, redacted = redact_pipeline(
        text=text, model_path=model_path,
        llm_routing=llm, llm_source=backend, llm_model=model,
    )
    print(json.dumps({"input": text, "entities": entities, "redacted": redacted}, indent=2, ensure_ascii=False))


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="nerguard",
        description="Detect and redact PII using entropy-gated hybrid NER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )

    # Input
    parser.add_argument("text", nargs="?", metavar="TEXT",
                        help="Text to redact (or use -f / stdin)")
    parser.add_argument("-f", "--file", metavar="PATH",
                        help="Read input from file")

    # Output format (mutually exclusive flags)
    fmt_group = parser.add_mutually_exclusive_group()
    fmt_group.add_argument("--rag", action="store_true",
                           help="RAG output: typed placeholders [NAME] [EMAIL] (default: block chars)")
    fmt_group.add_argument("--json", action="store_true", dest="as_json",
                           help="JSON output with full entity metadata")
    fmt_group.add_argument("--generic", action="store_true",
                           help="Generic output: compact [PII] marker, max token savings")

    parser.add_argument("--mapping", action="store_true",
                        help="Show entity→placeholder map (with --rag or --generic)")

    # LLM routing
    parser.add_argument("--llm", action="store_true",
                        help="Enable entropy-gated LLM routing for uncertain spans")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                        metavar="{openai,ollama}",
                        help="LLM backend (default: openai)")
    parser.add_argument("--model", default="gpt-4o", metavar="NAME",
                        help="LLM model name (default: gpt-4o)")

    # Other
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, metavar="PATH",
                        help="NER model path or HuggingFace ID (auto-downloads if not found)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress messages")
    parser.add_argument("-v", "--version", action="version", version=f"nerguard {_VERSION}")

    args = parser.parse_args()

    text = _read_input(args)
    if not text:
        parser.print_help()
        sys.exit(2)

    if not args.quiet:
        print(f"{Colors.DIM}Loading model...{Colors.ENDC}", file=sys.stderr)

    if args.rag:
        _run_rag(text, args.model_path, args.llm, args.backend, args.model,
                 typed=True, show_mapping=args.mapping)
    elif args.generic:
        _run_rag(text, args.model_path, args.llm, args.backend, args.model,
                 typed=False, show_mapping=args.mapping)
    elif args.as_json:
        _run_json(text, args.model_path, args.llm, args.backend, args.model)
    else:
        _run_human(text, args.model_path, args.llm, args.backend, args.model)


if __name__ == "__main__":
    main()
