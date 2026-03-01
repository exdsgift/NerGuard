#!/usr/bin/env bash
# Run hybrid evaluation (baseline vs DeBERTa + LLM routing)
# Usage: ./scripts/evaluate.sh [--no-llm] [--sample-limit N]
set -euo pipefail

uv run python -m src.scripts.evaluate "$@"
