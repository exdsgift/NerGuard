#!/usr/bin/env bash
# Run PII detection on text input
# Usage: ./scripts/inference.sh --text "..." [--llm] [--redact] [--interactive]
set -euo pipefail

uv run python -m src.scripts.inference "$@"
