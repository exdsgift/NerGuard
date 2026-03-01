#!/usr/bin/env bash
# Cross-lingual evaluation on WikiNeural (8 European languages)
# Usage: ./scripts/evaluate_multilingual.sh [--max-samples 1000]
set -euo pipefail

uv run python -m src.evaluation.multilingual "$@"
