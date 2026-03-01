#!/usr/bin/env bash
# Run LLM routing ablation study
# Usage: ./scripts/ablation_study.sh [--max-samples 500]
set -euo pipefail

uv run python -m src.evaluation.ablation_study "$@"
