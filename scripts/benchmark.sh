#!/usr/bin/env bash
# Run benchmark comparison (NerGuard vs GLiNER, Presidio, SpaCy)
# Usage: ./scripts/benchmark.sh
set -euo pipefail

uv run python -m src.evaluation.benchmark
