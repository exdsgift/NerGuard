#!/usr/bin/env bash
# Setup NerGuard environment
set -euo pipefail

echo "Installing dependencies..."
uv sync

echo "Downloading spaCy model for benchmarks..."
uv run python -m spacy download en_core_web_lg

echo "Setup complete."
