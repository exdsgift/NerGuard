#!/usr/bin/env bash
# Train NerGuard model
# Usage: ./scripts/train.sh [--epochs N] [--batch-size N] [--no-wandb]
set -euo pipefail

uv run python -m src.scripts.train "$@"
