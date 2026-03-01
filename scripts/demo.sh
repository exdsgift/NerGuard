#!/usr/bin/env bash
# Interactive NerGuard demo
# Usage: ./scripts/demo.sh [--text "Contact John at john@email.com"] [--llm-routing]
set -euo pipefail

uv run python -m src.scripts.demo "$@"
