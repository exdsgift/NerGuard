#!/usr/bin/env bash
# Run NerGuard PII detection with source tracking and redaction.
# Usage: ./redact.sh --text "Dear John Smith, your SSN is 555-01-4433"
#        ./redact.sh --file input.txt --llm --json
set -euo pipefail
cd "$(dirname "$0")"

# Suppress noisy warnings
export TF_CPP_MIN_LOG_LEVEL=3
unset VIRTUAL_ENV 2>/dev/null || true

uv run python -m src.scripts.redact "$@"
