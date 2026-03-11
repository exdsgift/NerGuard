#!/usr/bin/env bash
# NerGuard — PII detection and redaction.
# Shortcut for: nerguard [TEXT] [OPTIONS]
#
# Usage:
#   ./redact.sh "Hi, I'm John Smith. Email: john@acme.com"
#   ./redact.sh -f document.txt
#   ./redact.sh "..." --llm --backend ollama --model qwen2.5:7b
#   ./redact.sh "..." --format json
#
# Run 'nerguard --help' for all options.
set -euo pipefail
cd "$(dirname "$0")"

export TF_CPP_MIN_LOG_LEVEL=3
unset VIRTUAL_ENV 2>/dev/null || true

uv run nerguard "$@"
