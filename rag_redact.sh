#!/usr/bin/env bash
# NerGuard RAG — PII redaction with typed placeholders for pipeline integration.
# Shortcut for: nerguard --rag [TEXT] [OPTIONS]
#
# Usage:
#   ./rag_redact.sh "Hi, I'm John Smith. Email: john@acme.com"
#   ./rag_redact.sh -f document.txt --mapping
#   ./rag_redact.sh "..." --generic   # [PII] instead of typed labels
#   ./rag_redact.sh "..." --llm --backend ollama --model qwen2.5:7b
#
# Run 'nerguard --help' for all options.
set -euo pipefail
cd "$(dirname "$0")"

export TF_CPP_MIN_LOG_LEVEL=3
unset VIRTUAL_ENV 2>/dev/null || true

uv run nerguard --rag "$@"
