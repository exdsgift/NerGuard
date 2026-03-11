#!/usr/bin/env bash
# NerGuard interactive test REPL — model loaded once, stays hot in memory.
#
# Usage:
#   ./test.sh
#
# REPL commands:
#   <text>                         redact free text in the current mode
#   /mode [human|rag|json|generic] change output mode (no arg: cycle through)
#   /llm                           toggle LLM routing on/off
#   /backend [openai|ollama]       switch LLM backend
#   /model NAME                    set model (e.g. qwen2.5:7b, gpt-4o-mini)
#   /labels                        toggle entity label rows on/off
#   /mapping                       toggle entity→value map (rag/generic only)
#   /file PATH                     redact an entire file
#   /clear                         clear the screen
#   /help                          show commands inside the REPL
#   /quit  or  Ctrl+D              exit
#   Ctrl+C                         cancel current input, stay in REPL
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
unset VIRTUAL_ENV 2>/dev/null || true
exec uv run python -m src.scripts.repl
