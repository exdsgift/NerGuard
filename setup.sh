#!/usr/bin/env bash
# NerGuard setup script — run this once after cloning the repo.
# It installs dependencies, configures API keys, and verifies your environment.
set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
DIM='\033[2m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}   ▄   ▄███▄   █▄▄▄▄   ▄▀    ▄   ██   █▄▄▄▄ ██▄   ${NC}"
echo -e "${CYAN}${BOLD}    █  █▀   ▀  █  ▄▀ ▄▀       █  █ █  █  ▄▀ █  █  ${NC}"
echo -e "${CYAN}${BOLD}██   █ ██▄▄    █▀▀▌  █ ▀▄  █   █ █▄▄█ █▀▀▌  █   █ ${NC}"
echo -e "${CYAN}${BOLD}█ █  █ █▄   ▄▀ █  █  █   █ █   █ █  █ █  █  █  █  ${NC}"
echo -e "${CYAN}${BOLD}█  █ █ ▀███▀     █    ███  █▄ ▄█    █   █   ███▀   ${NC}"
echo -e "${CYAN}${BOLD}█   ██          ▀           ▀▀▀    █   ▀            ${NC}"
echo -e "${CYAN}${BOLD}                                  ▀                 ${NC}"
echo -e "${DIM}  Entropy-Gated Hybrid NER · PII Detection · Setup${NC}"
echo ""

# ── Step 1: Python version ────────────────────────────────────────────────────
info "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    error "Python 3 not found. Install Python 3.11+ and try again."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    error "Python 3.11+ required, found $PY_VERSION."
    exit 1
fi
success "Python $PY_VERSION detected."

# ── Step 2: uv ────────────────────────────────────────────────────────────────
info "Checking uv package manager..."
if ! command -v uv &>/dev/null; then
    error "'uv' not found. Install it with:"
    echo "       curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "    Then re-run this script."
    exit 1
fi
success "uv $(uv --version | awk '{print $2}') detected."

# ── Step 3: Install dependencies ─────────────────────────────────────────────
info "Installing Python dependencies (uv sync)..."
uv sync --quiet
success "Dependencies installed."

# ── Step 3b: Install nerguard-rag package ────────────────────────────────────
info "Installing nerguard-rag package..."
uv pip install -e ./nerguard_rag --quiet
success "nerguard-rag installed."

# ── Step 4: Model ─────────────────────────────────────────────────────────────
info "Checking NER model..."
MODEL_DIR="./models/mdeberta-pii-safe/final"
if [[ -d "$MODEL_DIR" ]]; then
    success "Model found at $MODEL_DIR."
else
    warn "Model not found at $MODEL_DIR."
    echo -e "  ${DIM}→ NerGuard will automatically download it from HuggingFace${NC}"
    echo -e "  ${DIM}  (exdsgift/NerGuard-0.3B) on first run. ~300 MB, cached by HF.${NC}"
fi

# ── Step 5: API key configuration ────────────────────────────────────────────
info "Configuring API keys..."
if [[ -f ".env" ]]; then
    success ".env already exists — skipping API key setup."
else
    echo ""
    echo -e "${BOLD}LLM Backend Selection${NC}"
    echo -e "NerGuard supports two LLM backends for hybrid routing:"
    echo -e "  ${BOLD}A)${NC} OpenAI (gpt-4o, gpt-4o-mini) — requires API key"
    echo -e "  ${BOLD}B)${NC} Ollama (qwen2.5:7b) — free, runs locally, no API key needed"
    echo ""

    read -r -p "Do you want to add an OpenAI API key? [y/N] " USE_OPENAI
    USE_OPENAI="${USE_OPENAI:-N}"

    if [[ "$USE_OPENAI" =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${DIM}Get your key at: https://platform.openai.com/api-keys${NC}"
        read -r -p "Paste your OPENAI_API_KEY (or press Enter to skip): " OPENAI_KEY
        if [[ -n "$OPENAI_KEY" ]]; then
            echo "OPENAI_API_KEY='$OPENAI_KEY'" > .env
            success ".env created with OPENAI_API_KEY."
        else
            touch .env
            warn "No key entered — .env created empty. You can add it later."
        fi
    else
        touch .env
        info ".env created empty (no OpenAI key configured)."
    fi
fi

# ── Step 6: Ollama reminder ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD} Local Inference with Ollama (optional)${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if command -v ollama &>/dev/null; then
    success "Ollama is installed ($(ollama --version 2>/dev/null | head -1))."
    echo -e "  ${DIM}Pull the recommended model if you haven't yet:${NC}"
    echo -e "  ${DIM}  ollama pull qwen2.5:7b${NC}"
else
    warn "Ollama is not installed."
    echo -e "  ${DIM}To use local inference (zero API cost, ~5 GB VRAM):${NC}"
    echo -e "  ${DIM}  1. Install: https://ollama.com/download${NC}"
    echo -e "  ${DIM}  2. Pull model: ollama pull qwen2.5:7b${NC}"
    echo -e "  ${DIM}  3. Use: --llm-source ollama --llm-model qwen2.5:7b${NC}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD} Setup complete! Next steps:${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}# Redact PII from text${NC}"
echo -e "  ${CYAN}nerguard \"Hi, I'm John Smith. Email: john@acme.com\"${NC}"
echo ""
echo -e "  ${BOLD}# RAG-optimized redaction (typed placeholders)${NC}"
echo -e "  ${CYAN}nerguard \"...\" --format rag${NC}"
echo ""
echo -e "  ${BOLD}# With LLM routing (OpenAI)${NC}"
echo -e "  ${CYAN}nerguard \"...\" --llm --backend openai --model gpt-4o${NC}"
echo ""
echo -e "  ${BOLD}# With local Ollama inference${NC}"
echo -e "  ${CYAN}nerguard \"...\" --llm --backend ollama --model qwen2.5:7b${NC}"
echo ""
echo -e "  ${BOLD}# All formats and options${NC}"
echo -e "  ${CYAN}nerguard --help${NC}"
echo ""
