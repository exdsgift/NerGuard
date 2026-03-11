#!/usr/bin/env bash
# Publish NerGuard to PyPI.
#
# Prerequisites:
#   1. Create a PyPI account at https://pypi.org/account/register/
#   2. Generate an API token at https://pypi.org/manage/account/token/
#   3. Set the token:  export UV_PUBLISH_TOKEN="pypi-..."
#      (or add it to ~/.pypirc)
#
# Usage:
#   export UV_PUBLISH_TOKEN="pypi-..."
#   ./publish.sh              # build + publish
#   ./publish.sh --dry-run    # build only, no upload
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

DRY_RUN=false
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Check token ───────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == false ]]; then
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
        error "UV_PUBLISH_TOKEN is not set. Export it or use --dry-run.\n  Get a token at: https://pypi.org/manage/account/token/"
    fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── nerguard ─────────────────────────────────────${NC}"

info "Building..."
rm -rf dist/
uv build --out-dir dist/
success "Built: $(ls dist/nerguard-*.whl dist/nerguard-*.tar.gz 2>/dev/null | tail -2 | tr '\n' ' ')"

# ── Publish ───────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == false ]]; then
    info "Publishing to PyPI..."
    uv publish dist/nerguard-*.whl dist/nerguard-*.tar.gz
    success "Published."
else
    info "Dry run — skipping upload."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Done! Install with:${NC}"
echo -e "  pip install nerguard"
echo -e ""
echo -e "${BOLD}CLI commands:${NC}"
echo -e "  nerguard     \"John Smith, email john@acme.com\""
echo -e "  nerguard-rag \"John Smith, email john@acme.com\"   # defaults to --rag output"
echo ""
