#!/usr/bin/env bash
# Publish NerGuard packages to PyPI.
#
# Prerequisites:
#   1. Create a PyPI account at https://pypi.org/account/register/
#   2. Generate an API token at https://pypi.org/manage/account/token/
#   3. Set the token:  export UV_PUBLISH_TOKEN="pypi-..."
#      (or add it to ~/.pypirc)
#
# Usage:
#   export UV_PUBLISH_TOKEN="pypi-..."
#   ./publish.sh              # publish both packages
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

# ── Check token ───────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == false ]]; then
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
        error "UV_PUBLISH_TOKEN is not set. Export it or use --dry-run.\n  Get a token at: https://pypi.org/manage/account/token/"
    fi
fi

# ── Publish nerguard (main package) ──────────────────────────────────────────
echo ""
echo -e "${BOLD}── nerguard (main package) ──────────────────────${NC}"
cd "$ROOT"

info "Building nerguard..."
uv build --out-dir dist/
success "Built: $(ls dist/nerguard-*.whl dist/nerguard-*.tar.gz 2>/dev/null | tail -2 | tr '\n' ' ')"

if [[ "$DRY_RUN" == false ]]; then
    info "Publishing nerguard to PyPI..."
    uv publish dist/nerguard-*.whl dist/nerguard-*.tar.gz
    success "nerguard published."
else
    info "Dry run — skipping upload for nerguard."
fi

# ── Publish nerguard-rag ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}── nerguard-rag ─────────────────────────────────${NC}"
cd "$ROOT/nerguard_rag"

info "Building nerguard-rag..."
uv build --out-dir dist/
success "Built: $(ls dist/nerguard_rag-*.whl dist/nerguard_rag-*.tar.gz 2>/dev/null | tail -2 | tr '\n' ' ')"

if [[ "$DRY_RUN" == false ]]; then
    info "Publishing nerguard-rag to PyPI..."
    uv publish dist/nerguard_rag-*.whl dist/nerguard_rag-*.tar.gz
    success "nerguard-rag published."
else
    info "Dry run — skipping upload for nerguard-rag."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Done! Users can now install with:${NC}"
echo -e "  pip install nerguard"
echo -e "  pip install nerguard-rag"
echo ""
