#!/bin/bash
#
# NerGuard Diagram Converter
# Converts all Mermaid (.mmd) diagrams to PDF
#
# Usage:
#   ./convert.sh              # Convert all diagrams
#   ./convert.sh --clean      # Clean and convert
#   ./convert.sh --svg        # Also keep SVG files
#   ./convert.sh --help       # Show help
#
# Requirements:
#   - Python 3.8+
#   - uv pip install cairosvg (for SVG to PDF conversion)
#   - Optional: npm install -g @mermaid-js/mermaid-cli (Node.js >=18 required)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
NerGuard Diagram Converter

Converts all Mermaid (.mmd) diagrams to PDF format.

Usage:
    $(basename "$0") [OPTIONS]

Options:
    --clean     Clean output directories before converting
    --svg       Also keep SVG intermediate files
    --method    Conversion method: auto, mmdc, or kroki (default: auto)
    --help      Show this help message

Examples:
    $(basename "$0")                    # Basic conversion
    $(basename "$0") --clean            # Clean and convert
    $(basename "$0") --clean --svg      # Clean, convert, keep SVGs

Requirements:
    Recommended (best quality):
        npm install -g @mermaid-js/mermaid-cli

    Fallback (uses kroki.io API):
        pip install cairosvg

    For best PDF quality, also install one of:
        - inkscape
        - librsvg2-bin (provides rsvg-convert)

EOF
}

# Check Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found"
        exit 1
    fi
}

# Check mmdc
check_mmdc() {
    if command -v mmdc &> /dev/null; then
        print_status "Mermaid CLI (mmdc) found - using high-quality rendering"
        return 0
    else
        print_warning "Mermaid CLI not found - using kroki.io API (may have rendering issues)"
        print_warning "For best results, install: npm install -g @mermaid-js/mermaid-cli"
        return 1
    fi
}

# Install Python dependencies if missing
check_python_deps() {
    python3 -c "import cairosvg" 2>/dev/null || {
        print_warning "cairosvg not installed - some PDF conversions may fail"
        print_warning "Install with: pip install cairosvg"
    }
}

# Main
main() {
    # Parse arguments
    ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --clean|--svg|--method)
                ARGS="$ARGS $1"
                if [[ "$1" == "--method" && -n "$2" ]]; then
                    ARGS="$ARGS $2"
                    shift
                fi
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    print_status "NerGuard Diagram Converter"
    echo ""

    # Check requirements
    check_python
    check_mmdc || true
    check_python_deps

    echo ""

    # Run Python converter
    cd "$SCRIPT_DIR"
    python3 convert_diagrams.py $ARGS

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        print_status "Conversion complete!"
        print_status "PDF files are in: $SCRIPT_DIR/pdf_output/"
    else
        print_error "Some conversions failed. Check the output above."
    fi

    exit $exit_code
}

main "$@"
