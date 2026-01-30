#!/usr/bin/env python3
"""
Unified Mermaid diagram to PDF converter.

This script converts all .mmd (Mermaid) files to PDF with proper rendering
of fonts, special characters, and mathematical symbols.

Usage:
    python convert_diagrams.py           # Convert all diagrams
    python convert_diagrams.py --clean   # Clean output directories first
    python convert_diagrams.py --svg     # Also keep SVG files

Requirements:
    - Node.js and npm (for Mermaid CLI)
    - @mermaid-js/mermaid-cli: npm install -g @mermaid-js/mermaid-cli

    Or fallback to kroki.io API (no dependencies, but may have rendering issues)
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import base64
import zlib


# Configuration
SCRIPT_DIR = Path(__file__).parent
SVG_OUTPUT_DIR = SCRIPT_DIR / "svg_output"
PDF_OUTPUT_DIR = SCRIPT_DIR / "pdf_output"
CONFIG_FILE = SCRIPT_DIR / "mermaid-config.json"

SUBDIRS = ["paper", "process_diagrams", "results"]


def check_mmdc_installed() -> bool:
    """Check if Mermaid CLI (mmdc) is installed."""
    try:
        result = subprocess.run(
            ["mmdc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_puppeteer_config() -> Path:
    """Create puppeteer config for better font handling."""
    config_path = SCRIPT_DIR / "puppeteer-config.json"
    config = {
        "executablePath": "",  # Use system Chrome if available
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--font-render-hinting=none"
        ]
    }

    # Try to find Chrome/Chromium
    chrome_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]

    for path in chrome_paths:
        if Path(path).exists():
            config["executablePath"] = path
            break

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def convert_with_mmdc(mmd_path: Path, output_path: Path, output_format: str = "pdf") -> bool:
    """Convert Mermaid file using mmdc (Mermaid CLI)."""
    puppeteer_config = check_puppeteer_config()

    cmd = [
        "mmdc",
        "-i", str(mmd_path),
        "-o", str(output_path),
        "-b", "white",
        "-t", "neutral",
        "--scale", "2",  # Higher quality
    ]

    # Add config file if exists
    if CONFIG_FILE.exists():
        cmd.extend(["-c", str(CONFIG_FILE)])

    # Add puppeteer config
    cmd.extend(["-p", str(puppeteer_config)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            print(f"    mmdc error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("    mmdc timeout")
        return False
    except Exception as e:
        print(f"    mmdc exception: {e}")
        return False


def convert_with_mermaid_ink(mmd_path: Path, svg_path: Path) -> bool:
    """Convert Mermaid file using mermaid.ink API."""
    try:
        with open(mmd_path, "r", encoding="utf-8") as f:
            content = f.read()

        # mermaid.ink uses base64 encoded diagram in URL
        encoded = base64.urlsafe_b64encode(content.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/svg/{encoded}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) NerGuard-Converter/1.0"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            svg_content = response.read()

        # Validate SVG
        if not svg_content.startswith(b"<svg"):
            print(f"    mermaid.ink error: Invalid SVG response")
            return False

        with open(svg_path, "wb") as f:
            f.write(svg_content)

        return True

    except urllib.error.URLError as e:
        print(f"    mermaid.ink error: {e}")
        return False
    except Exception as e:
        print(f"    mermaid.ink exception: {e}")
        return False


def convert_with_kroki(mmd_path: Path, svg_path: Path) -> bool:
    """Convert Mermaid file using kroki.io API (fallback)."""
    try:
        with open(mmd_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Compress and encode for GET request (more reliable than POST)
        compressed = zlib.compress(content.encode("utf-8"), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
        url = f"https://kroki.io/mermaid/svg/{encoded}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) NerGuard-Converter/1.0",
                "Accept": "image/svg+xml"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            svg_content = response.read()

        # Validate SVG
        if not svg_content.startswith(b"<svg"):
            print(f"    kroki error: Invalid SVG response")
            return False

        with open(svg_path, "wb") as f:
            f.write(svg_content)

        return True

    except urllib.error.URLError as e:
        print(f"    kroki error: {e}")
        return False
    except Exception as e:
        print(f"    kroki exception: {e}")
        return False


def svg_to_pdf_playwright(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using Playwright (browser-based, best foreignObject support)."""
    try:
        from playwright.sync_api import sync_playwright

        # Read SVG content
        svg_content = svg_path.read_text(encoding='utf-8')

        # Create minimal HTML wrapper - no whitespace to prevent extra content
        html_content = f"""<!DOCTYPE html><html><head><style>@page{{margin:0;size:auto}}*{{margin:0;padding:0;box-sizing:border-box}}html,body{{margin:0;padding:0;background:white;width:fit-content;height:fit-content}}svg{{display:block}}</style></head><body><div style="padding:20px;display:inline-block">{svg_content}</div></body></html>"""

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content, wait_until='networkidle')

            # Get the wrapper div dimensions
            dimensions = page.evaluate('''() => {
                const wrapper = document.querySelector('div');
                if (!wrapper) return null;
                const rect = wrapper.getBoundingClientRect();
                return {
                    width: Math.ceil(rect.width) + 1,
                    height: Math.ceil(rect.height) + 1
                };
            }''')

            if dimensions:
                page.pdf(
                    path=str(pdf_path),
                    width=f"{dimensions['width']}px",
                    height=f"{dimensions['height']}px",
                    print_background=True,
                    page_ranges='1'
                )
            else:
                page.pdf(path=str(pdf_path), format='A4', print_background=True, page_ranges='1')

            browser.close()

        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except ImportError:
        return False
    except Exception as e:
        print(f"    playwright error: {e}")
        return False


def svg_to_pdf_svglib(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using svglib+reportlab (better text handling)."""
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF

        drawing = svg2rlg(str(svg_path))
        if drawing is None:
            return False
        renderPDF.drawToFile(drawing, str(pdf_path))
        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except ImportError:
        return False
    except Exception as e:
        print(f"    svglib error: {e}")
        return False


def svg_to_pdf_cairosvg(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using cairosvg."""
    try:
        import cairosvg
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except ImportError:
        print("    cairosvg not installed")
        return False
    except Exception as e:
        print(f"    cairosvg error: {e}")
        return False


def svg_to_pdf_inkscape(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using Inkscape (better font handling)."""
    try:
        result = subprocess.run(
            ["inkscape", str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0 and pdf_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def svg_to_pdf_rsvg(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using rsvg-convert (librsvg)."""
    try:
        result = subprocess.run(
            ["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0 and pdf_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    """Convert SVG to PDF using the best available method."""
    # Try methods in order of quality
    methods = [
        ("playwright", svg_to_pdf_playwright),
        ("inkscape", svg_to_pdf_inkscape),
        ("rsvg-convert", svg_to_pdf_rsvg),
        ("svglib", svg_to_pdf_svglib),
        ("cairosvg", svg_to_pdf_cairosvg),
    ]

    for name, method in methods:
        if method(svg_path, pdf_path):
            return True

    return False


def find_mermaid_files() -> list[Path]:
    """Find all .mmd files in the diagrams directory."""
    files = []
    for subdir in SUBDIRS:
        subdir_path = SCRIPT_DIR / subdir
        if subdir_path.exists():
            files.extend(subdir_path.glob("*.mmd"))
    return sorted(files)


def setup_output_dirs(clean: bool = False):
    """Create output directories."""
    for output_dir in [SVG_OUTPUT_DIR, PDF_OUTPUT_DIR]:
        if clean and output_dir.exists():
            shutil.rmtree(output_dir)

        for subdir in SUBDIRS:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Convert Mermaid diagrams to PDF")
    parser.add_argument("--clean", action="store_true", help="Clean output directories first")
    parser.add_argument("--svg", action="store_true", help="Also keep SVG files")
    parser.add_argument("--method", choices=["auto", "mmdc", "kroki"], default="auto",
                        help="Conversion method (default: auto)")
    args = parser.parse_args()

    print("=" * 60)
    print("NerGuard Diagram Converter")
    print("=" * 60)
    print()

    # Check available tools
    has_mmdc = check_mmdc_installed()
    print(f"Mermaid CLI (mmdc): {'Available' if has_mmdc else 'Not found'}")

    if args.method == "mmdc" and not has_mmdc:
        print("\nERROR: mmdc requested but not installed")
        print("Install with: npm install -g @mermaid-js/mermaid-cli")
        sys.exit(1)

    use_mmdc = has_mmdc and args.method != "kroki"
    method_name = "mmdc (Mermaid CLI)" if use_mmdc else "mermaid.ink/kroki.io APIs"
    print(f"Using: {method_name}")

    if not use_mmdc:
        print()
        print("NOTE: For best quality, install Mermaid CLI:")
        print("  npm install -g @mermaid-js/mermaid-cli")
    print()

    # Setup directories
    setup_output_dirs(clean=args.clean)

    # Find files
    mmd_files = find_mermaid_files()
    total = len(mmd_files)
    print(f"Found {total} Mermaid files")
    print()

    success_count = 0
    error_count = 0

    for i, mmd_path in enumerate(mmd_files, 1):
        subdir = mmd_path.parent.name
        filename = mmd_path.stem

        svg_path = SVG_OUTPUT_DIR / subdir / f"{filename}.svg"
        pdf_path = PDF_OUTPUT_DIR / subdir / f"{filename}.pdf"

        print(f"[{i}/{total}] {subdir}/{filename}.mmd")

        if use_mmdc:
            # mmdc can directly output PDF
            if convert_with_mmdc(mmd_path, pdf_path, "pdf"):
                print(f"    -> {pdf_path}")
                success_count += 1

                # Also create SVG if requested
                if args.svg:
                    convert_with_mmdc(mmd_path, svg_path, "svg")
            else:
                print(f"    FAILED - trying API fallback")
                # Fallback to APIs
                converted = False
                for api_name, api_func in [("mermaid.ink", convert_with_mermaid_ink), ("kroki", convert_with_kroki)]:
                    if api_func(mmd_path, svg_path):
                        if convert_svg_to_pdf(svg_path, pdf_path):
                            print(f"    -> {pdf_path} (via {api_name})")
                            success_count += 1
                            converted = True
                            break

                if not converted:
                    print(f"    FAILED - all methods failed")
                    error_count += 1
        else:
            # Try mermaid.ink first, then kroki as fallback
            converted = False
            for api_name, api_func in [("mermaid.ink", convert_with_mermaid_ink), ("kroki", convert_with_kroki)]:
                if api_func(mmd_path, svg_path):
                    if convert_svg_to_pdf(svg_path, pdf_path):
                        print(f"    -> {pdf_path} (via {api_name})")
                        success_count += 1
                        converted = True
                        break
                    else:
                        print(f"    {api_name}: SVG ok but PDF conversion failed")

            if not converted:
                print(f"    FAILED - all API methods failed")
                error_count += 1

        # Clean up SVG if not requested
        if not args.svg and svg_path.exists():
            svg_path.unlink()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successful: {success_count}/{total}")
    print(f"Errors: {error_count}/{total}")
    print()
    print("Output directories:")
    print(f"  PDF: {PDF_OUTPUT_DIR}")
    if args.svg:
        print(f"  SVG: {SVG_OUTPUT_DIR}")
    print()

    # Print file counts
    print("File counts:")
    for subdir in SUBDIRS:
        pdf_count = len(list((PDF_OUTPUT_DIR / subdir).glob("*.pdf")))
        print(f"  {subdir}: {pdf_count} PDFs")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
