"""CLI entry point for nerguard-rag.

Delegates to the unified nerguard CLI with --format rag as default.
All flags are identical to the main nerguard CLI.
"""

import sys


def main() -> None:
    """nerguard-rag CLI — nerguard with --format rag as default."""
    from src.scripts.cli import main as nerguard_main

    # Inject --format rag as default if the user hasn't specified a format
    argv = sys.argv[1:]
    if "--format" not in argv:
        argv = ["--format", "rag"] + argv

    sys.argv = ["nerguard-rag"] + argv
    nerguard_main()


if __name__ == "__main__":
    main()
