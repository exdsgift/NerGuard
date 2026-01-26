"""
ANSI color codes for terminal output.

This module provides color constants for colorful terminal output.
"""


class Colors:
    """
    ANSI escape codes for terminal colors and styles.

    Usage:
        >>> print(f"{Colors.OKGREEN}Success!{Colors.ENDC}")
        >>> print(f"{Colors.BOLD}{Colors.FAIL}Error!{Colors.ENDC}")
    """

    # Text colors
    HEADER = "\033[95m"      # Light magenta
    OKBLUE = "\033[94m"      # Light blue
    OKCYAN = "\033[96m"      # Light cyan
    OKGREEN = "\033[92m"     # Light green
    WARNING = "\033[93m"     # Yellow
    FAIL = "\033[91m"        # Light red

    # Text styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

    # Reset
    ENDC = "\033[0m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """
        Wrap text with a color code.

        Args:
            text: Text to colorize
            color: Color attribute name (e.g., "OKGREEN", "FAIL")

        Returns:
            Colorized text string
        """
        color_code = getattr(cls, color.upper(), cls.ENDC)
        return f"{color_code}{text}{cls.ENDC}"

    @classmethod
    def success(cls, text: str) -> str:
        """Format text as success (green)."""
        return f"{cls.OKGREEN}{text}{cls.ENDC}"

    @classmethod
    def error(cls, text: str) -> str:
        """Format text as error (red)."""
        return f"{cls.FAIL}{text}{cls.ENDC}"

    @classmethod
    def warning(cls, text: str) -> str:
        """Format text as warning (yellow)."""
        return f"{cls.WARNING}{text}{cls.ENDC}"

    @classmethod
    def info(cls, text: str) -> str:
        """Format text as info (blue)."""
        return f"{cls.OKBLUE}{text}{cls.ENDC}"

    @classmethod
    def bold(cls, text: str) -> str:
        """Format text as bold."""
        return f"{cls.BOLD}{text}{cls.ENDC}"

    @classmethod
    def highlight(cls, text: str) -> str:
        """Format text with highlight (cyan background)."""
        return f"{cls.BG_CYAN}{cls.BOLD}{text}{cls.ENDC}"

    @classmethod
    def disable(cls) -> None:
        """
        Disable all colors (useful for non-TTY output).

        After calling this method, all color codes become empty strings.
        """
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")

    @classmethod
    def is_tty() -> bool:
        """Check if stdout is a TTY (supports colors)."""
        import sys
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
