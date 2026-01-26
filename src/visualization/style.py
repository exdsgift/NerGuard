"""
Visualization style configuration for NerGuard.

This module provides unified style settings for matplotlib and seaborn plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


# COLOR PALETTES
COLORS = {
    # Primary colors
    "baseline": "#1f77b4",      # Blue
    "hybrid": "#2ca02c",        # Green
    "primary": "#1f77b4",       # Blue

    # Status colors
    "positive": "#2ca02c",      # Green (success, improvement)
    "negative": "#d62728",      # Red (error, decrease)
    "neutral": "#7f7f7f",       # Gray

    # Accent colors
    "accent_1": "#ff7f0e",      # Orange
    "accent_2": "#9467bd",      # Purple
    "accent_3": "#8c564b",      # Brown
    "accent_4": "#e377c2",      # Pink

    # Background
    "background": "#ffffff",
    "grid": "#e0e0e0",
}

# Seaborn color palettes
PALETTE_COMPARISON = [COLORS["baseline"], COLORS["hybrid"]]
PALETTE_CATEGORICAL = sns.color_palette("husl", 10)


def get_color_palette(
    name: str = "comparison",
    n_colors: int = 10,
) -> List[str]:
    """
    Get a color palette by name.

    Args:
        name: Palette name ("comparison", "categorical", "sequential")
        n_colors: Number of colors for categorical palettes

    Returns:
        List of color hex codes
    """
    if name == "comparison":
        return PALETTE_COMPARISON
    elif name == "categorical":
        return list(sns.color_palette("husl", n_colors).as_hex())
    elif name == "sequential":
        return list(sns.color_palette("Blues", n_colors).as_hex())
    elif name == "diverging":
        return list(sns.color_palette("RdBu", n_colors).as_hex())
    else:
        return list(sns.color_palette(name, n_colors).as_hex())

# STYLE CONFIGURATION

def set_publication_style() -> None:
    """
    Configure matplotlib/seaborn for publication-quality figures.

    Sets academic-style fonts, high DPI, and clean aesthetics.
    """
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        # Font settings
        "font.family": "serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 11,

        # Figure settings
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        # Axes settings
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",

        # Tick settings
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # Legend settings
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.edgecolor": "#333333",
        "legend.facecolor": "white",

        # Grid settings
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
    })


def set_style(style_type: str = "publication") -> None:
    """
    Set the plotting style.

    Args:
        style_type: Style preset ("publication", "presentation", "minimal")
    """
    if style_type == "publication":
        set_publication_style()
    elif style_type == "presentation":
        _set_presentation_style()
    elif style_type == "minimal":
        _set_minimal_style()
    else:
        set_publication_style()


def _set_presentation_style() -> None:
    """Configure style for presentations (larger fonts)."""
    sns.set_theme(style="whitegrid", context="talk")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })


def _set_minimal_style() -> None:
    """Configure minimal style for quick plots."""
    sns.set_theme(style="white", context="notebook")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


# HELPER FUNCTIONS

def style_axis(
    ax,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    remove_top_right: bool = True,
) -> None:
    """
    Apply consistent styling to a matplotlib axis.

    Args:
        ax: Matplotlib axis object
        title: Axis title
        xlabel: X-axis label
        ylabel: Y-axis label
        remove_top_right: Whether to remove top and right spines
    """
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.grid(axis="y", linestyle=":", alpha=0.3, linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=10, length=4, width=1, colors="#333333")


def save_figure(
    fig,
    filename: str,
    output_dir: str = ".",
    formats: List[str] = ["png"],
    close: bool = True,
) -> None:
    """
    Save a figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save (e.g., ["png", "pdf", "svg"])
        close: Whether to close the figure after saving
    """
    import os
    from src.utils.io import ensure_dir

    ensure_dir(output_dir)

    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")

    if close:
        plt.close(fig)
