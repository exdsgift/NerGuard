"""
Visualization style configuration for NerGuard.

This module provides unified style settings for matplotlib and seaborn plots.
All plots use Helvetica font for consistency with publication standards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple, Any


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

# Benchmark/dataset colors (distinctive palette for multi-panel figures)
BENCHMARK_COLORS = {
    # Purple tones
    "purple": "#7B68EE",        # Medium Purple
    "violet": "#8A2BE2",        # Blue Violet

    # Blue tones
    "blue": "#4169E1",          # Royal Blue
    "navy": "#1E3A5F",          # Dark Navy

    # Green tones
    "green": "#228B22",         # Forest Green
    "teal": "#2E8B57",          # Sea Green

    # Orange tones
    "orange": "#FF8C00",        # Dark Orange
    "coral": "#FF6347",         # Tomato

    # Red/Pink tones
    "red": "#DC143C",           # Crimson
    "pink": "#DB7093",          # Pale Violet Red

    # Neutral
    "gray": "#4A4A4A",          # Dark Gray
}

# Ordered list for sequential assignment
BENCHMARK_PALETTE = [
    "#7B68EE",  # Purple
    "#FF8C00",  # Orange
    "#4169E1",  # Blue
    "#228B22",  # Green
    "#DC143C",  # Red
    "#DB7093",  # Pink
    "#2E8B57",  # Teal
    "#8A2BE2",  # Violet
    "#FF6347",  # Coral
    "#1E3A5F",  # Navy
]

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

    Uses Helvetica font, high DPI, and clean aesthetics consistent
    with academic publication standards.
    """
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update({
        # Font settings - Helvetica priority
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",

        # Figure settings
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        # Axes settings
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.facecolor": "white",

        # Tick settings
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # Legend settings
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.edgecolor": "0.8",
        "legend.facecolor": "white",

        # Grid settings
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",

        # Hatching
        "hatch.linewidth": 0.5,
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


def plot_comparison_bars(
    ax: Any,
    data_baseline: List[float],
    data_comparison: List[float],
    labels: List[str],
    color: str,
    baseline_label: str = "Baseline",
    comparison_label: str = "Treatment",
    show_significance: bool = False,
    significance_markers: Optional[List[bool]] = None,
    width: float = 0.35,
    rotate_labels: bool = True,
) -> Tuple[Any, Any]:
    """
    Create grouped bar chart with hatching for comparison studies.

    Solid bars represent baseline, hatched bars represent treatment/comparison.
    This style is consistent with academic publication standards.

    Args:
        ax: Matplotlib axis object
        data_baseline: Values for baseline condition (solid bars)
        data_comparison: Values for comparison/treatment condition (hatched bars)
        labels: X-axis labels for each group
        color: Bar color (same for both conditions)
        baseline_label: Legend label for baseline bars
        comparison_label: Legend label for comparison bars
        show_significance: Whether to add significance markers
        significance_markers: List of booleans indicating significance for each bar
        width: Bar width
        rotate_labels: Whether to rotate x-axis labels

    Returns:
        Tuple of (comparison_bars, baseline_bars) bar containers

    Example:
        >>> fig, ax = plt.subplots()
        >>> set_publication_style()
        >>> plot_comparison_bars(
        ...     ax,
        ...     data_baseline=[0.8, 0.75, 0.9],
        ...     data_comparison=[0.85, 0.78, 0.88],
        ...     labels=["Model A", "Model B", "Model C"],
        ...     color="#7B68EE",
        ...     comparison_label="With Enhancement",
        ...     show_significance=True,
        ...     significance_markers=[True, False, False]
        ... )
    """
    x = np.arange(len(labels))

    # Hatched bars (comparison/treatment) - left position
    bars_comparison = ax.bar(
        x - width / 2,
        data_comparison,
        width,
        color=color,
        hatch="//",
        edgecolor="white",
        linewidth=0.5,
        label=comparison_label,
    )

    # Solid bars (baseline) - right position
    bars_baseline = ax.bar(
        x + width / 2,
        data_baseline,
        width,
        color=color,
        label=baseline_label,
    )

    # Add significance markers
    if show_significance and significance_markers:
        for i, (bar, sig) in enumerate(zip(bars_comparison, significance_markers)):
            if sig:
                ax.annotate(
                    "*",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if rotate_labels else 0, ha="right" if rotate_labels else "center")

    return bars_comparison, bars_baseline


def get_benchmark_color(index: int) -> str:
    """
    Get a color from the benchmark palette by index.

    Colors cycle if index exceeds palette length.

    Args:
        index: Color index

    Returns:
        Hex color code
    """
    return BENCHMARK_PALETTE[index % len(BENCHMARK_PALETTE)]


def add_significance_stars(
    ax: Any,
    bars: Any,
    p_values: List[float],
    threshold: float = 0.05,
) -> None:
    """
    Add significance stars above bars based on p-values.

    Args:
        ax: Matplotlib axis
        bars: Bar container from ax.bar()
        p_values: List of p-values for each bar
        threshold: Significance threshold (default 0.05)
    """
    for bar, p in zip(bars, p_values):
        if p < threshold:
            ax.annotate(
                "*",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )
