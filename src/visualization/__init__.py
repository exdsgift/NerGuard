"""
Visualization module for NerGuard.

This module provides plotting utilities for:
- Model evaluation and comparison
- Confusion matrices
- Entropy and confidence distributions
- LLM impact analysis
- Benchmark comparisons
- Optimization analysis
"""

from src.visualization.style import (
    set_publication_style,
    set_style,
    COLORS,
    get_color_palette,
)
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_entropy_separation,
    plot_model_comparison,
    plot_metrics_radar,
)
from src.visualization.benchmark_plots import (
    plot_main_metrics,
    plot_efficiency_frontier,
    plot_entity_comparison,
    plot_confusion_matrix_single,
)
from src.visualization.optimization_plots import (
    plot_optimization_heatmap,
    plot_pareto_frontier,
    plot_quantization_metrics,
    plot_quantization_radar,
    save_quantization_report,
)

__all__ = [
    # Style
    "set_publication_style",
    "set_style",
    "COLORS",
    "get_color_palette",
    # Core plots
    "plot_confusion_matrix",
    "plot_entropy_separation",
    "plot_model_comparison",
    "plot_metrics_radar",
    # Benchmark plots
    "plot_main_metrics",
    "plot_efficiency_frontier",
    "plot_entity_comparison",
    "plot_confusion_matrix_single",
    # Optimization plots
    "plot_optimization_heatmap",
    "plot_pareto_frontier",
    "plot_quantization_metrics",
    "plot_quantization_radar",
    "save_quantization_report",
]
