"""
Experiments module for NerGuard.

Contains experimental scripts for evaluating and optimizing model components.
"""

__all__ = ["LLMRouterExperiment"]


def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name == "LLMRouterExperiment":
        from src.experiments.llm_router_experiment import LLMRouterExperiment
        return LLMRouterExperiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
