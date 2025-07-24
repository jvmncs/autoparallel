"""AutoParallel: Automatic LLM parallelization strategy optimization."""

__version__ = "0.1.0"

# Import public API functions and classes
from autoparallel.api.advanced import (
    AnalysisInsights,
    AutoParallel,
    Cluster,
    DetailedConfiguration,
    OptimizationResult,
    Preferences,
    Workload,
)
from autoparallel.api.simple import AnalysisResult, OptimizedConfig, analyze, optimize

# Make key classes and functions available at package level
__all__ = [
    # Simple API
    "analyze",
    "optimize",
    "AnalysisResult",
    "OptimizedConfig",
    # Advanced API
    "AutoParallel",
    "Cluster",
    "Workload",
    "Preferences",
    "OptimizationResult",
    "DetailedConfiguration",
    "AnalysisInsights",
]
