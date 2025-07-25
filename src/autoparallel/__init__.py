"""AutoParallel: Automatic LLM parallelization strategy optimization."""

__version__ = "0.1.0"

# Import simplified public API
from autoparallel.public_api import (
    InsufficientMemoryError,
    ModelNotFoundError,
    analyze,
    best_config,
    check_memory_requirements,
    estimate_cost,
    find_minimum_gpus,
    get_memory_estimate,
    # Convenience aliases
    get_parallelism_config,
)

# Export main API functions
__all__ = [
    "__version__",
    # Core API functions
    "analyze",
    "best_config",
    "check_memory_requirements",
    "estimate_cost",
    # Exceptions
    "ModelNotFoundError",
    "InsufficientMemoryError",
    # Convenience functions
    "get_parallelism_config",
    "get_memory_estimate",
    "find_minimum_gpus",
]
