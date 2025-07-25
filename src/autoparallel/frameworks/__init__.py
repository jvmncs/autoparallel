"""Framework integrations (vLLM, DeepSpeed) for model deployment."""

from autoparallel.frameworks.vllm_memory import (
    WorkloadProfile,
    get_vllm_default_capture_sizes,
    vLLMAutotuningParameters,
    vLLMMemoryEstimator,
)
from autoparallel.frameworks.vllm_config import (
    AutotuningParameters,
    calculate_model_memory_from_config,
    estimate_activation_memory_from_config,
    generate_deployment_recommendations,
    optimize_vllm_config_for_cluster,
    vLLMConfigOptimizer,
    vLLMPerformanceModel,
)

__all__ = [
    # vLLM Memory Estimation
    "WorkloadProfile",
    "get_vllm_default_capture_sizes",
    "vLLMAutotuningParameters",
    "vLLMMemoryEstimator",
    # vLLM Configuration System
    "AutotuningParameters",
    "calculate_model_memory_from_config",
    "estimate_activation_memory_from_config",
    "generate_deployment_recommendations",
    "optimize_vllm_config_for_cluster",
    "vLLMConfigOptimizer",
    "vLLMPerformanceModel",
]
