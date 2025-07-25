"""Simplified public API for autoparallel.

This module provides the main user-facing API with progressive disclosure.
Simple functions for the 95% use case with optional complexity for advanced users.
"""

import warnings
from typing import Any

try:
    from transformers import AutoConfig, PretrainedConfig

    HAS_TRANSFORMERS = True
except ImportError:
    # Handle case where transformers is not available
    HAS_TRANSFORMERS = False
    PretrainedConfig = dict  # type: ignore
    AutoConfig = None  # type: ignore

from autoparallel.constraints import get_model_architecture_info
from autoparallel.grid_search import find_best_config, find_valid_configs
from autoparallel.memory import check_memory_feasibility


class ModelNotFoundError(Exception):
    """Raised when model cannot be loaded or found."""

    pass


class InsufficientMemoryError(Exception):
    """Raised when model cannot fit in available GPU memory."""

    pass


def analyze(
    model: str,
    cluster: dict[str, Any],
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization: str = "fp16",
    max_configs: int = 10,
) -> list[dict[str, Any]]:
    """Main analysis function returning ranked parallelism configurations.

    This is the primary function for most users. It analyzes a model against
    a GPU cluster and returns the best parallelism strategies.

    Args:
        model: Hugging Face model identifier (e.g., "meta-llama/Llama-2-7b-hf")
        cluster: Cluster specification dict with keys:
            - gpu_count: Number of GPUs
            - gpu_memory_gb: Memory per GPU in GB
            - gpu_type: GPU type (optional, for context)
        sequence_length: Input sequence length (default: 2048)
        batch_size: Batch size per GPU (default: 1)
        quantization: Quantization format ("fp32", "fp16", "bf16", "int8", "fp8")
        max_configs: Maximum configurations to return (default: 10)

    Returns:
        List of configuration dictionaries, ranked by efficiency score.
        Each dict contains:
        - tensor_parallel: Tensor parallel size
        - pipeline_parallel: Pipeline parallel size
        - expert_parallel: Expert parallel size
        - data_parallel: Data parallel size
        - total_gpus: Total GPUs used
        - memory_per_gpu_gb: Memory usage per GPU
        - memory_utilization: Memory utilization ratio
        - score: Efficiency ranking score
        - memory_breakdown: Detailed memory breakdown

    Raises:
        ModelNotFoundError: If model cannot be loaded
        InsufficientMemoryError: If model cannot fit in cluster
        ValueError: If cluster specification is invalid
    """
    # Validate cluster specification
    _validate_cluster_spec(cluster)

    # Load model configuration
    model_config = _load_model_config(model)

    # Convert quantization to bytes
    from autoparallel.memory import get_quantization_bytes

    quantization_bytes = get_quantization_bytes(quantization)

    # Find valid configurations
    try:
        valid_configs = find_valid_configs(
            model_config=model_config,
            max_gpus=cluster["gpu_count"],
            gpu_memory_gb=cluster["gpu_memory_gb"],
            sequence_length=sequence_length,
            batch_size=batch_size,
            quantization_bytes=quantization_bytes,
            max_configs=max_configs,
        )
    except Exception as e:
        raise InsufficientMemoryError(f"Failed to find valid configurations: {e}")

    if not valid_configs:
        raise InsufficientMemoryError(
            f"Model '{model}' cannot fit in cluster with {cluster['gpu_count']} "
            f"x {cluster['gpu_memory_gb']}GB GPUs"
        )

    # Convert to output format
    return [config.to_dict() for config in valid_configs]


def best_config(
    model: str,
    cluster: dict[str, Any],
    objective: str = "minimize_gpus",
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization: str = "fp16",
) -> dict[str, Any]:
    """Get single best configuration for given objective.

    Convenience function for users who want a single recommendation.

    Args:
        model: Hugging Face model identifier
        cluster: Cluster specification dict
        objective: Optimization objective:
            - "minimize_gpus": Use fewest GPUs possible
            - "maximize_throughput": Optimize for throughput
            - "balance": Balanced resource usage (default scoring)
        sequence_length: Input sequence length
        batch_size: Batch size per GPU
        quantization: Quantization format

    Returns:
        Best configuration dictionary (same format as analyze())

    Raises:
        ModelNotFoundError: If model cannot be loaded
        InsufficientMemoryError: If model cannot fit in cluster
        ValueError: If objective is invalid
    """
    # Validate inputs
    _validate_cluster_spec(cluster)
    if objective not in ["minimize_gpus", "maximize_throughput", "balance"]:
        raise ValueError(
            f"Invalid objective '{objective}'. Use: minimize_gpus, maximize_throughput, balance"
        )

    # Load model configuration
    model_config = _load_model_config(model)

    # Convert quantization to bytes
    from autoparallel.memory import get_quantization_bytes

    quantization_bytes = get_quantization_bytes(quantization)

    # Find best configuration
    try:
        best_config_obj = find_best_config(
            model_config=model_config,
            max_gpus=cluster["gpu_count"],
            gpu_memory_gb=cluster["gpu_memory_gb"],
            objective=objective,
            sequence_length=sequence_length,
            batch_size=batch_size,
            quantization_bytes=quantization_bytes,
        )
    except Exception as e:
        raise InsufficientMemoryError(f"Failed to find best configuration: {e}")

    return best_config_obj.to_dict()


def check_memory_requirements(
    model: str,
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization: str = "fp16",
) -> dict[str, Any]:
    """Check memory requirements for a model without hardware constraints.

    Useful for understanding model memory needs before planning deployment.

    Args:
        model: Hugging Face model identifier
        sequence_length: Input sequence length
        batch_size: Batch size
        quantization: Quantization format

    Returns:
        Dictionary with memory analysis:
        - total_memory_gb: Total model memory in GB
        - breakdown: Component breakdown (weights, activations, kv_cache, overhead)
        - single_gpu_requirements: Minimum GPU memory needed for single GPU
        - recommended_gpu_memory_gb: Recommended GPU memory with safety margin
        - architecture_info: Model architecture details

    Raises:
        ModelNotFoundError: If model cannot be loaded
    """
    # Load model configuration
    model_config = _load_model_config(model)

    # Convert quantization to bytes
    from autoparallel.memory import get_quantization_bytes

    quantization_bytes = get_quantization_bytes(quantization)

    # Get memory feasibility analysis
    feasibility = check_memory_feasibility(
        model_config=model_config,
        gpu_memory_gb=1000.0,  # Large value to get base requirements
        sequence_length=sequence_length,
        batch_size=batch_size,
        quantization_bytes=quantization_bytes,
    )

    # Get architecture info
    arch_info = get_model_architecture_info(model_config)

    # Calculate recommendations
    total_memory_gb = feasibility["total_memory_gb"]
    recommended_gpu_memory = total_memory_gb / 0.8  # 80% utilization target

    return {
        "total_memory_gb": total_memory_gb,
        "breakdown": feasibility["breakdown"],
        "single_gpu_requirements": {
            "min_memory_gb": total_memory_gb,
            "recommended_memory_gb": recommended_gpu_memory,
            "fits_in_common_gpus": {
                "A100_40GB": total_memory_gb <= 36.0,  # 90% of 40GB
                "A100_80GB": total_memory_gb <= 72.0,  # 90% of 80GB
                "H100_80GB": total_memory_gb <= 72.0,  # 90% of 80GB
                "RTX_4090_24GB": total_memory_gb <= 21.6,  # 90% of 24GB
                "RTX_3090_24GB": total_memory_gb <= 21.6,  # 90% of 24GB
            },
        },
        "architecture_info": {
            "model_type": arch_info["model_type"],
            "num_parameters_estimate": _estimate_param_count_simple(arch_info),
            "is_moe": arch_info["is_moe"],
            "num_experts": arch_info["num_experts"] if arch_info["is_moe"] else 0,
            "num_layers": arch_info["num_hidden_layers"],
            "hidden_size": arch_info["hidden_size"],
            "attention_heads": arch_info["num_attention_heads"],
            "vocab_size": arch_info["vocab_size"],
        },
    }


def estimate_cost(
    model: str,
    cluster: dict[str, Any],
    hours_per_month: int = 730,
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization: str = "fp16",
    cost_per_gpu_hour: float = 1.0,
) -> dict[str, Any]:
    """Estimate deployment cost for different configurations.

    Basic cost estimation based on GPU usage.

    Args:
        model: Hugging Face model identifier
        cluster: Cluster specification dict
        hours_per_month: Usage hours per month
        sequence_length: Input sequence length
        batch_size: Batch size per GPU
        quantization: Quantization format
        cost_per_gpu_hour: Cost per GPU per hour in USD

    Returns:
        Dictionary with cost analysis for different objectives

    Raises:
        ModelNotFoundError: If model cannot be loaded
        InsufficientMemoryError: If model cannot fit in cluster
    """
    warnings.warn(
        "Cost estimation is a basic heuristic. Actual costs depend on cloud provider, "
        "region, spot pricing, and other factors.",
        UserWarning,
    )

    # Get configurations for different objectives
    objectives = ["minimize_gpus", "maximize_throughput", "balance"]
    cost_analysis = {}

    for objective in objectives:
        try:
            config = best_config(
                model=model,
                cluster=cluster,
                objective=objective,
                sequence_length=sequence_length,
                batch_size=batch_size,
                quantization=quantization,
            )

            gpus_used = config["total_gpus"]
            monthly_cost = gpus_used * hours_per_month * cost_per_gpu_hour

            cost_analysis[objective] = {
                "gpus_used": gpus_used,
                "cost_per_hour": gpus_used * cost_per_gpu_hour,
                "cost_per_month": monthly_cost,
                "memory_utilization": config["memory_utilization"],
                "configuration": config,
            }

        except Exception as e:
            cost_analysis[objective] = {
                "error": str(e),
                "gpus_used": None,
                "cost_per_hour": None,
                "cost_per_month": None,
            }

    return {
        "cost_analysis": cost_analysis,
        "assumptions": {
            "hours_per_month": hours_per_month,
            "cost_per_gpu_hour": cost_per_gpu_hour,
            "cluster": cluster,
        },
    }


def _validate_cluster_spec(cluster: dict[str, Any]) -> None:
    """Validate cluster specification."""
    required_keys = ["gpu_count", "gpu_memory_gb"]

    for key in required_keys:
        if key not in cluster:
            raise ValueError(f"Cluster specification missing required key: {key}")

    if not isinstance(cluster["gpu_count"], int) or cluster["gpu_count"] <= 0:
        raise ValueError("gpu_count must be a positive integer")

    if (
        not isinstance(cluster["gpu_memory_gb"], (int, float))
        or cluster["gpu_memory_gb"] <= 0
    ):
        raise ValueError("gpu_memory_gb must be a positive number")


def _load_model_config(model: str) -> PretrainedConfig | dict[str, Any]:
    """Load model configuration from Hugging Face."""
    if not HAS_TRANSFORMERS:
        raise ModelNotFoundError(
            "transformers library is required to load model configurations. "
            "Install with: pip install transformers"
        )

    try:
        # Try to load from Hugging Face Hub
        config = AutoConfig.from_pretrained(model, trust_remote_code=False)
        return config
    except Exception as e:
        # If loading fails, provide helpful error message
        error_msg = f"Failed to load model '{model}': {e}"

        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            error_msg += "\n\nCommon causes:\n"
            error_msg += "- Model name is misspelled\n"
            error_msg += "- Model is private and requires authentication\n"
            error_msg += "- Model doesn't exist on Hugging Face Hub\n"
            error_msg += "\nTry: huggingface-cli login (if model is private)"

        raise ModelNotFoundError(error_msg)


def _estimate_param_count_simple(arch_info: dict[str, Any]) -> str:
    """Simple parameter count estimation for display."""
    from autoparallel.memory import _estimate_param_count

    param_count = _estimate_param_count(
        vocab_size=arch_info["vocab_size"],
        hidden_size=arch_info["hidden_size"],
        num_layers=arch_info["num_hidden_layers"],
        intermediate_size=arch_info["intermediate_size"],
        num_experts=arch_info["num_experts"],
    )

    # Convert to human-readable format
    if param_count >= 1_000_000_000:
        return f"{param_count / 1_000_000_000:.1f}B"
    elif param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.1f}M"
    else:
        return f"{param_count:,}"


# Convenience aliases for backward compatibility and ease of use
def get_parallelism_config(
    model: str, cluster: dict[str, Any], **kwargs
) -> dict[str, Any]:
    """Alias for best_config() with default 'balance' objective."""
    return best_config(model, cluster, objective="balance", **kwargs)


def get_memory_estimate(model: str, **kwargs) -> dict[str, Any]:
    """Alias for check_memory_requirements()."""
    return check_memory_requirements(model, **kwargs)


def find_minimum_gpus(model: str, gpu_memory_gb: float, **kwargs) -> dict[str, Any]:
    """Find minimum GPU count needed for a model."""
    # Use a large cluster to find minimum requirements
    large_cluster = {"gpu_count": 128, "gpu_memory_gb": gpu_memory_gb}

    try:
        config = best_config(model, large_cluster, objective="minimize_gpus", **kwargs)
        return {
            "min_gpus": config["total_gpus"],
            "memory_per_gpu_gb": config["memory_per_gpu_gb"],
            "memory_utilization": config["memory_utilization"],
            "configuration": config,
        }
    except Exception as e:
        raise InsufficientMemoryError(f"Cannot determine minimum GPUs: {e}")
