"""Simplified constraint analysis for autoparallel configurations.

This module provides simple functional constraint analysis without complex class
hierarchies. Focus on determining valid parallelism sizes based on model
architecture divisibility rules.
"""

from typing import Any

try:
    from transformers import PretrainedConfig
except ImportError:
    # Handle case where transformers is not available during testing
    PretrainedConfig = dict  # type: ignore


def valid_tensor_parallel_sizes(
    model_config: PretrainedConfig | dict[str, Any], max_size: int
) -> list[int]:
    """Return valid tensor parallel sizes based on attention heads divisibility.

    Args:
        model_config: Hugging Face model configuration
        max_size: Maximum tensor parallel size to consider

    Returns:
        List of valid tensor parallel sizes in ascending order
    """
    # Extract attention head parameters
    num_attention_heads = getattr(model_config, "num_attention_heads", 32)
    num_key_value_heads = getattr(
        model_config, "num_key_value_heads", num_attention_heads
    )

    valid_sizes = []

    # Check divisibility constraints for each potential TP size
    for tp_size in range(1, max_size + 1):
        # Primary constraint: attention heads must be divisible by TP size
        # Secondary constraint: key-value heads must be divisible (for GQA models)
        if (num_attention_heads % tp_size == 0 and
                num_key_value_heads % tp_size == 0):
            valid_sizes.append(tp_size)

    return valid_sizes


def valid_pipeline_parallel_sizes(
    model_config: PretrainedConfig | dict[str, Any], max_size: int
) -> list[int]:
    """Return valid pipeline parallel sizes based on layer count.

    Args:
        model_config: Hugging Face model configuration
        max_size: Maximum pipeline parallel size to consider

    Returns:
        List of valid pipeline parallel sizes in ascending order
    """
    # Extract layer count
    num_layers = getattr(model_config, "num_hidden_layers", 32)

    valid_sizes = []
    min_layers_per_stage = 2  # Minimum layers per pipeline stage

    # Check feasibility for each potential PP size
    for pp_size in range(1, min(max_size, num_layers) + 1):
        layers_per_stage = num_layers / pp_size

        # Ensure minimum layers per stage requirement
        if layers_per_stage >= min_layers_per_stage:
            valid_sizes.append(pp_size)

    return valid_sizes


def valid_expert_parallel_sizes(
    model_config: PretrainedConfig | dict[str, Any], max_size: int
) -> list[int]:
    """Return valid expert parallel sizes for MoE models (1 for dense models).

    Args:
        model_config: Hugging Face model configuration
        max_size: Maximum expert parallel size to consider

    Returns:
        List of valid expert parallel sizes in ascending order
    """
    # Extract MoE parameters
    num_experts = getattr(
        model_config, "num_local_experts", getattr(model_config, "num_experts", 0)
    )

    # For non-MoE models, only EP=1 is valid
    if num_experts == 0:
        return [1]

    valid_sizes = []
    min_experts_per_device = 1  # Minimum experts per device

    # Check divisibility for each potential EP size
    for ep_size in range(1, min(max_size, num_experts) + 1):
        # Experts must be evenly distributed
        if num_experts % ep_size == 0:
            experts_per_device = num_experts / ep_size

            # Ensure minimum experts per device
            if experts_per_device >= min_experts_per_device:
                valid_sizes.append(ep_size)

    return valid_sizes


def get_divisors(n: int, max_divisor: int | None = None) -> list[int]:
    """Get all divisors of a number up to max_divisor.

    Args:
        n: Number to find divisors for
        max_divisor: Maximum divisor to consider (default: n)

    Returns:
        List of divisors in ascending order
    """
    if max_divisor is None:
        max_divisor = n

    divisors = []

    # Find divisors efficiently by checking up to sqrt(n)
    for i in range(1, min(int(n**0.5) + 1, max_divisor + 1)):
        if n % i == 0:
            divisors.append(i)
            # Add the corresponding divisor (n/i) if it's different and within limit
            if i != n // i and n // i <= max_divisor:
                divisors.append(n // i)

    return sorted(divisors)


def is_power_of_2(n: int) -> bool:
    """Check if a number is a power of 2.

    Args:
        n: Number to check

    Returns:
        True if n is a power of 2, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0


def validate_parallelism_config(
    tensor_parallel: int,
    pipeline_parallel: int,
    expert_parallel: int,
    data_parallel: int,
    total_gpus: int,
) -> tuple[bool, list[str]]:
    """Validate that parallelism configuration is mathematically consistent.

    Args:
        tensor_parallel: Tensor parallel size
        pipeline_parallel: Pipeline parallel size
        expert_parallel: Expert parallel size
        data_parallel: Data parallel size
        total_gpus: Total available GPUs

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check positive values
    sizes = [
        ("tensor_parallel", tensor_parallel),
        ("pipeline_parallel", pipeline_parallel),
        ("expert_parallel", expert_parallel),
        ("data_parallel", data_parallel),
    ]

    for name, size in sizes:
        if size <= 0:
            errors.append(f"{name} must be positive, got {size}")

    # Check GPU allocation consistency
    required_gpus = (
        tensor_parallel * pipeline_parallel * expert_parallel * data_parallel
    )

    if required_gpus != total_gpus:
        errors.append(
            f"Parallelism sizes multiply to {required_gpus} but "
            f"total GPUs is {total_gpus}"
        )

    return len(errors) == 0, errors


def get_model_architecture_info(
    model_config: PretrainedConfig | dict[str, Any],
) -> dict:
    """Extract basic model architecture information for constraint analysis.

    Args:
        model_config: Hugging Face model configuration

    Returns:
        Dictionary with model architecture parameters
    """
    return {
        "hidden_size": getattr(model_config, "hidden_size", 4096),
        "num_attention_heads": getattr(model_config, "num_attention_heads", 32),
        "num_key_value_heads": getattr(
            model_config,
            "num_key_value_heads",
            getattr(model_config, "num_attention_heads", 32),
        ),
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", 32),
        "vocab_size": getattr(model_config, "vocab_size", 32000),
        "intermediate_size": getattr(
            model_config,
            "intermediate_size",
            4 * getattr(model_config, "hidden_size", 4096),
        ),
        "num_experts": getattr(
            model_config, "num_local_experts", getattr(model_config, "num_experts", 0)
        ),
        "num_experts_per_token": getattr(
            model_config, "num_experts_per_tok", getattr(model_config, "top_k", 2)
        ),
        "model_type": getattr(model_config, "model_type", "unknown"),
        "is_moe": getattr(
            model_config, "num_local_experts", getattr(model_config, "num_experts", 0)
        )
        > 0,
    }


def check_tensor_parallel_efficiency(
    model_config: PretrainedConfig | dict[str, Any], tensor_parallel_size: int
) -> list[str]:
    """Check tensor parallel configuration for efficiency warnings.

    Args:
        model_config: Hugging Face model configuration
        tensor_parallel_size: Proposed tensor parallel size

    Returns:
        List of efficiency warnings
    """
    warnings = []
    arch_info = get_model_architecture_info(model_config)

    # Check if hidden size is efficiently divisible
    if arch_info["hidden_size"] % tensor_parallel_size != 0:
        warnings.append(
            f"Hidden size ({arch_info['hidden_size']}) not divisible by "
            f"tensor parallel size ({tensor_parallel_size}) - "
            f"may cause padding overhead"
        )

    # Check if intermediate size is efficiently divisible
    if arch_info["intermediate_size"] % tensor_parallel_size != 0:
        warnings.append(
            f"Intermediate size ({arch_info['intermediate_size']}) not divisible by "
            f"tensor parallel size ({tensor_parallel_size}) - may reduce MLP efficiency"
        )

    # Check if vocabulary size is efficiently divisible
    if arch_info["vocab_size"] % tensor_parallel_size != 0:
        warnings.append(
            f"Vocabulary size ({arch_info['vocab_size']}) not divisible by "
            f"tensor parallel size ({tensor_parallel_size}) - "
            f"may cause embedding padding"
        )

    # Recommend power-of-2 sizes for large TP
    if tensor_parallel_size > 8 and not is_power_of_2(tensor_parallel_size):
        warnings.append(
            f"Consider using power-of-2 tensor parallel size instead of "
            f"{tensor_parallel_size} for better communication efficiency"
        )

    return warnings


def check_pipeline_parallel_efficiency(
    model_config: PretrainedConfig | dict[str, Any], pipeline_parallel_size: int
) -> list[str]:
    """Check pipeline parallel configuration for efficiency warnings.

    Args:
        model_config: Hugging Face model configuration
        pipeline_parallel_size: Proposed pipeline parallel size

    Returns:
        List of efficiency warnings
    """
    warnings = []
    arch_info = get_model_architecture_info(model_config)

    num_layers = arch_info["num_hidden_layers"]
    layers_per_stage = num_layers / pipeline_parallel_size

    # Check for uneven layer distribution
    if num_layers % pipeline_parallel_size != 0:
        warnings.append(
            f"Number of layers ({num_layers}) not evenly divisible by "
            f"pipeline parallel size ({pipeline_parallel_size}) - "
            f"some stages will have more layers causing load imbalance"
        )

    # Warn about too many pipeline stages
    if pipeline_parallel_size > 4:
        warnings.append(
            f"Pipeline parallel size of {pipeline_parallel_size} may introduce "
            f"significant pipeline bubble overhead"
        )

    # Check minimum layers per stage
    min_layers_per_stage = 2
    if layers_per_stage < min_layers_per_stage:
        warnings.append(
            f"Pipeline configuration results in {layers_per_stage:.1f} "
            f"layers per stage, below recommended minimum of {min_layers_per_stage}"
        )

    return warnings


def check_expert_parallel_efficiency(
    model_config: PretrainedConfig | dict[str, Any], expert_parallel_size: int
) -> list[str]:
    """Check expert parallel configuration for efficiency warnings.

    Args:
        model_config: Hugging Face model configuration
        expert_parallel_size: Proposed expert parallel size

    Returns:
        List of efficiency warnings
    """
    warnings = []
    arch_info = get_model_architecture_info(model_config)

    # Non-MoE models should use EP=1
    if not arch_info["is_moe"] and expert_parallel_size > 1:
        warnings.append(
            f"Expert parallel size ({expert_parallel_size}) "
            f"should be 1 for non-MoE models"
        )
        return warnings

    if arch_info["is_moe"]:
        num_experts = arch_info["num_experts"]
        experts_per_device = num_experts / expert_parallel_size

        # Warn about too many expert parallel groups
        if expert_parallel_size > 8:
            warnings.append(
                f"Expert parallel size of {expert_parallel_size} may cause "
                f"excessive all-to-all communication overhead"
            )

        # Check minimum experts per device
        if experts_per_device < 1:
            warnings.append(
                f"Expert parallel size ({expert_parallel_size}) results in "
                f"{experts_per_device:.1f} experts per device, below minimum of 1"
            )

    return warnings
