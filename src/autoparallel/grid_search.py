"""Simplified grid search for autoparallel configuration generation.

This module provides simple grid search with heuristic ranking without complex
multi-objective optimization. Focus on finding memory-feasible configurations
and ranking by simple heuristics.
"""

import itertools
from dataclasses import dataclass
from typing import Any

try:
    from transformers import PretrainedConfig
except ImportError:
    # Handle case where transformers is not available during testing
    PretrainedConfig = dict  # type: ignore

from autoparallel.constraints import (
    valid_expert_parallel_sizes,
    valid_pipeline_parallel_sizes,
    valid_tensor_parallel_sizes,
    validate_parallelism_config,
)
from autoparallel.memory import (
    MemoryBreakdown,
    estimate_memory_for_config,
)


@dataclass
class ParallelismConfig:
    """Parallelism configuration with performance metrics."""

    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    data_parallel: int

    # Memory breakdown for this configuration
    memory_breakdown: MemoryBreakdown

    # Performance metrics
    total_gpus: int
    memory_per_gpu_gb: float
    memory_utilization: float

    # Ranking score (higher is better)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "expert_parallel": self.expert_parallel,
            "data_parallel": self.data_parallel,
            "total_gpus": self.total_gpus,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "memory_utilization": self.memory_utilization,
            "score": self.score,
            "memory_breakdown": {
                "weights_gb": self.memory_breakdown.weights / (1024**3),
                "activations_gb": self.memory_breakdown.activations / (1024**3),
                "kv_cache_gb": self.memory_breakdown.kv_cache / (1024**3),
                "framework_overhead_gb": self.memory_breakdown.framework_overhead
                / (1024**3),
                "total_gb": self.memory_breakdown.total / (1024**3),
            },
        }


def find_valid_configs(
    model_config: PretrainedConfig | dict[str, Any],
    max_gpus: int,
    gpu_memory_gb: float,
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization_bytes: int = 2,
    framework_overhead_gb: float = 2.0,
    max_configs: int = 50,
) -> list[ParallelismConfig]:
    """Find valid parallelism configurations through grid search.

    Args:
        model_config: Model configuration
        max_gpus: Maximum number of GPUs available
        gpu_memory_gb: Memory per GPU in GB
        sequence_length: Input sequence length
        batch_size: Batch size per GPU
        quantization_bytes: Bytes per parameter
        framework_overhead_gb: Framework overhead in GB
        max_configs: Maximum number of configurations to return

    Returns:
        List of valid ParallelismConfig objects, ranked by score
    """
    # Get valid parallelism sizes for each dimension
    valid_tp_sizes = valid_tensor_parallel_sizes(model_config, max_gpus)
    valid_pp_sizes = valid_pipeline_parallel_sizes(model_config, max_gpus)
    valid_ep_sizes = valid_expert_parallel_sizes(model_config, max_gpus)

    valid_configs = []

    # Generate all combinations of TP x PP x EP
    for tp, pp, ep in itertools.product(valid_tp_sizes, valid_pp_sizes, valid_ep_sizes):
        # Calculate required GPUs for this combination
        required_gpus = tp * pp * ep

        if required_gpus > max_gpus:
            continue

        # Calculate data parallel size to use all available GPUs
        if max_gpus % required_gpus == 0:
            dp = max_gpus // required_gpus
        else:
            # Skip configurations that don't use GPUs efficiently
            continue

        # Validate basic parallelism constraints
        is_valid, errors = validate_parallelism_config(tp, pp, ep, dp, max_gpus)
        if not is_valid:
            continue

        # Estimate memory for this configuration
        try:
            memory_breakdown = estimate_memory_for_config(
                model_config=model_config,
                sequence_length=sequence_length,
                batch_size=batch_size,
                tensor_parallel=tp,
                pipeline_parallel=pp,
                expert_parallel=ep,
                quantization_bytes=quantization_bytes,
                framework_overhead_gb=framework_overhead_gb,
            )
        except Exception:
            # Skip configurations that cause memory estimation errors
            continue

        # Check if memory fits in GPU
        if not memory_breakdown.fits_in_gpu(gpu_memory_gb):
            continue

        # Calculate performance metrics
        memory_per_gpu_gb = memory_breakdown.total / (1024**3)
        memory_utilization = memory_per_gpu_gb / gpu_memory_gb

        # Calculate ranking score
        score = _calculate_config_score(
            tp=tp,
            pp=pp,
            ep=ep,
            dp=dp,
            memory_utilization=memory_utilization,
            total_gpus=max_gpus,
        )

        config = ParallelismConfig(
            tensor_parallel=tp,
            pipeline_parallel=pp,
            expert_parallel=ep,
            data_parallel=dp,
            memory_breakdown=memory_breakdown,
            total_gpus=max_gpus,
            memory_per_gpu_gb=memory_per_gpu_gb,
            memory_utilization=memory_utilization,
            score=score,
        )

        valid_configs.append(config)

    # Sort by score (highest first) and limit results
    valid_configs.sort(key=lambda x: x.score, reverse=True)
    return valid_configs[:max_configs]


def find_best_config(
    model_config: PretrainedConfig | dict[str, Any],
    max_gpus: int,
    gpu_memory_gb: float,
    objective: str = "minimize_gpus",
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization_bytes: int = 2,
    framework_overhead_gb: float = 2.0,
) -> ParallelismConfig:
    """Find single best configuration for given objective.

    Args:
        model_config: Model configuration
        max_gpus: Maximum number of GPUs available
        gpu_memory_gb: Memory per GPU in GB
        objective: Optimization objective ("minimize_gpus", "maximize_throughput",
            "balance")
        sequence_length: Input sequence length
        batch_size: Batch size per GPU
        quantization_bytes: Bytes per parameter
        framework_overhead_gb: Framework overhead in GB

    Returns:
        Best ParallelismConfig for the given objective

    Raises:
        ValueError: If no valid configurations found
    """
    # Get all valid configurations
    valid_configs = find_valid_configs(
        model_config=model_config,
        max_gpus=max_gpus,
        gpu_memory_gb=gpu_memory_gb,
        sequence_length=sequence_length,
        batch_size=batch_size,
        quantization_bytes=quantization_bytes,
        framework_overhead_gb=framework_overhead_gb,
        max_configs=100,  # Get more configs for objective selection
    )

    if not valid_configs:
        raise ValueError(
            "No valid parallelism configurations found for the given constraints"
        )

    # Select best config based on objective
    if objective == "minimize_gpus":
        # Find configuration that uses fewest GPUs
        return _find_min_gpu_config(valid_configs)
    elif objective == "maximize_throughput":
        # Find configuration with best throughput estimate
        return _find_max_throughput_config(valid_configs)
    elif objective == "balance":
        # Find balanced configuration (default scoring)
        return valid_configs[0]  # Already sorted by score
    else:
        raise ValueError(
            f"Unknown objective: {objective}. "
            f"Use 'minimize_gpus', 'maximize_throughput', or 'balance'"
        )


def _calculate_config_score(
    tp: int, pp: int, ep: int, dp: int, memory_utilization: float, total_gpus: int
) -> float:
    """Calculate ranking score for configuration.

    Higher scores are better. Scoring heuristics:
    1. Prefer fewer total GPUs (resource efficiency)
    2. Prefer higher memory utilization (within reasonable bounds)
    3. Prefer balanced parallelism (avoid extreme values)
    4. Penalize excessive pipeline parallelism (bubble overhead)

    Args:
        tp: Tensor parallel size
        pp: Pipeline parallel size
        ep: Expert parallel size
        dp: Data parallel size
        memory_utilization: Memory utilization ratio (0-1)
        total_gpus: Total GPUs used

    Returns:
        Configuration score (higher is better)
    """
    score = 100.0  # Base score

    # Prefer fewer GPUs (resource efficiency)
    gpu_efficiency = total_gpus / (tp * pp * ep * dp)
    score += gpu_efficiency * 20

    # Prefer good memory utilization (60-90% is ideal)
    if 0.6 <= memory_utilization <= 0.9:
        score += 30
    elif 0.4 <= memory_utilization <= 0.95:
        score += 20
    else:
        score -= 10

    # Prefer balanced parallelism sizes
    parallelism_sizes = [tp, pp, ep, dp]
    max_size = max(parallelism_sizes)
    sizes_gt_1 = [s for s in parallelism_sizes if s > 1]

    if sizes_gt_1:  # Only apply balance scoring if there are sizes > 1
        min_size = min(sizes_gt_1)
        balance_ratio = min_size / max_size
        score += balance_ratio * 15

    # Penalize excessive pipeline parallelism (bubble overhead)
    if pp > 4:
        score -= (pp - 4) * 5

    # Slight preference for power-of-2 tensor parallelism
    if tp > 1 and (tp & (tp - 1)) == 0:  # Power of 2
        score += 5

    # Prefer configurations that use all available resources
    resource_utilization = (tp * pp * ep * dp) / total_gpus
    score += resource_utilization * 10

    return score


def _find_min_gpu_config(configs: list[ParallelismConfig]) -> ParallelismConfig:
    """Find configuration that uses minimum GPUs."""
    min_gpus = min(config.total_gpus for config in configs)
    min_gpu_configs = [config for config in configs if config.total_gpus == min_gpus]

    # Among min GPU configs, return the one with highest score
    return max(min_gpu_configs, key=lambda x: x.score)


def _find_max_throughput_config(configs: list[ParallelismConfig]) -> ParallelismConfig:
    """Find configuration with estimated maximum throughput.

    Simple throughput heuristic: prefer more data parallelism and tensor parallelism,
    avoid excessive pipeline parallelism.
    """

    def throughput_score(config: ParallelismConfig) -> float:
        # Simple throughput estimate based on parallelism strategy
        tp_benefit = config.tensor_parallel * 0.8  # TP scales well
        dp_benefit = config.data_parallel * 1.0  # DP scales perfectly
        pp_penalty = (
            config.pipeline_parallel * 0.2 if config.pipeline_parallel > 1 else 0
        )

        return tp_benefit + dp_benefit - pp_penalty

    return max(configs, key=throughput_score)


def get_gpu_requirements(
    model_config: PretrainedConfig | dict[str, Any],
    gpu_memory_gb: float,
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization_bytes: int = 2,
    framework_overhead_gb: float = 2.0,
) -> dict[str, Any]:
    """Get minimum GPU requirements for a model.

    Args:
        model_config: Model configuration
        gpu_memory_gb: Memory per GPU in GB
        sequence_length: Input sequence length
        batch_size: Batch size per GPU
        quantization_bytes: Bytes per parameter
        framework_overhead_gb: Framework overhead in GB

    Returns:
        Dictionary with GPU requirements analysis
    """
    # Estimate base memory (no parallelism)
    from autoparallel.memory import estimate_memory

    base_memory = estimate_memory(
        model_config=model_config,
        sequence_length=sequence_length,
        batch_size=batch_size,
        quantization_bytes=quantization_bytes,
        framework_overhead_gb=framework_overhead_gb,
    )

    # Check if fits in single GPU
    fits_single_gpu = base_memory.fits_in_gpu(gpu_memory_gb)

    if fits_single_gpu:
        min_gpus = 1
        min_memory_gb = base_memory.total / (1024**3)
    else:
        # Estimate minimum GPUs needed using tensor parallelism
        # Start with weights memory as primary constraint
        weights_gb = base_memory.weights / (1024**3)
        gpu_usable_memory = gpu_memory_gb * 0.9  # 90% usable

        # Rough estimate: divide weights by usable memory per GPU
        min_gpus_estimate = max(1, int(weights_gb / gpu_usable_memory) + 1)

        # Try to find a working configuration
        for num_gpus in range(min_gpus_estimate, min_gpus_estimate + 8):
            try:
                configs = find_valid_configs(
                    model_config=model_config,
                    max_gpus=num_gpus,
                    gpu_memory_gb=gpu_memory_gb,
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    quantization_bytes=quantization_bytes,
                    framework_overhead_gb=framework_overhead_gb,
                    max_configs=1,
                )
                if configs:
                    min_gpus = num_gpus
                    min_memory_gb = configs[0].memory_per_gpu_gb
                    break
            except Exception:
                continue
        else:
            # Fallback estimate
            min_gpus = min_gpus_estimate
            min_memory_gb = weights_gb / min_gpus

    return {
        "fits_in_single_gpu": fits_single_gpu,
        "min_gpus_required": min_gpus,
        "memory_per_gpu_gb": min_memory_gb,
        "total_model_memory_gb": base_memory.total / (1024**3),
        "breakdown": {
            "weights_gb": base_memory.weights / (1024**3),
            "activations_gb": base_memory.activations / (1024**3),
            "kv_cache_gb": base_memory.kv_cache / (1024**3),
            "framework_overhead_gb": base_memory.framework_overhead / (1024**3),
        },
    }


def generate_search_space(
    model_config: PretrainedConfig | dict[str, Any],
    max_gpus: int,
    prioritize_efficiency: bool = True,
) -> dict[str, list[int]]:
    """Generate search space for parallelism dimensions.

    Args:
        model_config: Model configuration
        max_gpus: Maximum number of GPUs
        prioritize_efficiency: Whether to prioritize efficient sizes

    Returns:
        Dictionary with search space for each parallelism dimension
    """
    # Get all valid sizes
    valid_tp = valid_tensor_parallel_sizes(model_config, max_gpus)
    valid_pp = valid_pipeline_parallel_sizes(model_config, max_gpus)
    valid_ep = valid_expert_parallel_sizes(model_config, max_gpus)

    if prioritize_efficiency:
        # Filter to power-of-2 and small sizes for efficiency
        valid_tp = [
            tp for tp in valid_tp if tp <= 16 and (tp == 1 or (tp & (tp - 1)) == 0)
        ]
        valid_pp = [pp for pp in valid_pp if pp <= 8]
        valid_ep = [ep for ep in valid_ep if ep <= 8]

    # Data parallel sizes are determined by other dimensions
    max_dp = max_gpus
    valid_dp = list(range(1, max_dp + 1))

    return {
        "tensor_parallel": valid_tp,
        "pipeline_parallel": valid_pp,
        "expert_parallel": valid_ep,
        "data_parallel": valid_dp,
    }
