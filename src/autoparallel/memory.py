"""Simplified memory estimation for autoparallel configurations.

This module provides simple memory estimation using proven heuristics without complex
inheritance hierarchies. Focus on core memory components: weights, activations,
KV cache, and framework overhead.
"""

from dataclasses import dataclass
from typing import Any

try:
    from transformers import PretrainedConfig
except ImportError:
    # Handle case where transformers is not available during testing
    PretrainedConfig = dict  # type: ignore


@dataclass
class MemoryBreakdown:
    """Simple memory breakdown with core components.

    Simplified from the original MemoryComponents to focus on essential components
    for the 95% use case: LLM inference memory estimation.
    """

    weights: int = 0
    """Model weights memory usage in bytes"""

    activations: int = 0
    """Activation memory usage in bytes"""

    kv_cache: int = 0
    """Key-Value cache memory usage in bytes"""

    framework_overhead: int = 0
    """Framework-specific overhead in bytes"""

    @property
    def total(self) -> int:
        """Total memory usage across all components."""
        return self.weights + self.activations + self.kv_cache + self.framework_overhead

    def fits_in_gpu(self, gpu_memory_gb: float) -> bool:
        """Check if memory breakdown fits in GPU memory.

        Args:
            gpu_memory_gb: Available GPU memory in GB

        Returns:
            True if memory fits with safety margin
        """
        gpu_memory_bytes = gpu_memory_gb * (1024**3)
        safety_margin = 0.9  # Use 90% of available memory for safety
        return self.total <= (gpu_memory_bytes * safety_margin)

    def scale_by_parallelism(
        self,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        expert_parallel: int = 1,
    ) -> "MemoryBreakdown":
        """Scale memory breakdown by parallelism strategy.

        Args:
            tensor_parallel: Tensor parallel size
            pipeline_parallel: Pipeline parallel size
            expert_parallel: Expert parallel size

        Returns:
            New MemoryBreakdown with parallelism scaling applied
        """
        # Weights: divide by tensor parallel and expert parallel
        scaled_weights = self.weights // (tensor_parallel * expert_parallel)

        # Activations: divide by tensor parallel and pipeline parallel
        scaled_activations = self.activations // (tensor_parallel * pipeline_parallel)

        # KV cache: divide by tensor parallel
        scaled_kv_cache = self.kv_cache // tensor_parallel

        # Framework overhead: doesn't scale with parallelism
        scaled_framework = self.framework_overhead

        return MemoryBreakdown(
            weights=scaled_weights,
            activations=scaled_activations,
            kv_cache=scaled_kv_cache,
            framework_overhead=scaled_framework,
        )


def estimate_memory(
    model_config: PretrainedConfig | dict[str, Any],
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization_bytes: int = 2,
    enable_kv_cache: bool = True,
    framework_overhead_gb: float = 2.0,
) -> MemoryBreakdown:
    """Estimate memory usage for given model configuration.

    Uses simplified heuristics optimized for ~20% estimation error and <100ms analysis time.

    Args:
        model_config: Hugging Face model configuration or dict
        sequence_length: Input sequence length
        batch_size: Batch size
        quantization_bytes: Bytes per parameter (2 for fp16/bf16, 4 for fp32, 1 for int8)
        enable_kv_cache: Whether to include KV cache estimation
        framework_overhead_gb: Framework overhead in GB (default: 2GB for vLLM)

    Returns:
        MemoryBreakdown with estimated memory components
    """
    # Extract model parameters with defaults
    if hasattr(model_config, "__getattribute__"):
        # PretrainedConfig object
        vocab_size = getattr(model_config, "vocab_size", 32000)
        hidden_size = getattr(model_config, "hidden_size", 4096)
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        num_attention_heads = getattr(model_config, "num_attention_heads", 32)
        num_key_value_heads = getattr(
            model_config, "num_key_value_heads", num_attention_heads
        )
        intermediate_size = getattr(model_config, "intermediate_size", 4 * hidden_size)
        num_experts = getattr(
            model_config, "num_local_experts", getattr(model_config, "num_experts", 0)
        )
    else:
        # Dictionary
        vocab_size = model_config.get("vocab_size", 32000)
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        num_attention_heads = model_config.get("num_attention_heads", 32)
        num_key_value_heads = model_config.get(
            "num_key_value_heads", num_attention_heads
        )
        intermediate_size = model_config.get("intermediate_size", 4 * hidden_size)
        num_experts = model_config.get(
            "num_local_experts", model_config.get("num_experts", 0)
        )

    # Estimate weights memory
    weights_memory = (
        _estimate_param_count(
            vocab_size, hidden_size, num_layers, intermediate_size, num_experts
        )
        * quantization_bytes
    )

    # Estimate activations memory
    activations_memory = (
        _estimate_activations(
            batch_size, sequence_length, hidden_size, num_layers, intermediate_size
        )
        * 2
    )  # fp16 activations

    # Estimate KV cache memory
    if enable_kv_cache:
        kv_cache_memory = (
            _estimate_kv_cache(
                batch_size,
                sequence_length,
                num_layers,
                num_key_value_heads,
                hidden_size // num_attention_heads,
            )
            * 2
        )  # fp16 KV cache
    else:
        kv_cache_memory = 0

    # Framework overhead
    framework_overhead_bytes = int(framework_overhead_gb * (1024**3))

    return MemoryBreakdown(
        weights=weights_memory,
        activations=activations_memory,
        kv_cache=kv_cache_memory,
        framework_overhead=framework_overhead_bytes,
    )


def _estimate_param_count(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
    num_experts: int = 0,
) -> int:
    """Estimate parameter count for transformer model.

    Uses simplified counting for both dense and MoE models.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        intermediate_size: MLP intermediate size
        num_experts: Number of experts (0 for dense models)

    Returns:
        Estimated parameter count
    """
    # Embedding parameters: input + output embeddings
    embedding_params = 2 * vocab_size * hidden_size

    # Per-layer parameters
    # Attention: 4 projections (Q, K, V, O) of size hidden_size x hidden_size
    attention_params = 4 * hidden_size * hidden_size

    # MLP parameters depend on whether this is MoE or dense
    if num_experts > 0:
        # MoE: each expert has its own MLP, plus router
        expert_mlp_params = num_experts * 2 * hidden_size * intermediate_size
        router_params = hidden_size * num_experts
        mlp_params = expert_mlp_params + router_params
    else:
        # Dense: single MLP with up and down projections
        mlp_params = 2 * hidden_size * intermediate_size

    # Layer norms: 2 per layer (pre-attention and pre-MLP)
    layernorm_params = 2 * hidden_size

    # Total per layer
    per_layer_params = attention_params + mlp_params + layernorm_params

    # Final layer norm
    final_layernorm_params = hidden_size

    # Total parameters
    total_params = (
        embedding_params + (num_layers * per_layer_params) + final_layernorm_params
    )

    return total_params


def _estimate_activations(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
) -> int:
    """Estimate activation memory using simplified heuristic.

    Based on empirical analysis of transformer memory usage patterns.
    Uses gradient checkpointing assumption (only store every 4th layer).

    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        hidden_size: Hidden dimension
        num_layers: Number of layers
        intermediate_size: MLP intermediate size

    Returns:
        Estimated activation memory in elements (multiply by dtype bytes)
    """
    # Base activation size per token
    base_activation_size = batch_size * sequence_length * hidden_size

    # Attention scores memory: batch * heads * seq * seq
    # Approximate as batch * seq * hidden (since num_heads * head_dim = hidden)
    attention_scores = batch_size * sequence_length * hidden_size

    # MLP intermediate activations (only for active layers with gradient checkpointing)
    # Assume we store ~4 layers worth of intermediate activations
    stored_layers = min(4, num_layers)
    mlp_intermediate = stored_layers * batch_size * sequence_length * intermediate_size

    # Use empirical scaling factor from vLLM profiling: 0.3x peak activation factor
    empirical_factor = 0.3

    total_activations = (
        base_activation_size + attention_scores + mlp_intermediate
    ) * empirical_factor

    return int(total_activations)


def _estimate_kv_cache(
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    num_key_value_heads: int,
    head_dim: int,
) -> int:
    """Estimate KV cache memory for attention.

    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        num_layers: Number of layers
        num_key_value_heads: Number of key-value heads (for GQA support)
        head_dim: Dimension per attention head

    Returns:
        Estimated KV cache memory in elements (multiply by dtype bytes)
    """
    # KV cache: 2 (K and V) * batch * num_kv_heads * seq_len * head_dim * num_layers
    kv_cache_elements = (
        2 * batch_size * num_key_value_heads * sequence_length * head_dim * num_layers
    )

    return int(kv_cache_elements)


def estimate_memory_for_config(
    model_config: PretrainedConfig | dict[str, Any],
    sequence_length: int = 2048,
    batch_size: int = 1,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    expert_parallel: int = 1,
    quantization_bytes: int = 2,
    framework_overhead_gb: float = 2.0,
) -> MemoryBreakdown:
    """Estimate memory for a specific parallelism configuration.

    Convenience function that estimates base memory and applies parallelism scaling.

    Args:
        model_config: Model configuration
        sequence_length: Input sequence length
        batch_size: Batch size
        tensor_parallel: Tensor parallel size
        pipeline_parallel: Pipeline parallel size
        expert_parallel: Expert parallel size
        quantization_bytes: Bytes per parameter
        framework_overhead_gb: Framework overhead in GB

    Returns:
        MemoryBreakdown scaled for the parallelism configuration
    """
    # Get base memory estimate
    base_memory = estimate_memory(
        model_config=model_config,
        sequence_length=sequence_length,
        batch_size=batch_size,
        quantization_bytes=quantization_bytes,
        framework_overhead_gb=framework_overhead_gb,
    )

    # Apply parallelism scaling
    scaled_memory = base_memory.scale_by_parallelism(
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        expert_parallel=expert_parallel,
    )

    return scaled_memory


def get_quantization_bytes(precision: str) -> int:
    """Get bytes per parameter for different precision formats.

    Args:
        precision: Precision format ("fp32", "fp16", "bf16", "int8", "fp8")

    Returns:
        Bytes per parameter
    """
    precision_map = {
        "fp32": 4,
        "float32": 4,
        "fp16": 2,
        "float16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "int8": 1,
        "fp8": 1,
    }

    return precision_map.get(precision.lower(), 2)  # Default to fp16


def check_memory_feasibility(
    model_config: PretrainedConfig | dict[str, Any],
    gpu_memory_gb: float,
    sequence_length: int = 2048,
    batch_size: int = 1,
    quantization_bytes: int = 2,
    framework_overhead_gb: float = 2.0,
) -> dict[str, Any]:
    """Check if model fits in single GPU memory.

    Args:
        model_config: Model configuration
        gpu_memory_gb: Available GPU memory in GB
        sequence_length: Input sequence length
        batch_size: Batch size
        quantization_bytes: Bytes per parameter
        framework_overhead_gb: Framework overhead in GB

    Returns:
        Dictionary with feasibility analysis
    """
    memory_breakdown = estimate_memory(
        model_config=model_config,
        sequence_length=sequence_length,
        batch_size=batch_size,
        quantization_bytes=quantization_bytes,
        framework_overhead_gb=framework_overhead_gb,
    )

    fits = memory_breakdown.fits_in_gpu(gpu_memory_gb)
    utilization = memory_breakdown.total / (gpu_memory_gb * (1024**3))

    return {
        "fits_in_single_gpu": fits,
        "memory_utilization": utilization,
        "total_memory_gb": memory_breakdown.total / (1024**3),
        "available_memory_gb": gpu_memory_gb,
        "breakdown": {
            "weights_gb": memory_breakdown.weights / (1024**3),
            "activations_gb": memory_breakdown.activations / (1024**3),
            "kv_cache_gb": memory_breakdown.kv_cache / (1024**3),
            "framework_overhead_gb": memory_breakdown.framework_overhead / (1024**3),
        },
    }
