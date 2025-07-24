"""Memory estimation framework for model parallelization."""

from abc import ABC, abstractmethod
from typing import Any

from .components import MemoryComponents
from .config import MemoryConfig


class MemoryEstimator(ABC):
    """Abstract base class for memory estimation.

    This class provides the interface for estimating memory usage
    of transformer models across different parallelism strategies.
    """

    def __init__(self, config: MemoryConfig | None = None):
        """Initialize memory estimator.

        Args:
            config: Memory configuration. Uses defaults if None.
        """
        self.config = config or MemoryConfig()

    @abstractmethod
    def estimate_memory(
        self,
        model_config: dict[str, Any],
        sequence_length: int,
        batch_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        is_training: bool = False,
    ) -> MemoryComponents:
        """Estimate memory usage for the given configuration.

        Args:
            model_config: Model configuration dict (from transformers)
            sequence_length: Input sequence length
            batch_size: Batch size
            tensor_parallel_size: Tensor parallelism size
            pipeline_parallel_size: Pipeline parallelism size
            expert_parallel_size: Expert parallelism size (for MoE)
            is_training: Whether this is for training (affects optimizer memory)

        Returns:
            MemoryComponents with detailed breakdown
        """
        pass

    def estimate_weights_memory(self, model_config: dict[str, Any]) -> int:
        """Estimate model weights memory usage.

        Args:
            model_config: Model configuration dict

        Returns:
            Memory usage in bytes
        """
        # Get model parameters
        vocab_size = model_config.get("vocab_size", 32000)
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        intermediate_size = model_config.get("intermediate_size", 11008)

        # Calculate parameter count
        # Embedding: vocab_size * hidden_size
        embedding_params = vocab_size * hidden_size

        # Each transformer layer:
        # - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
        # - MLP: 2 * hidden_size * intermediate_size (up and down projections)
        # - LayerNorm: 2 * hidden_size (pre and post attention)
        attention_params = 4 * hidden_size * hidden_size
        mlp_params = 2 * hidden_size * intermediate_size
        layernorm_params = 2 * hidden_size

        layer_params = attention_params + mlp_params + layernorm_params
        total_layer_params = num_layers * layer_params

        # Final layer norm and output projection
        final_params = hidden_size + vocab_size * hidden_size

        total_params = embedding_params + total_layer_params + final_params

        # Convert to bytes based on quantization format
        bytes_per_param = self.config.quantization_dtype_bytes
        return int(total_params * bytes_per_param)

    def estimate_activations_memory(
        self, model_config: dict[str, Any], sequence_length: int, batch_size: int
    ) -> int:
        """Estimate activation memory usage.

        Args:
            model_config: Model configuration dict
            sequence_length: Input sequence length
            batch_size: Batch size

        Returns:
            Memory usage in bytes
        """
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        num_attention_heads = model_config.get("num_attention_heads", 32)
        intermediate_size = model_config.get("intermediate_size", 11008)

        # Activation memory per layer:
        # - Input activations: batch_size * sequence_length * hidden_size
        # - Attention scores: batch_size * num_heads * sequence_length^2
        # - MLP intermediate: batch_size * sequence_length * intermediate_size

        input_activations = batch_size * sequence_length * hidden_size
        attention_scores = (
            batch_size * num_attention_heads * sequence_length * sequence_length
        )
        mlp_intermediate = batch_size * sequence_length * intermediate_size

        # Memory for one layer
        layer_activations = input_activations + attention_scores + mlp_intermediate

        # Total activations (assuming gradient checkpointing for some layers)
        # We only need to store activations for a subset of layers
        stored_layers = min(num_layers, 4)  # Store activations for 4 layers max
        total_activations = stored_layers * layer_activations

        # Convert to bytes (assume fp16 for activations)
        bytes_per_element = 2  # fp16
        return int(total_activations * bytes_per_element)

    def estimate_kv_cache_memory(
        self, model_config: dict[str, Any], sequence_length: int, batch_size: int
    ) -> int:
        """Estimate KV cache memory usage.

        Args:
            model_config: Model configuration dict
            sequence_length: Input sequence length
            batch_size: Batch size

        Returns:
            Memory usage in bytes
        """
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        num_key_value_heads = model_config.get("num_key_value_heads", 32)

        # KV cache stores keys and values for each layer
        # Shape: [batch_size, num_kv_heads, sequence_length, head_dim]
        head_dim = hidden_size // model_config.get("num_attention_heads", 32)

        # Keys and values for all layers
        kv_cache_elements = (
            2  # Keys and values
            * batch_size
            * num_key_value_heads
            * sequence_length
            * head_dim
            * num_layers
        )

        # Convert to bytes (assume fp16 for KV cache)
        bytes_per_element = 2  # fp16
        return int(kv_cache_elements * bytes_per_element)

    def estimate_cuda_graphs_memory(self, model_config: dict[str, Any]) -> int:
        """Estimate CUDA graphs memory overhead.

        Args:
            model_config: Model configuration dict

        Returns:
            Memory usage in bytes
        """
        # Fixed overhead from configuration
        overhead_mb = self.config.cuda_graph_overhead_mb
        return int(overhead_mb * 1024 * 1024)  # Convert MB to bytes

    def estimate_optimizer_memory(
        self, model_config: dict[str, Any], is_training: bool = False
    ) -> int:
        """Estimate optimizer state memory usage.

        Args:
            model_config: Model configuration dict
            is_training: Whether this is for training

        Returns:
            Memory usage in bytes
        """
        if not is_training:
            return 0

        # Optimizer memory is typically 2x model weights (Adam optimizer)
        weights_memory = self.estimate_weights_memory(model_config)
        return int(weights_memory * self.config.optimizer_memory_fraction)

    def estimate_fragmentation_overhead(self, base_memory: int) -> int:
        """Estimate memory fragmentation overhead.

        Args:
            base_memory: Base memory usage before overhead

        Returns:
            Fragmentation overhead in bytes
        """
        return int(base_memory * self.config.fragmentation_overhead)

    def estimate_safety_margin(self, base_memory: int) -> int:
        """Estimate safety margin memory.

        Args:
            base_memory: Base memory usage before margin

        Returns:
            Safety margin in bytes
        """
        return int(base_memory * self.config.safety_margin)


class TransformersMemoryEstimator(MemoryEstimator):
    """Memory estimator for Hugging Face transformers models."""

    def estimate_memory(
        self,
        model_config: dict[str, Any],
        sequence_length: int,
        batch_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        is_training: bool = False,
    ) -> MemoryComponents:
        """Estimate memory usage for transformers model.

        Args:
            model_config: Model configuration dict (from transformers)
            sequence_length: Input sequence length
            batch_size: Batch size
            tensor_parallel_size: Tensor parallelism size
            pipeline_parallel_size: Pipeline parallelism size
            expert_parallel_size: Expert parallelism size (for MoE)
            is_training: Whether this is for training

        Returns:
            MemoryComponents with detailed breakdown
        """
        # Calculate base memory components
        weights = self.estimate_weights_memory(model_config)
        activations = self.estimate_activations_memory(
            model_config, sequence_length, batch_size
        )
        kv_cache = self.estimate_kv_cache_memory(
            model_config, sequence_length, batch_size
        )
        cuda_graphs = self.estimate_cuda_graphs_memory(model_config)
        optimizer_states = self.estimate_optimizer_memory(model_config, is_training)

        # Calculate base memory before overhead
        base_memory = weights + activations + kv_cache + cuda_graphs + optimizer_states

        # Calculate overhead
        fragmentation_overhead = self.estimate_fragmentation_overhead(base_memory)
        framework_overhead = int(base_memory * 0.05)  # 5% framework overhead
        safety_margin = self.estimate_safety_margin(base_memory)

        # Create components object
        components = MemoryComponents(
            weights=weights,
            activations=activations,
            kv_cache=kv_cache,
            cuda_graphs=cuda_graphs,
            optimizer_states=optimizer_states,
            fragmentation_overhead=fragmentation_overhead,
            framework_overhead=framework_overhead,
            safety_margin=safety_margin,
        )

        # Scale by parallelism
        return components.scale_by_parallelism(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )


class MoEMemoryEstimator(MemoryEstimator):
    """Memory estimator for Mixture of Experts (MoE) models."""

    def __init__(self, config: MemoryConfig | None = None):
        """Initialize MoE memory estimator.

        Args:
            config: Memory configuration. Uses defaults if None.
        """
        super().__init__(config)
        self._transformer_estimator = TransformersMemoryEstimator(config)

    def estimate_memory(
        self,
        model_config: dict[str, Any],
        sequence_length: int,
        batch_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        is_training: bool = False,
    ) -> MemoryComponents:
        """Estimate memory usage for MoE model.

        For MoE models, we need to account for:
        - Expert parameters (distributed across expert parallel devices)
        - Expert activation patterns
        - Router overhead
        """
        # Get MoE-specific config
        num_experts = model_config.get("num_experts", 8)
        num_experts_per_token = model_config.get("num_experts_per_token", 2)

        # Base transformer estimation
        base_components = self._transformer_estimator.estimate_memory(
            model_config,
            sequence_length,
            batch_size,
            tensor_parallel_size,
            pipeline_parallel_size,
            expert_parallel_size,
            is_training,
        )

        # Calculate expert-specific memory
        expert_memory_factor = num_experts_per_token / num_experts
        expert_memory_per_device = int(
            base_components.weights * expert_memory_factor / expert_parallel_size
        )

        # Router overhead (small compared to experts)
        router_overhead = int(base_components.weights * 0.01)  # 1% overhead

        # Update weights to account for expert distribution
        scaled_weights = expert_memory_per_device + router_overhead

        return MemoryComponents(
            weights=scaled_weights,
            activations=base_components.activations,
            kv_cache=base_components.kv_cache,
            cuda_graphs=base_components.cuda_graphs,
            optimizer_states=int(
                base_components.optimizer_states / expert_parallel_size
            ),
            fragmentation_overhead=base_components.fragmentation_overhead,
            framework_overhead=base_components.framework_overhead + router_overhead,
            safety_margin=base_components.safety_margin,
        )
