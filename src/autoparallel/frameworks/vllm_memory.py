"""vLLM-specific memory estimation for autoparallel optimization."""

from dataclasses import dataclass, field
from typing import Any

from autoparallel.memory.components import MemoryComponents
from autoparallel.memory.config import MemoryConfig
from autoparallel.memory.estimator import MemoryEstimator


@dataclass
class vLLMAutotuningParameters:
    """Configurable parameters for vLLM autotuning behavior"""

    # CUDA graph memory estimation parameters
    graph_memory_overhead_base_ratio: float = 0.1
    """Base overhead as fraction of model memory"""

    graph_memory_batch_scaling_factor: float = 0.02
    """Additional memory per batch element"""

    compilation_memory_multiplier_full: float = 1.8
    """Memory multiplier for FULL compilation"""

    compilation_memory_multiplier_piecewise: float = 1.0
    """Memory multiplier for PIECEWISE compilation"""

    compilation_level: str = "PIECEWISE"
    """Default compilation level: "FULL" or "PIECEWISE\""""

    # Memory utilization bounds
    min_gpu_memory_utilization: float = 0.8
    """Conservative lower bound"""

    max_gpu_memory_utilization: float = 0.98
    """Aggressive upper bound"""

    # Performance scoring weights
    throughput_batch_weight: float = 0.7
    """Weight for batch size in throughput scoring"""

    throughput_graph_weight: float = 0.3
    """Weight for graph coverage in throughput scoring"""

    latency_graph_weight: float = 0.8
    """Weight for graph coverage in latency scoring"""

    latency_batch_weight: float = 0.2
    """Weight for batch size in latency scoring"""

    # Memory safety margins
    fragmentation_overhead_factor: float = 1.05
    """Account for memory fragmentation"""

    min_kv_cache_ratio: float = 0.05
    """Minimum KV cache as fraction of GPU memory"""


@dataclass
class WorkloadProfile:
    """Characterize expected vLLM workload patterns"""

    # Request patterns
    requests_per_second: float
    batch_size_distribution: dict[int, float] = field(default_factory=dict)
    sequence_length_distribution: dict[int, float] = field(default_factory=dict)

    # Performance priorities
    target_metric: str = "throughput"
    latency_budget_ms: float = 100.0

    @classmethod
    def create_synthetic(
        cls,
        workload_type: str,
        requests_per_second: float | None = None,
        latency_budget_ms: float | None = None,
    ) -> "WorkloadProfile":
        """Create synthetic workload profiles for common scenarios"""

        if workload_type == "chatbot":
            return cls(
                requests_per_second=requests_per_second or 100,
                batch_size_distribution={1: 0.4, 2: 0.3, 4: 0.2, 8: 0.1},
                sequence_length_distribution={512: 0.3, 1024: 0.4, 2048: 0.3},
                target_metric="latency",
                latency_budget_ms=latency_budget_ms or 100,
            )
        elif workload_type == "batch_inference":
            return cls(
                requests_per_second=requests_per_second or 10,
                batch_size_distribution={16: 0.2, 32: 0.4, 64: 0.3, 128: 0.1},
                sequence_length_distribution={1024: 0.5, 2048: 0.4, 4096: 0.1},
                target_metric="throughput",
                latency_budget_ms=latency_budget_ms or 1000,
            )
        elif workload_type == "interactive":
            return cls(
                requests_per_second=requests_per_second or 50,
                batch_size_distribution={1: 0.8, 2: 0.15, 4: 0.05},
                sequence_length_distribution={256: 0.4, 512: 0.4, 1024: 0.2},
                target_metric="latency",
                latency_budget_ms=latency_budget_ms or 50,
            )

        raise ValueError(f"Unknown workload type: {workload_type}")

    def get_expected_max_batch_size(self, percentile: float = 0.95) -> int:
        """Get expected maximum batch size at given percentile"""
        if not self.batch_size_distribution:
            return 32  # Default fallback

        cumulative = 0.0
        sorted_batches = sorted(self.batch_size_distribution.items())

        for batch_size, frequency in sorted_batches:
            cumulative += frequency
            if cumulative >= percentile:
                return batch_size

        return max(self.batch_size_distribution.keys())

    def get_expected_max_sequence_length(self, percentile: float = 0.95) -> int:
        """Get expected maximum sequence length at given percentile"""
        if not self.sequence_length_distribution:
            return 2048  # Default fallback

        cumulative = 0.0
        sorted_lengths = sorted(self.sequence_length_distribution.items())

        for seq_len, frequency in sorted_lengths:
            cumulative += frequency
            if cumulative >= percentile:
                return seq_len

        return max(self.sequence_length_distribution.keys())


def get_vllm_default_capture_sizes(
    max_limit: int, vllm_version: str = "v1"
) -> list[int]:
    """Get vLLM's actual default CUDA graph capture sizes

    Args:
        max_limit: Maximum capture size to include
        vllm_version: vLLM version ("v0" or "v1") - controls whether max_limit
                     applies to max_num_seqs (v0) or max_num_batched_tokens (v1)
    """

    # vllm/config.py: batch_size_capture_list = [1, 2, 4] + list(range(8, 513, 8))
    base_sizes = [1, 2, 4] + list(range(8, 513, 8))

    # Filter by version-specific limits
    return [size for size in base_sizes if size <= max_limit]


class vLLMMemoryEstimator(MemoryEstimator):
    """Memory estimator specifically designed for vLLM inference engines.

    This estimator implements vLLM-specific memory calculations including:
    - Advanced KV cache sizing with different data types (fp16, fp8)
    - CUDA graph memory estimation based on capture sizes
    - Effective batch size calculation based on memory constraints
    - Graph coverage calculation for workload-based optimization
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        autotuning_params: vLLMAutotuningParameters | None = None,
    ):
        """Initialize vLLM memory estimator.

        Args:
            config: Memory configuration. Uses defaults if None.
            autotuning_params: vLLM autotuning parameters. Uses defaults if None.
        """
        super().__init__(config)
        self.autotuning_params = autotuning_params or vLLMAutotuningParameters()

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
        """Estimate memory usage for vLLM configuration.

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
        # Calculate base memory components
        weights = self.estimate_weights_memory(model_config)
        activations = self.estimate_vllm_activations_memory(
            model_config, sequence_length, batch_size
        )
        kv_cache = self.estimate_vllm_kv_cache_memory(
            model_config, sequence_length, batch_size
        )
        cuda_graphs = self.estimate_vllm_cuda_graphs_memory(
            model_config, get_vllm_default_capture_sizes(batch_size)
        )
        optimizer_states = self.estimate_optimizer_memory(model_config, is_training)

        # Calculate base memory before overhead
        base_memory = weights + activations + kv_cache + cuda_graphs + optimizer_states

        # Calculate vLLM-specific overhead
        fragmentation_overhead = self.estimate_fragmentation_overhead(base_memory)
        framework_overhead = self.estimate_vllm_framework_overhead(base_memory)
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

    def estimate_vllm_activations_memory(
        self, model_config: dict[str, Any], sequence_length: int, batch_size: int
    ) -> int:
        """Estimate vLLM-specific activation memory usage.

        Uses vLLM's approach of profiling with meta tensors and accounts for
        inference-specific memory patterns.

        Args:
            model_config: Model configuration dict
            sequence_length: Input sequence length
            batch_size: Batch size

        Returns:
            Memory usage in bytes
        """
        hidden_size = model_config.get("hidden_size", 4096)
        num_attention_heads = model_config.get("num_attention_heads", 32)
        intermediate_size = model_config.get("intermediate_size", 11008)

        # vLLM-specific activation calculation
        # Attention activations (Q, K, V tensors)
        attention_memory = 3 * batch_size * sequence_length * hidden_size * 2  # fp16

        # Attention scores: batch_size × num_heads × seq_len × seq_len
        attention_scores = (
            batch_size * num_attention_heads * sequence_length * sequence_length * 2
        )

        # MLP intermediate activations
        mlp_memory = batch_size * sequence_length * intermediate_size * 2

        # Total per layer
        per_layer_memory = attention_memory + attention_scores + mlp_memory

        # Peak memory (conservative estimate based on empirical profiling)
        # vLLM inference doesn't need all layer activations simultaneously
        peak_activation_memory = per_layer_memory * 0.3

        return int(peak_activation_memory)

    def estimate_vllm_kv_cache_memory(
        self,
        model_config: dict[str, Any],
        sequence_length: int,
        batch_size: int,
        kv_cache_dtype: str = "auto",
    ) -> int:
        """Estimate vLLM KV cache memory with support for different data types.

        Args:
            model_config: Model configuration dict
            sequence_length: Input sequence length
            batch_size: Batch size
            kv_cache_dtype: KV cache data type ("auto", "fp8", "fp8_e4m3")

        Returns:
            Memory usage in bytes
        """
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        num_attention_heads = model_config.get("num_attention_heads", 32)
        num_key_value_heads = model_config.get(
            "num_key_value_heads", num_attention_heads
        )

        # Calculate head dimension
        head_dim = hidden_size // num_attention_heads

        # Determine bytes per element based on dtype
        if kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3":
            dtype_bytes = 1
        else:  # "auto" typically means fp16/bf16
            dtype_bytes = 2

        # KV cache stores keys and values for each layer
        # Shape: [batch_size, num_kv_heads, sequence_length, head_dim]
        kv_cache_elements = (
            2  # Keys and values
            * batch_size
            * num_key_value_heads
            * sequence_length
            * head_dim
            * num_layers
        )

        return int(kv_cache_elements * dtype_bytes)

    def estimate_vllm_cuda_graphs_memory(
        self, model_config: dict[str, Any], capture_sizes: list[int]
    ) -> int:
        """Estimate vLLM CUDA graphs memory based on capture sizes.

        Args:
            model_config: Model configuration dict
            capture_sizes: List of batch sizes to capture in CUDA graphs

        Returns:
            Memory usage in bytes
        """
        if not capture_sizes:
            return 0

        # Calculate model memory for base calculations
        model_memory_bytes = self.estimate_weights_memory(model_config)

        total_graph_memory = 0
        for batch_size in capture_sizes:
            # Graph memory scales with model size and batch size
            graph_memory = (
                model_memory_bytes
                * self.autotuning_params.graph_memory_overhead_base_ratio
                * (
                    1
                    + batch_size
                    * self.autotuning_params.graph_memory_batch_scaling_factor
                )
            )

            # Apply compilation level multiplier
            if self.autotuning_params.compilation_level == "FULL":
                graph_memory *= (
                    self.autotuning_params.compilation_memory_multiplier_full
                )
            else:  # PIECEWISE
                graph_memory *= (
                    self.autotuning_params.compilation_memory_multiplier_piecewise
                )

            total_graph_memory += graph_memory

        return int(total_graph_memory)

    def estimate_vllm_framework_overhead(self, base_memory: int) -> int:
        """Estimate vLLM-specific framework overhead.

        Args:
            base_memory: Base memory usage before overhead

        Returns:
            Framework overhead in bytes
        """
        # vLLM has additional overhead for:
        # - Block manager
        # - Scheduler state
        # - Request tracking
        # - PagedAttention workspace
        vllm_overhead_factor = 0.08  # 8% overhead based on empirical observations
        return int(base_memory * vllm_overhead_factor)

    def calculate_effective_batch_size(
        self,
        model_config: dict[str, Any],
        max_model_len: int,
        gpu_memory_capacity_bytes: int,
        gpu_memory_utilization: float = 0.9,
        kv_cache_dtype: str = "auto",
    ) -> int:
        """Calculate maximum concurrent sequences given KV cache constraints.

        This implements vLLM's memory allocation priority:
        1. model_weights (fixed)
        2. activation_memory (measured)
        3. cuda_graph_memory (calculated)
        4. kv_cache_memory (remaining)

        Args:
            model_config: Model configuration dict
            max_model_len: Maximum sequence length
            gpu_memory_capacity_bytes: Total GPU memory in bytes
            gpu_memory_utilization: GPU memory utilization fraction
            kv_cache_dtype: KV cache data type

        Returns:
            Maximum concurrent sequences
        """
        # Calculate fixed memory components
        model_memory = self.estimate_weights_memory(model_config)
        activation_memory = self.estimate_vllm_activations_memory(
            model_config,
            max_model_len,
            1,  # Use batch_size=1 for baseline
        )

        # Use default capture sizes for CUDA graph estimation
        default_capture_sizes = get_vllm_default_capture_sizes(64)  # Reasonable default
        cuda_graph_memory = self.estimate_vllm_cuda_graphs_memory(
            model_config, default_capture_sizes
        )

        # Calculate available memory for KV cache
        total_available = gpu_memory_capacity_bytes * gpu_memory_utilization
        kv_cache_memory_available = (
            total_available - model_memory - activation_memory - cuda_graph_memory
        )

        if kv_cache_memory_available <= 0:
            return 0

        # Calculate memory per sequence
        hidden_size = model_config.get("hidden_size", 4096)
        num_layers = model_config.get("num_hidden_layers", 32)
        num_attention_heads = model_config.get("num_attention_heads", 32)
        num_key_value_heads = model_config.get(
            "num_key_value_heads", num_attention_heads
        )

        head_dim = hidden_size // num_attention_heads

        # Determine bytes per element based on dtype
        if kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3":
            dtype_bytes = 1
        else:  # "auto" typically means fp16/bf16
            dtype_bytes = 2

        # Memory per token: keys + values for all layers
        memory_per_token = num_key_value_heads * head_dim * 2 * dtype_bytes * num_layers

        # Maximum concurrent sequences
        memory_per_sequence = memory_per_token * max_model_len
        max_concurrent_seqs = int(kv_cache_memory_available / memory_per_sequence)

        return max_concurrent_seqs

    def calculate_graph_coverage(
        self, workload: WorkloadProfile, capture_sizes: list[int]
    ) -> float:
        """Estimate percentage of requests that benefit from CUDA graphs.

        Args:
            workload: Workload profile with batch size distribution
            capture_sizes: List of captured batch sizes

        Returns:
            Graph coverage ratio (0.0 to 1.0)
        """
        if not capture_sizes or not workload.batch_size_distribution:
            return 0.0

        covered_requests = 0
        total_requests = 0

        for batch_size, frequency in workload.batch_size_distribution.items():
            if batch_size in capture_sizes:
                covered_requests += frequency
            total_requests += frequency

        return covered_requests / total_requests if total_requests > 0 else 0.0

    def calculate_memory_breakdown(
        self,
        model_config: dict[str, Any],
        gpu_memory_capacity_gb: float,
        gpu_memory_utilization: float,
        capture_sizes: list[int],
        max_model_len: int = 2048,
    ) -> dict[str, float]:
        """Calculate detailed memory allocation breakdown.

        Args:
            model_config: Model configuration dict
            gpu_memory_capacity_gb: GPU memory capacity in GB
            gpu_memory_utilization: Memory utilization fraction
            capture_sizes: CUDA graph capture sizes
            max_model_len: Maximum model length

        Returns:
            Memory breakdown dict with components in GB
        """
        # Calculate components in bytes
        model_memory = self.estimate_weights_memory(model_config)
        activation_memory = self.estimate_vllm_activations_memory(
            model_config, max_model_len, 32
        )
        cuda_graph_memory = self.estimate_vllm_cuda_graphs_memory(
            model_config, capture_sizes
        )

        # Available memory for KV cache
        total_available_bytes = (
            gpu_memory_capacity_gb * (1024**3) * gpu_memory_utilization
        )
        kv_cache_memory = (
            total_available_bytes - model_memory - activation_memory - cuda_graph_memory
        )

        # Convert to GB
        return {
            "model_memory": model_memory / (1024**3),
            "activation_memory": activation_memory / (1024**3),
            "cuda_graph_memory": cuda_graph_memory / (1024**3),
            "kv_cache_memory": max(0, kv_cache_memory) / (1024**3),
            "total_used": total_available_bytes / (1024**3),
            "utilization_ratio": gpu_memory_utilization,
        }

    def evaluate_config_performance(
        self,
        model_config: dict[str, Any],
        workload: WorkloadProfile,
        gpu_memory_capacity_gb: float,
        gpu_memory_utilization: float,
        capture_sizes: list[int],
        max_model_len: int = 2048,
    ) -> float:
        """Evaluate vLLM configuration performance score.

        Args:
            model_config: Model configuration dict
            workload: Workload profile
            gpu_memory_capacity_gb: GPU memory capacity in GB
            gpu_memory_utilization: Memory utilization fraction
            capture_sizes: CUDA graph capture sizes
            max_model_len: Maximum model length

        Returns:
            Performance score (higher is better)
        """
        # Calculate effective batch size
        effective_batch_size = self.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=max_model_len,
            gpu_memory_capacity_bytes=int(gpu_memory_capacity_gb * (1024**3)),
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Calculate graph coverage
        graph_coverage = self.calculate_graph_coverage(workload, capture_sizes)

        if workload.target_metric == "throughput":
            # Throughput score: prioritize batch size, bonus for graph coverage
            score = (
                effective_batch_size * self.autotuning_params.throughput_batch_weight
                + graph_coverage
                * effective_batch_size
                * self.autotuning_params.throughput_graph_weight
            )
        elif workload.target_metric == "latency":
            # Latency score: prioritize graph coverage, moderate batch size
            score = (
                graph_coverage * self.autotuning_params.latency_graph_weight
                + (effective_batch_size / 32)
                * self.autotuning_params.latency_batch_weight
            )
        else:
            raise ValueError(f"Unknown target metric: {workload.target_metric}")

        return score
