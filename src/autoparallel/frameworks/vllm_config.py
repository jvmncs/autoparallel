"""vLLM configuration system for autotuning and optimization."""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import transformers

from autoparallel.frameworks.vllm_memory import (
    WorkloadProfile,
    get_vllm_default_capture_sizes,
    vLLMAutotuningParameters,
    vLLMMemoryEstimator,
)


@dataclass
class AutotuningParameters:
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

    # Performance scoring weights (tuned based on empirical vLLM performance analysis)
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


def calculate_model_memory_from_config(config: transformers.PretrainedConfig) -> float:
    """Calculate model memory from transformers config without loading weights

    Args:
        config: Transformers model configuration

    Returns:
        Model memory in GB
    """
    # Extract quantization info
    quant_config = getattr(config, "quantization_config", {})
    if quant_config:
        bits = quant_config.get("bits", 16)
    else:
        # Check torch_dtype
        torch_dtype = getattr(config, "torch_dtype", "float32")
        if "float16" in str(torch_dtype) or "bfloat16" in str(torch_dtype):
            bits = 16
        elif "fp8" in str(torch_dtype):
            bits = 8
        else:
            bits = 32

    bytes_per_param = bits / 8

    # Estimate total parameters from config
    vocab_size = getattr(config, "vocab_size", 50257)
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)

    # Rough parameter count estimation
    embedding_params = vocab_size * hidden_size

    # Per-layer parameters (attention + MLP)
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    mlp_params = 2 * hidden_size * intermediate_size  # up, down projections
    layer_norm_params = 2 * hidden_size  # Two layer norms per layer

    layer_params = attention_params + mlp_params + layer_norm_params
    total_params = (
        embedding_params + (num_layers * layer_params) + hidden_size
    )  # Final layer norm

    return (total_params * bytes_per_param) / (1024**3)  # Convert to GB


def estimate_activation_memory_from_config(
    config: transformers.PretrainedConfig, batch_size: int, sequence_length: int
) -> float:
    """Estimate activation memory using CUDA graph memory estimation methodology

    This function estimates peak activation memory during inference by analyzing:
    1. Attention mechanism memory requirements (Q, K, V tensors, attention scores)
    2. MLP intermediate activations
    3. Gradient checkpointing effects during inference

    The conservative estimate factor (0.3) accounts for the fact that not all layers
    are active simultaneously during forward pass, based on empirical memory profiling.

    Args:
        config: Transformers model configuration
        batch_size: Batch size for estimation
        sequence_length: Sequence length for estimation

    Returns:
        Activation memory in GB
    """
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads

    # Attention activations (dominant memory component)
    # Q, K, V tensors: batch_size × seq_len × hidden_size each
    attention_memory = 3 * batch_size * sequence_length * hidden_size * 2  # fp16

    # Attention scores: batch_size × num_heads × seq_len × seq_len
    attention_scores = (
        batch_size * num_attention_heads * sequence_length * sequence_length * 2
    )

    # MLP intermediate activations
    intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)
    mlp_memory = batch_size * sequence_length * intermediate_size * 2

    # Total per layer, accounting for gradient storage during training
    per_layer_memory = attention_memory + attention_scores + mlp_memory

    # Peak memory (not all layers active simultaneously, but need buffer)
    # Conservative estimate based on empirical memory profiling of inference workloads
    peak_activation_memory = per_layer_memory * 0.3

    return peak_activation_memory / (1024**3)  # Convert to GB


@dataclass
class vLLMPerformanceModel:
    """Model for predicting vLLM performance without engine instantiation"""

    # Hardware constraints
    gpu_memory_capacity_gb: float

    # Model-derived parameters (from transformers config)
    model_memory_gb: float
    activation_memory_gb: float
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int  # For GQA models
    vocab_size: int

    # Tunable parameters
    gpu_memory_utilization: float
    cudagraph_capture_sizes: list[int]
    max_model_len: int
    kv_cache_dtype: str  # "auto", "fp8", "fp8_e4m3"

    # Autotuning configuration
    tuning_params: AutotuningParameters

    @classmethod
    def from_transformers_config(
        cls,
        config: transformers.PretrainedConfig,
        gpu_memory_capacity_gb: float,
        gpu_memory_utilization: float,
        max_model_len: int,
        tuning_params: AutotuningParameters | None = None,
        **kwargs: Any,
    ) -> "vLLMPerformanceModel":
        """Create performance model from transformers config

        Args:
            config: Transformers model configuration
            gpu_memory_capacity_gb: GPU memory capacity in GB
            gpu_memory_utilization: GPU memory utilization fraction
            max_model_len: Maximum model length
            tuning_params: Autotuning parameters
            **kwargs: Additional arguments for configuration

        Returns:
            vLLMPerformanceModel instance
        """
        if tuning_params is None:
            tuning_params = AutotuningParameters()

        # Extract architecture parameters from config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(
            config, "num_key_value_heads", num_attention_heads
        )
        vocab_size = getattr(config, "vocab_size", 50257)

        # Calculate model memory from config
        model_memory_gb = calculate_model_memory_from_config(config)

        # Estimate activation memory using CUDA graph memory estimator methodology
        activation_memory_gb = estimate_activation_memory_from_config(
            config, batch_size=32, sequence_length=512
        )

        # Get default values from kwargs or use defaults
        cudagraph_capture_sizes = kwargs.get(
            "cudagraph_capture_sizes", get_vllm_default_capture_sizes(512)
        )
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")

        return cls(
            gpu_memory_capacity_gb=gpu_memory_capacity_gb,
            model_memory_gb=model_memory_gb,
            activation_memory_gb=activation_memory_gb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            gpu_memory_utilization=gpu_memory_utilization,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            tuning_params=tuning_params,
        )

    def calculate_memory_breakdown(self) -> dict[str, float]:
        """Calculate memory allocation without instantiating vLLM

        Returns:
            Memory breakdown dict with components in GB
        """
        # CUDA graph memory calculation
        cuda_graph_memory = 0.0
        for batch_size in self.cudagraph_capture_sizes:
            # Graph memory scales with model size and batch size
            graph_memory = (
                self.model_memory_gb
                * self.tuning_params.graph_memory_overhead_base_ratio
                * (
                    1
                    + batch_size * self.tuning_params.graph_memory_batch_scaling_factor
                )
            )

            # Apply compilation level multiplier based on configured level
            if self.tuning_params.compilation_level == "FULL":
                graph_memory *= self.tuning_params.compilation_memory_multiplier_full
            else:  # PIECEWISE
                graph_memory *= (
                    self.tuning_params.compilation_memory_multiplier_piecewise
                )
            cuda_graph_memory += graph_memory

        # Available memory for KV cache
        total_available = self.gpu_memory_capacity_gb * self.gpu_memory_utilization
        kv_cache_memory = (
            total_available
            - self.model_memory_gb
            - self.activation_memory_gb
            - cuda_graph_memory
        )

        return {
            "model_memory": self.model_memory_gb,
            "activation_memory": self.activation_memory_gb,
            "cuda_graph_memory": cuda_graph_memory,
            "kv_cache_memory": kv_cache_memory,
            "total_used": total_available,
            "utilization_ratio": total_available / self.gpu_memory_capacity_gb,
        }

    def calculate_effective_batch_size(self) -> int:
        """Calculate maximum concurrent sequences given KV cache constraints

        Returns:
            Maximum concurrent sequences
        """
        memory_breakdown = self.calculate_memory_breakdown()
        kv_cache_memory_bytes = memory_breakdown["kv_cache_memory"] * (1024**3)

        if kv_cache_memory_bytes <= 0:
            return 0

        # KV cache per sequence calculation from architecture
        head_dim = self.hidden_size // self.num_attention_heads

        # Determine bytes per element based on dtype
        if self.kv_cache_dtype == "fp8" or self.kv_cache_dtype == "fp8_e4m3":
            dtype_bytes = 1
        else:  # "auto" typically means fp16/bf16
            dtype_bytes = 2

        # Memory per token: keys + values for all layers
        memory_per_token = (
            self.num_key_value_heads * head_dim * 2 * dtype_bytes * self.num_layers
        )

        # Maximum concurrent sequences
        memory_per_sequence = memory_per_token * self.max_model_len
        max_concurrent_seqs = int(kv_cache_memory_bytes / memory_per_sequence)

        return max_concurrent_seqs

    def calculate_graph_coverage(self, workload: WorkloadProfile) -> float:
        """Estimate % of requests that benefit from CUDA graphs

        Args:
            workload: Workload profile with batch size distribution

        Returns:
            Graph coverage ratio (0.0 to 1.0)
        """
        covered_requests = 0
        total_requests = 0

        for batch_size, frequency in workload.batch_size_distribution.items():
            if batch_size in self.cudagraph_capture_sizes:
                covered_requests += frequency
            total_requests += frequency

        return covered_requests / total_requests if total_requests > 0 else 0.0


class vLLMConfigOptimizer:
    """Search for optimal vLLM configurations using performance models"""

    def __init__(
        self,
        model_name: str,
        gpu_memory_capacity_gb: float,
        tuning_params: AutotuningParameters | None = None,
    ):
        """Initialize vLLM config optimizer

        Args:
            model_name: Model name for optimization
            gpu_memory_capacity_gb: GPU memory capacity in GB
            tuning_params: Autotuning parameters
        """
        self.model_name = model_name
        self.gpu_memory_capacity_gb = gpu_memory_capacity_gb
        self.tuning_params = tuning_params or AutotuningParameters()

        # Load model config for analysis
        self.config = transformers.AutoConfig.from_pretrained(model_name)

        # Initialize memory estimator for integration
        self.memory_estimator = vLLMMemoryEstimator(
            autotuning_params=vLLMAutotuningParameters(
                graph_memory_overhead_base_ratio=self.tuning_params.graph_memory_overhead_base_ratio,
                graph_memory_batch_scaling_factor=self.tuning_params.graph_memory_batch_scaling_factor,
                compilation_memory_multiplier_full=self.tuning_params.compilation_memory_multiplier_full,
                compilation_memory_multiplier_piecewise=self.tuning_params.compilation_memory_multiplier_piecewise,
                compilation_level=self.tuning_params.compilation_level,
                min_gpu_memory_utilization=self.tuning_params.min_gpu_memory_utilization,
                max_gpu_memory_utilization=self.tuning_params.max_gpu_memory_utilization,
                throughput_batch_weight=self.tuning_params.throughput_batch_weight,
                throughput_graph_weight=self.tuning_params.throughput_graph_weight,
                latency_graph_weight=self.tuning_params.latency_graph_weight,
                latency_batch_weight=self.tuning_params.latency_batch_weight,
                fragmentation_overhead_factor=self.tuning_params.fragmentation_overhead_factor,
                min_kv_cache_ratio=self.tuning_params.min_kv_cache_ratio,
            )
        )

    def search_optimal_config(
        self, workload: WorkloadProfile, search_space: dict[str, list] | None = None
    ) -> dict[str, Any]:
        """Search for optimal vLLM configuration

        Args:
            workload: Workload profile
            search_space: Configuration search space

        Returns:
            Dictionary with optimal configuration and results
        """
        if search_space is None:
            search_space = self.get_default_search_space(workload)

        best_config = None
        best_score = float("-inf")
        evaluated_configs = []

        # Grid search over configuration space
        for config_params in self.generate_configs(search_space):
            config = vLLMPerformanceModel.from_transformers_config(
                config=self.config,
                gpu_memory_capacity_gb=self.gpu_memory_capacity_gb,
                tuning_params=self.tuning_params,
                **config_params,
            )

            if not self.is_feasible_config(config):
                continue

            score = self.evaluate_config(config, workload)
            evaluated_configs.append((config, score))

            if score > best_score:
                best_score = score
                best_config = config

        return {
            "optimal_config": best_config,
            "performance_score": best_score,
            "memory_breakdown": best_config.calculate_memory_breakdown()
            if best_config
            else None,
            "predictions": self.get_config_predictions(best_config, workload)
            if best_config
            else None,
            "all_evaluated_configs": evaluated_configs,
        }

    def get_default_search_space(self, workload: WorkloadProfile) -> dict[str, list]:
        """Define search space based on workload characteristics

        Args:
            workload: Workload profile

        Returns:
            Search space dictionary
        """
        # Determine reasonable max_model_len based on workload
        max_seq_len = workload.get_expected_max_sequence_length(percentile=0.99)
        max_model_len_options = [
            length
            for length in [1024, 2048, 4096, 8192, 16384]
            if length >= max_seq_len
        ]

        # Determine CUDA graph capture sizes based on workload batch patterns
        max_batch = workload.get_expected_max_batch_size(percentile=0.95)

        # Generate capture size options
        conservative_captures = [1, 2, 4]
        balanced_captures = get_vllm_default_capture_sizes(min(32, max_batch))
        aggressive_captures = get_vllm_default_capture_sizes(min(128, max_batch))

        return {
            "gpu_memory_utilization": [
                self.tuning_params.min_gpu_memory_utilization,
                0.85,
                0.90,
                0.95,
                self.tuning_params.max_gpu_memory_utilization,
            ],
            "cudagraph_capture_sizes": [
                [],  # No CUDA graphs
                conservative_captures,
                balanced_captures,
                aggressive_captures,
            ],
            "max_model_len": max_model_len_options,
            "kv_cache_dtype": (
                ["auto", "fp8_e4m3"]
                if workload.target_metric == "throughput"
                else ["auto"]
            ),
        }

    def generate_configs(
        self, search_space: dict[str, list]
    ) -> Iterator[dict[str, Any]]:
        """Generate all combinations from search space

        Args:
            search_space: Configuration search space

        Yields:
            Configuration dictionaries
        """
        keys = list(search_space.keys())
        values = list(search_space.values())

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination, strict=False))

    def is_feasible_config(self, config: vLLMPerformanceModel) -> bool:
        """Check if configuration fits within memory constraints

        Args:
            config: vLLM performance model configuration

        Returns:
            True if configuration is feasible
        """
        memory_breakdown = config.calculate_memory_breakdown()

        # Check KV cache has minimum required space
        min_kv_cache_gb = (
            config.gpu_memory_capacity_gb * self.tuning_params.min_kv_cache_ratio
        )
        if memory_breakdown["kv_cache_memory"] < min_kv_cache_gb:
            return False

        # Check effective batch size is reasonable
        effective_batch_size = config.calculate_effective_batch_size()
        return effective_batch_size >= 1

    def evaluate_config(
        self, config: vLLMPerformanceModel, workload: WorkloadProfile
    ) -> float:
        """Score configuration based on predicted performance

        Args:
            config: vLLM performance model configuration
            workload: Workload profile

        Returns:
            Performance score (higher is better)
        """
        effective_batch_size = config.calculate_effective_batch_size()
        graph_coverage = config.calculate_graph_coverage(workload)

        if workload.target_metric == "throughput":
            # Throughput score: prioritize batch size, bonus for graph coverage
            score = (
                effective_batch_size * self.tuning_params.throughput_batch_weight
                + graph_coverage
                * effective_batch_size
                * self.tuning_params.throughput_graph_weight
            )

        elif workload.target_metric == "latency":
            # Latency score: prioritize graph coverage, moderate batch size
            # Batch size normalization by 32 represents typical interactive workload scale
            score = (
                graph_coverage * self.tuning_params.latency_graph_weight
                + (effective_batch_size / 32) * self.tuning_params.latency_batch_weight
            )

        else:
            raise ValueError(f"Unknown target metric: {workload.target_metric}")

        return score

    def get_config_predictions(
        self, config: vLLMPerformanceModel, workload: WorkloadProfile
    ) -> dict[str, Any]:
        """Generate detailed predictions for a configuration

        Args:
            config: vLLM performance model configuration
            workload: Workload profile

        Returns:
            Predictions dictionary
        """
        return {
            "effective_batch_size": config.calculate_effective_batch_size(),
            "graph_coverage": config.calculate_graph_coverage(workload),
            "memory_breakdown": config.calculate_memory_breakdown(),
            "recommended_max_num_seqs": min(
                config.calculate_effective_batch_size(),
                workload.get_expected_max_batch_size(percentile=0.95),
            ),
        }

    def validate_configuration(self, config: vLLMPerformanceModel) -> dict[str, Any]:
        """Validate vLLM configuration parameters

        Args:
            config: vLLM performance model configuration

        Returns:
            Validation results with warnings and recommendations
        """
        warnings = []
        recommendations = []
        validation_results = {"valid": True}

        # Memory validation
        memory_breakdown = config.calculate_memory_breakdown()

        # Check if KV cache memory is too low
        kv_cache_ratio = (
            memory_breakdown["kv_cache_memory"] / config.gpu_memory_capacity_gb
        )
        if kv_cache_ratio < 0.1:
            warnings.append("KV cache memory is less than 10% of GPU memory")
            recommendations.append(
                "Consider reducing CUDA graph capture sizes or model length"
            )

        # Check if memory utilization is too aggressive
        if config.gpu_memory_utilization > 0.95:
            warnings.append("Memory utilization above 95% may cause OOM errors")
            recommendations.append("Consider reducing memory utilization to 90-95%")

        # Check CUDA graph efficiency
        if len(config.cudagraph_capture_sizes) > 20:
            warnings.append(
                "Too many CUDA graph capture sizes may increase memory overhead"
            )
            recommendations.append(
                "Consider reducing capture sizes to most common batch sizes"
            )

        # Check effective batch size
        effective_batch_size = config.calculate_effective_batch_size()
        if effective_batch_size < 4:
            warnings.append("Effective batch size is very low, may impact throughput")
            recommendations.append(
                "Consider using fp8 KV cache or reducing max_model_len"
            )

        validation_results.update(
            {
                "warnings": warnings,
                "recommendations": recommendations,
                "effective_batch_size": effective_batch_size,
                "memory_breakdown": memory_breakdown,
            }
        )

        return validation_results


def optimize_vllm_config_for_cluster(
    model_name: str,
    gpu_memory_capacity_gb: float,
    workload: WorkloadProfile,
    parallelism_strategy: dict[str, int],
    tuning_params: AutotuningParameters | None = None,
) -> dict[str, Any]:
    """Integrate vLLM config optimization with autoparallel

    Args:
        model_name: Model name for optimization
        gpu_memory_capacity_gb: GPU memory capacity in GB
        workload: Workload profile
        parallelism_strategy: Parallelism strategy configuration
        tuning_params: Autotuning parameters

    Returns:
        Optimized configuration for cluster deployment
    """
    # Calculate effective resources per vLLM instance
    dp_size = parallelism_strategy["dp"]

    # Memory per vLLM instance
    effective_gpu_memory = gpu_memory_capacity_gb

    # Create optimizer for single vLLM instance
    optimizer = vLLMConfigOptimizer(model_name, effective_gpu_memory, tuning_params)

    # Optimize configuration
    optimal_result = optimizer.search_optimal_config(workload)

    # Scale predictions for full cluster
    cluster_predictions = {
        "total_throughput": optimal_result["predictions"]["effective_batch_size"]
        * dp_size
        if optimal_result["predictions"]
        else 0,
        "instances_per_cluster": dp_size,
        "memory_efficiency": (
            optimal_result["memory_breakdown"]["kv_cache_memory"] / effective_gpu_memory
            if optimal_result["memory_breakdown"]
            else 0
        ),
        "graph_coverage": optimal_result["predictions"]["graph_coverage"]
        if optimal_result["predictions"]
        else 0,
    }

    return {
        "vllm_config": optimal_result["optimal_config"],
        "parallelism_strategy": parallelism_strategy,
        "cluster_predictions": cluster_predictions,
        "recommendations": generate_deployment_recommendations(
            optimal_result, parallelism_strategy
        ),
        "optimization_results": optimal_result,
    }


def generate_deployment_recommendations(
    optimal_result: dict[str, Any], parallelism_strategy: dict[str, int]
) -> list[str]:
    """Generate actionable deployment recommendations

    Args:
        optimal_result: Optimization results
        parallelism_strategy: Parallelism strategy configuration

    Returns:
        List of deployment recommendations
    """
    recommendations = []
    config = optimal_result["optimal_config"]

    if (
        not config
        or not optimal_result["memory_breakdown"]
        or not optimal_result["predictions"]
    ):
        recommendations.append(
            "No feasible configuration found - check memory constraints"
        )
        return recommendations

    # Memory recommendations
    kv_cache_ratio = (
        optimal_result["memory_breakdown"]["kv_cache_memory"]
        / config.gpu_memory_capacity_gb
    )
    if kv_cache_ratio < 0.1:
        recommendations.append(
            "Consider reducing CUDA graph capture sizes to free more memory for KV cache"
        )

    # Performance recommendations
    if optimal_result["predictions"]["graph_coverage"] < 0.5:
        recommendations.append(
            "Low CUDA graph coverage - consider adjusting capture sizes for your workload"
        )

    # Parallelism recommendations
    if (
        parallelism_strategy["tp"] > 4
        and optimal_result["predictions"]["effective_batch_size"] < 8
    ):
        recommendations.append(
            "High tensor parallelism with low batch size may be inefficient"
        )

    # Memory utilization recommendations
    if config.gpu_memory_utilization > 0.95:
        recommendations.append(
            "Memory utilization above 95% - consider reducing to avoid OOM errors"
        )

    return recommendations
