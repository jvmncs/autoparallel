"""Simple public API for autoparallel library."""

from dataclasses import dataclass
from typing import Any

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    transformers = None  # type: ignore
    AutoConfig = None  # type: ignore

from autoparallel.config.generator import (
    ConfigurationGenerator,
    ScoredConfiguration,
)
from autoparallel.config.optimizer import (
    HardwareProfile,
    OptimizationObjective,
    WorkloadProfile,
    WorkloadType,
)
from autoparallel.constraints.analyzer import analyze_model_constraints
from autoparallel.memory.estimator import (
    MoEMemoryEstimator,
    TransformersMemoryEstimator,
)


@dataclass
class AnalysisResult:
    """Result of analyzing parallelization options for a model."""

    model_name: str
    """The model that was analyzed"""

    cluster_info: dict[str, Any]
    """Hardware cluster specification used"""

    total_configurations: int
    """Total number of valid configurations found"""

    configurations: list[dict[str, Any]]
    """List of valid configurations with details"""

    recommendations: dict[str, Any]
    """High-level recommendations and insights"""

    def __post_init__(self) -> None:
        """Validate analysis result."""
        if self.total_configurations != len(self.configurations):
            raise ValueError(
                f"Mismatch between total_configurations ({self.total_configurations}) "
                f"and configurations list length ({len(self.configurations)})"
            )


@dataclass
class OptimizedConfig:
    """Result of optimizing for a specific workload."""

    model_name: str
    """The model that was optimized"""

    workload_type: str
    """The workload type this was optimized for"""

    best_configuration: dict[str, Any]
    """The recommended configuration"""

    performance_estimate: dict[str, Any]
    """Expected performance metrics"""

    deployment_command: str
    """Ready-to-run deployment command (if available)"""

    alternative_configurations: list[dict[str, Any]]
    """Alternative configurations ranked by performance"""


def _convert_cluster_dict_to_hardware_profile(
    cluster: dict[str, Any],
) -> HardwareProfile:
    """Convert simple cluster dict to HardwareProfile."""
    return HardwareProfile(
        gpu_memory_gb=cluster.get("gpu_memory_gb", 80.0),
        gpus_per_node=cluster.get("gpus_per_node", 8),
        num_nodes=cluster.get("num_nodes", 1),
        gpu_model=cluster.get("gpu_model", "H100"),
        intra_node_bandwidth_gbps=cluster.get("intra_node_bandwidth_gbps", 900.0),
        inter_node_bandwidth_gbps=cluster.get("inter_node_bandwidth_gbps", 200.0),
        network_topology=cluster.get("network_topology", "fat_tree"),
    )


def _convert_workload_string_to_profile(workload: str) -> WorkloadProfile:
    """Convert workload string to WorkloadProfile with defaults."""
    workload_type_map = {
        "inference": WorkloadType.INFERENCE,
        "training": WorkloadType.TRAINING,
        "batch_processing": WorkloadType.BATCH_PROCESSING,
        "interactive": WorkloadType.INTERACTIVE,
        "chatbot": WorkloadType.CHATBOT,
    }

    workload_type = workload_type_map.get(workload.lower(), WorkloadType.INFERENCE)

    # Set reasonable defaults based on workload type
    if workload_type == WorkloadType.INFERENCE:
        return WorkloadProfile(
            workload_type=workload_type,
            batch_size=32,
            sequence_length=2048,
            requests_per_second=100.0,
            latency_budget_ms=100.0,
            throughput_target=1000.0,
            is_training=False,
        )
    elif workload_type == WorkloadType.TRAINING:
        return WorkloadProfile(
            workload_type=workload_type,
            batch_size=64,
            sequence_length=4096,
            requests_per_second=10.0,
            latency_budget_ms=1000.0,
            throughput_target=500.0,
            is_training=True,
        )
    elif workload_type == WorkloadType.CHATBOT:
        return WorkloadProfile(
            workload_type=workload_type,
            batch_size=16,
            sequence_length=1024,
            requests_per_second=50.0,
            latency_budget_ms=150.0,
            throughput_target=800.0,
            is_training=False,
        )
    else:
        return WorkloadProfile(
            workload_type=workload_type,
            batch_size=32,
            sequence_length=2048,
            requests_per_second=100.0,
            latency_budget_ms=100.0,
            throughput_target=1000.0,
            is_training=workload_type == WorkloadType.TRAINING,
        )


def _load_model_config(model: str) -> Any:
    """Load model configuration from HuggingFace or local path."""
    if transformers is None or AutoConfig is None:
        raise ImportError(
            "transformers library is required for model configuration loading"
        )

    try:
        # Try loading from HuggingFace or local path
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        return config  # Return the config object directly, not as dict
    except Exception as e:
        raise ValueError(
            f"Failed to load model configuration for '{model}': {e}"
        ) from e


def _convert_scored_config_to_dict(
    scored_config: ScoredConfiguration,
) -> dict[str, Any]:
    """Convert ScoredConfiguration to dictionary for public API."""
    config = scored_config.configuration
    metrics = scored_config.performance_metrics
    memory = scored_config.memory_components

    return {
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "expert_parallel_size": config.expert_parallel_size,
        "data_parallel_size": config.data_parallel_size,
        "total_gpus": config.total_gpus_needed,
        "performance": {
            "throughput_tokens_per_second": metrics.throughput_score
            * 1000,  # Convert score to approximate tokens/sec
            "latency_ms": 1000
            / max(metrics.latency_score, 0.001),  # Convert latency score to ms
            "memory_utilization": metrics.memory_utilization_gb_per_gpu,
            "cost_per_hour": 100
            - metrics.cost_score,  # Convert cost score to approximate $/hour
            "communication_efficiency": metrics.communication_efficiency,
            "is_feasible": metrics.is_feasible,
        },
        "memory_breakdown": {
            "total_memory_gb": memory.total_memory / (1024**3),  # Convert bytes to GB
            "weights_gb": memory.weights / (1024**3),
            "activations_gb": memory.activations / (1024**3),
            "kv_cache_gb": memory.kv_cache / (1024**3),
            "cuda_graphs_gb": memory.cuda_graphs / (1024**3),
            "optimizer_states_gb": memory.optimizer_states / (1024**3),
            "fragmentation_overhead_gb": memory.fragmentation_overhead / (1024**3),
        },
        "is_valid": scored_config.is_valid,
    }


def analyze(model: str, cluster: dict[str, Any]) -> AnalysisResult:
    """Analyze all viable parallelization configurations for a model.

    Args:
        model: Model identifier (HuggingFace model name or local path)
        cluster: Hardware cluster specification dict with keys:
            - gpu_memory_gb: GPU memory per device in GB
            - gpus_per_node: Number of GPUs per node
            - num_nodes: Number of nodes in cluster
            - gpu_model (optional): GPU model name
            - intra_node_bandwidth_gbps (optional): Bandwidth within node
            - inter_node_bandwidth_gbps (optional): Bandwidth between nodes

    Returns:
        AnalysisResult with all viable configurations and recommendations

    Example:
        >>> result = analyze(
        ...     model="meta-llama/Llama-2-7b-hf",
        ...     cluster={"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 1}
        ... )
        >>> print(f"Found {result.total_configurations} valid configurations")
        >>> best_config = result.configurations[0]
        >>> print(f"Best TP size: {best_config['tensor_parallel_size']}")
    """
    # Load model configuration
    model_config = _load_model_config(model)

    # Convert inputs to internal types
    hardware_profile = _convert_cluster_dict_to_hardware_profile(cluster)

    # Use inference workload for analysis (neutral optimization)
    workload_profile = WorkloadProfile(
        workload_type=WorkloadType.INFERENCE,
        batch_size=32,
        sequence_length=2048,
        requests_per_second=100.0,
        latency_budget_ms=100.0,
        throughput_target=1000.0,
        is_training=False,
    )

    # Analyze model constraints
    model_constraints = analyze_model_constraints(model_config)

    # Choose appropriate memory estimator based on model type
    is_moe = any(
        hasattr(model_config, attr)
        for attr in ["num_local_experts", "moe_intermediate_size"]
    )
    memory_estimator = MoEMemoryEstimator() if is_moe else TransformersMemoryEstimator()

    # Generate configurations
    generator = ConfigurationGenerator(memory_estimator=memory_estimator)
    result = generator.generate_valid_configs(
        model_config=model_config.to_dict(),
        hardware_profile=hardware_profile,
        workload_profile=workload_profile,
        model_constraints=model_constraints,
        max_configurations=100,
        optimization_objective=OptimizationObjective.BALANCE_EFFICIENCY,
    )

    # Convert to public API format
    configurations = [
        _convert_scored_config_to_dict(config) for config in result.configurations
    ]

    # Generate recommendations
    model_dict = model_config.to_dict()
    recommendations = {
        "model_type": "MoE" if is_moe else "Dense Transformer",
        "total_parameters": model_dict.get("n_parameters", "Unknown"),
        "memory_constraints": {
            "minimum_gpus_required": min(
                config["total_gpus"] for config in configurations
            )
            if configurations
            else hardware_profile.total_gpus,
            "maximum_memory_utilization": max(
                config["performance"]["memory_utilization"] for config in configurations
            )
            if configurations
            else 0.0,
        },
        "parallelism_insights": {
            "max_tensor_parallel": model_constraints.max_tensor_parallel,
            "max_pipeline_parallel": model_constraints.max_pipeline_parallel,
            "max_expert_parallel": model_constraints.max_expert_parallel,
            "supports_gqa": model_constraints.supports_grouped_query_attention,
        },
        "best_configuration_summary": (
            f"TP={configurations[0]['tensor_parallel_size']}, "
            f"PP={configurations[0]['pipeline_parallel_size']}, "
            f"EP={configurations[0]['expert_parallel_size']}, "
            f"DP={configurations[0]['data_parallel_size']}"
        )
        if configurations
        else "No valid configurations found",
    }

    return AnalysisResult(
        model_name=model,
        cluster_info=cluster,
        total_configurations=len(configurations),
        configurations=configurations,
        recommendations=recommendations,
    )


def optimize(model: str, cluster: dict[str, Any], workload: str) -> OptimizedConfig:
    """Find optimal parallelization configuration for a specific workload.

    Args:
        model: Model identifier (HuggingFace model name or local path)
        cluster: Hardware cluster specification dict (same as analyze())
        workload: Workload type string ("inference", "training", "chatbot",
                  "batch_processing", "interactive")

    Returns:
        OptimizedConfig with best configuration and deployment information

    Example:
        >>> result = optimize(
        ...     model="meta-llama/Llama-2-70b-hf",
        ...     cluster={"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 4},
        ...     workload="inference"
        ... )
        >>> print(f"Best configuration: {result.best_configuration}")
        >>> print(f"Expected throughput: {result.performance_estimate['throughput_tokens_per_second']} tokens/sec")
    """
    # Load model configuration
    model_config = _load_model_config(model)

    # Convert inputs to internal types
    hardware_profile = _convert_cluster_dict_to_hardware_profile(cluster)
    workload_profile = _convert_workload_string_to_profile(workload)

    # Analyze model constraints
    model_constraints = analyze_model_constraints(model_config)

    # Choose appropriate memory estimator based on model type
    is_moe = any(
        hasattr(model_config, attr)
        for attr in ["num_local_experts", "moe_intermediate_size"]
    )
    memory_estimator = MoEMemoryEstimator() if is_moe else TransformersMemoryEstimator()

    # Determine optimization objective based on workload
    objective_map = {
        "inference": OptimizationObjective.MAXIMIZE_THROUGHPUT,
        "training": OptimizationObjective.MAXIMIZE_THROUGHPUT,
        "chatbot": OptimizationObjective.MINIMIZE_LATENCY,
        "interactive": OptimizationObjective.MINIMIZE_LATENCY,
        "batch_processing": OptimizationObjective.MAXIMIZE_THROUGHPUT,
    }
    optimization_objective = objective_map.get(
        workload.lower(), OptimizationObjective.BALANCE_EFFICIENCY
    )

    # Generate and optimize configurations
    generator = ConfigurationGenerator(memory_estimator=memory_estimator)
    result = generator.generate_valid_configs(
        model_config=model_config.to_dict(),
        hardware_profile=hardware_profile,
        workload_profile=workload_profile,
        model_constraints=model_constraints,
        max_configurations=50,  # Fewer configs for optimization focus
        optimization_objective=optimization_objective,
    )

    if not result.configurations:
        raise RuntimeError(
            f"No valid configurations found for model '{model}' on given cluster"
        )

    # Get best configuration
    best_scored_config = result.configurations[0]
    best_config = _convert_scored_config_to_dict(best_scored_config)

    # Generate performance estimates
    performance_estimate = {
        "throughput_tokens_per_second": best_config["performance"][
            "throughput_tokens_per_second"
        ],
        "latency_ms": best_config["performance"]["latency_ms"],
        "memory_utilization": best_config["performance"]["memory_utilization"],
        "cost_per_hour": best_config["performance"]["cost_per_hour"],
        "gpu_efficiency": best_config["performance"]["communication_efficiency"],
        "workload_suitability": "Excellent"
        if best_config["performance"]["is_feasible"]
        else "Limited",
    }

    # Generate basic deployment command (placeholder for now)
    deployment_command = _generate_basic_deployment_command(
        model, best_scored_config.configuration, workload
    )

    # Get alternative configurations
    alternatives = [
        _convert_scored_config_to_dict(config)
        for config in result.configurations[1:6]  # Top 5 alternatives
    ]

    return OptimizedConfig(
        model_name=model,
        workload_type=workload,
        best_configuration=best_config,
        performance_estimate=performance_estimate,
        deployment_command=deployment_command,
        alternative_configurations=alternatives,
    )


def _generate_basic_deployment_command(model: str, config: Any, workload: str) -> str:
    """Generate a basic deployment command string."""
    # This is a simplified placeholder - full deployment command generation
    # will be implemented in Phase 5 (vLLM Integration)

    tp_size = config.tensor_parallel_size
    pp_size = config.pipeline_parallel_size
    total_gpus = config.total_gpus_needed

    if workload.lower() in ("inference", "chatbot", "interactive"):
        # vLLM-style command
        return (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model} "
            f"--tensor-parallel-size {tp_size} "
            f"--pipeline-parallel-size {pp_size} "
            f"--host 0.0.0.0 --port 8000"
        )
    else:
        # Training-style command
        return (
            f"torchrun --nproc_per_node={total_gpus} train.py "
            f"--model {model} "
            f"--tensor_parallel_size {tp_size} "
            f"--pipeline_parallel_size {pp_size}"
        )
