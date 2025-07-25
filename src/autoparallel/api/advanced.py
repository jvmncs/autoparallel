"""Advanced public API for autoparallel library with detailed configuration options."""

from dataclasses import dataclass
from typing import Any

try:
    import transformers
    from transformers import AutoConfig
except ImportError:
    transformers = None  # type: ignore
    AutoConfig = None  # type: ignore

from autoparallel.config.generator import (
    ConfigurationGenerationResult,
    ConfigurationGenerator,
    ScoredConfiguration,
)
from autoparallel.config.optimizer import (
    HardwareProfile,
    OptimizationObjective,
    WorkloadProfile,
    WorkloadType,
)
from autoparallel.constraints.analyzer import (
    ModelConstraints,
    ParallelismConstraintParameters,
    analyze_model_constraints,
)
from autoparallel.memory.config import MemoryConfig
from autoparallel.memory.estimator import (
    MoEMemoryEstimator,
    TransformersMemoryEstimator,
)


@dataclass
class Cluster:
    """Detailed cluster specification for advanced configuration."""

    gpu_model: str = "H100"
    """GPU model name (H100, A100, etc.)"""

    gpu_memory_gb: float = 80.0
    """GPU memory per device in GB"""

    gpus_per_node: int = 8
    """Number of GPUs per node"""

    num_nodes: int = 1
    """Number of nodes in cluster"""

    intra_node_bandwidth_gbps: float = 900.0
    """Bandwidth within node (NVSwitch) in GB/s"""

    inter_node_bandwidth_gbps: float = 200.0
    """Bandwidth between nodes (RDMA) in GB/s"""

    network_topology: str = "fat_tree"
    """Network topology (fat_tree, mesh, torus)"""

    @classmethod
    def from_dict(cls, cluster_dict: dict[str, Any]) -> "Cluster":
        """Create Cluster from dictionary specification."""
        return cls(
            gpu_model=cluster_dict.get("gpu_model", "H100"),
            gpu_memory_gb=cluster_dict.get("gpu_memory_gb", 80.0),
            gpus_per_node=cluster_dict.get("gpus_per_node", 8),
            num_nodes=cluster_dict.get("num_nodes", 1),
            intra_node_bandwidth_gbps=cluster_dict.get(
                "intra_node_bandwidth_gbps", 900.0
            ),
            inter_node_bandwidth_gbps=cluster_dict.get(
                "inter_node_bandwidth_gbps", 200.0
            ),
            network_topology=cluster_dict.get("network_topology", "fat_tree"),
        )

    @classmethod
    def from_slurm(cls) -> "Cluster":
        """Auto-detect cluster configuration from SLURM environment."""
        # Placeholder for SLURM auto-detection (Phase 8)
        # For now, return reasonable defaults
        import os

        # Try to detect from SLURM environment variables
        gpus_per_node = int(os.environ.get("SLURM_GPUS_PER_NODE", "8"))
        num_nodes = int(os.environ.get("SLURM_NNODES", "1"))

        return cls(
            gpu_model="H100",  # Default assumption
            gpu_memory_gb=80.0,  # H100 default
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
        )

    def to_hardware_profile(self) -> HardwareProfile:
        """Convert to internal HardwareProfile."""
        return HardwareProfile(
            gpu_model=self.gpu_model,
            gpu_memory_gb=self.gpu_memory_gb,
            gpus_per_node=self.gpus_per_node,
            num_nodes=self.num_nodes,
            intra_node_bandwidth_gbps=self.intra_node_bandwidth_gbps,
            inter_node_bandwidth_gbps=self.inter_node_bandwidth_gbps,
            network_topology=self.network_topology,
        )


@dataclass
class Workload:
    """Detailed workload specification for optimization."""

    workload_type: str = "inference"
    """Workload type (inference, training, chatbot, batch_processing, interactive)"""

    batch_size: int = 32
    """Expected batch size"""

    sequence_length: int = 2048
    """Expected sequence length"""

    requests_per_second: float = 100.0
    """Expected request rate"""

    latency_budget_ms: float = 100.0
    """P99 latency budget in milliseconds"""

    throughput_target: float = 1000.0
    """Target throughput (tokens/second)"""

    cost_budget_per_hour: float = 100.0
    """Cost budget per hour in dollars"""

    is_training: bool = False
    """Whether this is a training workload"""

    def to_workload_profile(self) -> WorkloadProfile:
        """Convert to internal WorkloadProfile."""
        workload_type_map = {
            "inference": WorkloadType.INFERENCE,
            "training": WorkloadType.TRAINING,
            "batch_processing": WorkloadType.BATCH_PROCESSING,
            "interactive": WorkloadType.INTERACTIVE,
            "chatbot": WorkloadType.CHATBOT,
        }

        workload_type = workload_type_map.get(
            self.workload_type.lower(), WorkloadType.INFERENCE
        )

        return WorkloadProfile(
            workload_type=workload_type,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            requests_per_second=self.requests_per_second,
            latency_budget_ms=self.latency_budget_ms,
            throughput_target=self.throughput_target,
            cost_budget_per_hour=self.cost_budget_per_hour,
            is_training=self.is_training,
        )


@dataclass
class Preferences:
    """User preferences for optimization."""

    memory_conservatism: str = "moderate"
    """Memory usage conservatism (conservative, moderate, aggressive)"""

    precision: str = "auto"
    """Precision preference (auto, fp16, bf16, fp8, int8, int4)"""

    framework: str = "auto"
    """Target framework (auto, vllm, deepspeed, transformers)"""

    optimization_objective: str = "balance_efficiency"
    """Optimization target (maximize_throughput, minimize_latency, minimize_cost, balance_efficiency)"""

    max_configurations: int = 100
    """Maximum number of configurations to generate"""

    min_memory_utilization: float = 0.5
    """Minimum acceptable memory utilization"""

    max_memory_utilization: float = 0.9
    """Maximum acceptable memory utilization"""

    def to_optimization_objective(self) -> OptimizationObjective:
        """Convert to internal OptimizationObjective."""
        objective_map = {
            "maximize_throughput": OptimizationObjective.MAXIMIZE_THROUGHPUT,
            "minimize_latency": OptimizationObjective.MINIMIZE_LATENCY,
            "minimize_cost": OptimizationObjective.MINIMIZE_COST,
            "balance_efficiency": OptimizationObjective.BALANCE_EFFICIENCY,
            "maximize_memory_efficiency": OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY,
        }

        return objective_map.get(
            self.optimization_objective.lower(),
            OptimizationObjective.BALANCE_EFFICIENCY,
        )

    def to_memory_config(self) -> MemoryConfig:
        """Convert to internal MemoryConfig."""
        utilization_map = {
            "conservative": 0.75,
            "moderate": 0.85,
            "aggressive": 0.95,
        }

        utilization_bound = utilization_map.get(self.memory_conservatism.lower(), 0.85)

        return MemoryConfig(
            utilization_bound=utilization_bound,
            fragmentation_overhead=0.10,
            safety_margin=0.05,
            quantization_format="fp16",  # Default, can be overridden
        )


@dataclass
class AnalysisInsights:
    """Detailed insights from model analysis."""

    model_architecture: str
    """Model architecture type (dense, moe, multimodal)"""

    parameter_count: int | str
    """Total parameter count"""

    memory_requirements: dict[str, Any]
    """Detailed memory requirements breakdown"""

    parallelism_constraints: dict[str, Any]
    """Architecture-specific parallelism constraints"""

    performance_bottlenecks: list[str]
    """Identified performance bottlenecks"""

    optimization_recommendations: list[str]
    """Specific optimization recommendations"""


@dataclass
class DetailedConfiguration:
    """Detailed configuration with comprehensive information."""

    tensor_parallel_size: int
    """Tensor parallelism size"""

    pipeline_parallel_size: int
    """Pipeline parallelism size"""

    expert_parallel_size: int
    """Expert parallelism size (for MoE models)"""

    data_parallel_size: int
    """Data parallelism size"""

    total_gpus: int
    """Total number of GPUs required"""

    performance_metrics: dict[str, Any]
    """Expected performance metrics"""

    memory_breakdown: dict[str, Any]
    """Detailed memory usage breakdown"""

    deployment_commands: dict[str, str]
    """Framework-specific deployment commands"""

    configuration_rationale: str
    """Explanation of why this configuration was chosen"""

    is_recommended: bool
    """Whether this is a recommended configuration"""

    @classmethod
    def from_scored_configuration(
        cls, scored_config: ScoredConfiguration, is_recommended: bool = False
    ) -> "DetailedConfiguration":
        """Create from internal ScoredConfiguration."""
        config = scored_config.configuration
        metrics = scored_config.performance_metrics
        memory = scored_config.memory_components

        return cls(
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            expert_parallel_size=config.expert_parallel_size,
            data_parallel_size=config.data_parallel_size,
            total_gpus=config.total_gpus_needed,
            performance_metrics={
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
            memory_breakdown={
                "total_memory_gb": memory.total_memory
                / (1024**3),  # Convert bytes to GB
                "weights_gb": memory.weights / (1024**3),
                "activations_gb": memory.activations / (1024**3),
                "kv_cache_gb": memory.kv_cache / (1024**3),
                "cuda_graphs_gb": memory.cuda_graphs / (1024**3),
                "optimizer_states_gb": memory.optimizer_states / (1024**3),
                "fragmentation_overhead_gb": memory.fragmentation_overhead / (1024**3),
            },
            deployment_commands={
                "vllm": f"# vLLM deployment command for TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}",
                "deepspeed": f"# DeepSpeed deployment command for TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}",
            },
            configuration_rationale=f"Optimized for {config.tensor_parallel_size}x tensor parallelism with {config.pipeline_parallel_size}x pipeline parallelism",
            is_recommended=is_recommended,
        )


@dataclass
class OptimizationResult:
    """Comprehensive optimization result with detailed analysis."""

    model_name: str
    """The model that was analyzed"""

    cluster: Cluster
    """Cluster specification used"""

    workload: Workload
    """Workload specification used"""

    preferences: Preferences
    """Optimization preferences used"""

    recommended_configuration: DetailedConfiguration | None
    """Best recommended configuration"""

    all_configurations: list[DetailedConfiguration]
    """All valid configurations ranked by performance"""

    insights: AnalysisInsights
    """Detailed analysis insights"""

    generation_time_seconds: float
    """Time taken to generate configurations"""

    def deploy_vllm(self) -> str:
        """Generate optimized vLLM deployment command for recommended configuration."""
        if not self.recommended_configuration:
            raise ValueError("No recommended configuration available")

        from autoparallel.deployment.vllm_commands import generate_vllm_deployment_command
        from autoparallel.frameworks.vllm_optimizer import GPUArchitecture

        try:
            # Generate optimized deployment command
            result = generate_vllm_deployment_command(
                model_name=self.model_name,
                hardware_profile=self.cluster,
                workload=self.workload,
                parallelism_config=self.recommended_configuration,
                gpu_architecture=GPUArchitecture.A100,  # Could be made configurable
            )
            
            return result.command
            
        except Exception as e:
            # Fallback to deployment commands if available
            fallback = self.recommended_configuration.deployment_commands.get("vllm")
            if fallback:
                return fallback
            
            # Generate basic fallback command
            config = self.recommended_configuration
            return (
                f"python -m vllm.entrypoints.openai.api_server "
                f"--model {self.model_name} "
                f"--tensor-parallel-size {config.tensor_parallel_size} "
                f"--pipeline-parallel-size {config.pipeline_parallel_size} "
                f"--host 0.0.0.0 --port 8000"
            )

    def deploy_deepspeed(self) -> dict[str, Any]:
        """Generate DeepSpeed configuration for recommended configuration."""
        if not self.recommended_configuration:
            raise ValueError("No recommended configuration available")

        # Placeholder for DeepSpeed config generation (Phase 7)
        config = self.recommended_configuration
        return {
            "train_batch_size": self.workload.batch_size,
            "train_micro_batch_size_per_gpu": self.workload.batch_size
            // config.data_parallel_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "AdamW"},
            "scheduler": {"type": "WarmupLR"},
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 3 if config.total_gpus > 8 else 2},
        }

    def explain(self) -> str:
        """Generate human-readable explanation of the optimization results."""
        if not self.recommended_configuration:
            return "No valid configurations found for the given constraints."

        config = self.recommended_configuration
        explanation = [
            f"Analysis of {self.model_name} on {self.cluster.num_nodes} nodes with {self.cluster.gpus_per_node} GPUs each:",
            "",
            "Recommended Configuration:",
            f"  - Tensor Parallelism: {config.tensor_parallel_size}x",
            f"  - Pipeline Parallelism: {config.pipeline_parallel_size}x",
            f"  - Expert Parallelism: {config.expert_parallel_size}x",
            f"  - Data Parallelism: {config.data_parallel_size}x",
            f"  - Total GPUs: {config.total_gpus}",
            "",
            "Expected Performance:",
            f"  - Throughput: {config.performance_metrics['throughput_tokens_per_second']:.1f} tokens/sec",
            f"  - Latency: {config.performance_metrics['latency_ms']:.1f} ms",
            f"  - Memory Utilization: {config.performance_metrics['memory_utilization']:.1%}",
            f"  - Cost: ${config.performance_metrics['cost_per_hour']:.2f}/hour",
            "",
            "Memory Breakdown:",
            f"  - Model Weights: {config.memory_breakdown['weights_gb']:.1f} GB",
            f"  - Activations: {config.memory_breakdown['activations_gb']:.1f} GB",
            f"  - KV Cache: {config.memory_breakdown['kv_cache_gb']:.1f} GB",
            f"  - Total: {config.memory_breakdown['total_memory_gb']:.1f} GB",
            "",
            "Key Insights:",
        ]

        for insight in self.insights.optimization_recommendations:
            explanation.append(f"  - {insight}")

        return "\n".join(explanation)


class AutoParallel:
    """Advanced autoparallel interface with detailed configuration options."""

    def __init__(
        self,
        cluster: Cluster | None = None,
        preferences: Preferences | None = None,
        constraint_params: ParallelismConstraintParameters | None = None,
    ):
        """Initialize AutoParallel with advanced configuration.

        Args:
            cluster: Detailed cluster specification
            preferences: User optimization preferences
            constraint_params: Custom parallelism constraint parameters
        """
        self.cluster = cluster or Cluster()
        self.preferences = preferences or Preferences()
        self.constraint_params = constraint_params or ParallelismConstraintParameters()

        # Initialize internal components
        self.memory_config = self.preferences.to_memory_config()

    def analyze_model(
        self,
        model: str,
        workload: Workload | None = None,
    ) -> OptimizationResult:
        """Perform comprehensive model analysis with detailed insights.

        Args:
            model: Model identifier (HuggingFace model name or local path)
            workload: Workload specification (defaults to inference)

        Returns:
            OptimizationResult with comprehensive analysis

        Example:
            >>> autoparallel = AutoParallel(
            ...     cluster=Cluster(gpu_memory_gb=80, gpus_per_node=8, num_nodes=4),
            ...     preferences=Preferences(optimization_objective="minimize_latency")
            ... )
            >>> result = autoparallel.analyze_model(
            ...     model="meta-llama/Llama-2-70b-hf",
            ...     workload=Workload(workload_type="chatbot", batch_size=16)
            ... )
            >>> print(result.explain())
        """
        import time

        start_time = time.time()

        # Use default workload if not provided
        workload = workload or Workload()

        # Load model configuration
        model_config = self._load_model_config(model)

        # Convert to internal types
        hardware_profile = self.cluster.to_hardware_profile()
        workload_profile = workload.to_workload_profile()

        # Analyze model constraints
        model_constraints = analyze_model_constraints(
            model_config, self.constraint_params
        )

        # Choose appropriate memory estimator
        is_moe = any(
            hasattr(model_config, attr)
            for attr in ["num_local_experts", "moe_intermediate_size"]
        )
        memory_estimator = (
            MoEMemoryEstimator() if is_moe else TransformersMemoryEstimator()
        )

        # Generate configurations
        generator = ConfigurationGenerator(memory_estimator=memory_estimator)
        result = generator.generate_valid_configs(
            model_config=model_config.to_dict(),
            hardware_profile=hardware_profile,
            workload_profile=workload_profile,
            model_constraints=model_constraints,
            max_configurations=self.preferences.max_configurations,
            optimization_objective=self.preferences.to_optimization_objective(),
        )

        # Convert to public API format
        configurations = [
            DetailedConfiguration.from_scored_configuration(
                config, is_recommended=(i == 0)
            )
            for i, config in enumerate(result.configurations)
        ]

        # Generate insights
        insights = self._generate_insights(model_config, model_constraints, result)

        end_time = time.time()

        return OptimizationResult(
            model_name=model,
            cluster=self.cluster,
            workload=workload,
            preferences=self.preferences,
            recommended_configuration=configurations[0] if configurations else None,
            all_configurations=configurations,
            insights=insights,
            generation_time_seconds=end_time - start_time,
        )

    def optimize(
        self,
        model: str,
        workload: Workload,
        custom_constraints: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Optimize configuration with custom constraints.

        Args:
            model: Model identifier
            workload: Detailed workload specification
            custom_constraints: Custom parallelism constraints

        Returns:
            OptimizationResult with optimized configuration
        """
        # For now, delegate to analyze_model
        # Custom constraints support will be added in future phases
        return self.analyze_model(model, workload)

    def _load_model_config(self, model: str) -> Any:
        """Load model configuration from HuggingFace or local path."""
        if transformers is None or AutoConfig is None:
            raise ImportError(
                "transformers library is required for model configuration loading"
            )

        try:
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            return config  # Return the config object directly, not as dict
        except Exception as e:
            raise ValueError(
                f"Failed to load model configuration for '{model}': {e}"
            ) from e

    def _generate_insights(
        self,
        model_config: Any,
        model_constraints: ModelConstraints,
        result: ConfigurationGenerationResult,
    ) -> AnalysisInsights:
        """Generate detailed analysis insights."""
        is_moe = any(
            hasattr(model_config, attr)
            for attr in ["num_local_experts", "moe_intermediate_size"]
        )

        architecture = "MoE" if is_moe else "Dense Transformer"
        if any(
            hasattr(model_config, attr) for attr in ["vision_config", "text_config"]
        ):
            architecture = "Multimodal " + architecture

        model_dict = model_config.to_dict()
        parameter_count = model_dict.get("n_parameters", "Unknown")
        if isinstance(parameter_count, (int, float)):
            if parameter_count > 1e12:
                parameter_count = f"{parameter_count / 1e12:.1f}T"
            elif parameter_count > 1e9:
                parameter_count = f"{parameter_count / 1e9:.1f}B"
            elif parameter_count > 1e6:
                parameter_count = f"{parameter_count / 1e6:.1f}M"

        # Identify bottlenecks
        bottlenecks = []
        recommendations = []

        if model_constraints.max_tensor_parallel <= 8:
            bottlenecks.append(
                "Limited tensor parallelism due to attention head constraints"
            )
            recommendations.append("Consider using pipeline parallelism for scaling")

        if is_moe and model_constraints.max_expert_parallel > 16:
            recommendations.append("Use expert parallelism for efficient MoE scaling")

        if result.configurations:
            best_config = result.configurations[0]
            if (
                best_config.performance_metrics.memory_utilization_gb_per_gpu > 70
            ):  # Assuming high if > 70GB
                bottlenecks.append("High memory utilization may cause OOM issues")
                recommendations.append(
                    "Consider gradient checkpointing or ZeRO optimization"
                )

        return AnalysisInsights(
            model_architecture=architecture,
            parameter_count=parameter_count,
            memory_requirements={
                "minimum_memory_gb": min(
                    config.memory_components.total_memory / (1024**3)
                    for config in result.configurations
                )
                if result.configurations
                else 0,
                "recommended_memory_gb": result.configurations[
                    0
                ].memory_components.total_memory
                / (1024**3)
                if result.configurations
                else 0,
            },
            parallelism_constraints={
                "max_tensor_parallel": model_constraints.max_tensor_parallel,
                "max_pipeline_parallel": model_constraints.max_pipeline_parallel,
                "max_expert_parallel": model_constraints.max_expert_parallel,
                "supports_gqa": model_constraints.supports_grouped_query_attention,
            },
            performance_bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
        )
