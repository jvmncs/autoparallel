"""Configuration generation for model parallelization strategies."""

from dataclasses import dataclass
from typing import Any

from autoparallel.config.optimizer import (
    HardwareProfile,
    OptimizationObjective,
    ParallelismConfiguration,
    PerformanceMetrics,
    WorkloadProfile,
    WorkloadType,
)
from autoparallel.constraints.analyzer import (
    ModelConstraints,
    analyze_model_constraints,
)
from autoparallel.constraints.validation import ValidationResult
from autoparallel.memory.components import MemoryComponents
from autoparallel.memory.estimator import (
    MoEMemoryEstimator,
    TransformersMemoryEstimator,
)


@dataclass
class ScoredConfiguration:
    """A parallelism configuration with associated performance metrics."""

    configuration: ParallelismConfiguration
    """The parallelism configuration"""

    performance_metrics: PerformanceMetrics
    """Performance metrics for this configuration"""

    memory_components: MemoryComponents
    """Detailed memory breakdown"""

    validation_result: ValidationResult | None = None
    """Configuration validation result"""

    @property
    def is_valid(self) -> bool:
        """Whether this configuration is valid."""
        return self.performance_metrics.is_feasible and (
            self.validation_result is None or self.validation_result.is_valid
        )


@dataclass
class ConfigurationGenerationResult:
    """Result of configuration generation with ranked configurations."""

    configurations: list[ScoredConfiguration]
    """List of valid configurations sorted by performance score"""

    best_configuration: ScoredConfiguration | None
    """Highest scoring configuration (None if no valid configs)"""

    generation_metadata: dict[str, Any]
    """Metadata about the generation process"""

    def __post_init__(self) -> None:
        """Validate generation result."""
        if self.configurations and not self.best_configuration:
            self.best_configuration = self.configurations[0]


class ConfigurationGenerator:
    """Generates and ranks parallelization configurations."""

    def __init__(
        self,
        memory_estimator: TransformersMemoryEstimator
        | MoEMemoryEstimator
        | None = None,
    ):
        """Initialize configuration generator.

        Args:
            memory_estimator: Memory estimator instance (defaults to
                TransformersMemoryEstimator)
        """
        self.memory_estimator = memory_estimator or TransformersMemoryEstimator()

    def generate_valid_configs(
        self,
        model_config: dict[str, Any],
        hardware_profile: HardwareProfile,
        workload_profile: WorkloadProfile,
        model_constraints: ModelConstraints | None = None,
        max_configurations: int = 100,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.MAXIMIZE_THROUGHPUT
        ),
    ) -> ConfigurationGenerationResult:
        """Generate and rank valid parallelization configurations.

        Args:
            model_config: Model configuration dict from transformers
            hardware_profile: Hardware specifications
            workload_profile: Workload characteristics
            model_constraints: Model constraints (auto-detected if None)
            max_configurations: Maximum configurations to return
            optimization_objective: Primary optimization objective

        Returns:
            ConfigurationGenerationResult with ranked configurations
        """
        # Auto-detect model constraints if not provided
        if model_constraints is None:
            try:
                # Try to create a PretrainedConfig from dict
                from transformers import PretrainedConfig

                config = PretrainedConfig.from_dict(model_config)
                model_constraints = analyze_model_constraints(config)
            except Exception:
                # Fallback to basic constraints based on model config
                model_constraints = self._create_fallback_constraints(model_config)

        # Generate all valid parallelism combinations
        base_configs = self._enumerate_parallelism_configs(
            model_constraints, hardware_profile
        )

        # Score and validate configurations
        scored_configs = []
        for config in base_configs:
            scored_config = self._score_configuration(
                config, model_config, hardware_profile, workload_profile
            )
            if scored_config and scored_config.is_valid:
                scored_configs.append(scored_config)

        # Rank configurations based on optimization objective
        ranked_configs = self._rank_configurations(
            scored_configs, workload_profile, optimization_objective
        )

        # Limit to max_configurations
        if max_configurations > 0:
            ranked_configs = ranked_configs[:max_configurations]

        # Generate metadata
        metadata = {
            "total_enumerated": len(base_configs),
            "valid_configs": len(scored_configs),
            "final_configs": len(ranked_configs),
            "optimization_objective": optimization_objective.value,
            "hardware_profile": {
                "total_gpus": hardware_profile.total_gpus,
                "memory_per_gpu_gb": hardware_profile.gpu_memory_gb,
                "gpu_model": hardware_profile.gpu_model,
            },
            "workload_type": workload_profile.workload_type.value,
        }

        return ConfigurationGenerationResult(
            configurations=ranked_configs,
            best_configuration=ranked_configs[0] if ranked_configs else None,
            generation_metadata=metadata,
        )

    def _enumerate_parallelism_configs(
        self,
        model_constraints: ModelConstraints,
        hardware_profile: HardwareProfile,
    ) -> list[ParallelismConfiguration]:
        """Enumerate all valid parallelism configurations."""
        valid_configs = []

        # Get valid parallelism sizes based on constraints and hardware
        tensor_parallel_sizes = model_constraints.get_valid_tensor_parallel_sizes(
            hardware_profile.total_gpus
        )
        expert_parallel_sizes = model_constraints.get_valid_expert_parallel_sizes(
            hardware_profile.total_gpus
        )
        pipeline_parallel_sizes = model_constraints.get_valid_pipeline_parallel_sizes(
            hardware_profile.num_nodes
        )

        # Ensure we have at least basic options
        if not tensor_parallel_sizes:
            tensor_parallel_sizes = [1]
        if not expert_parallel_sizes:
            expert_parallel_sizes = [1]
        if not pipeline_parallel_sizes:
            pipeline_parallel_sizes = [1]

        # Enumerate all combinations
        for tp in tensor_parallel_sizes:
            for ep in expert_parallel_sizes:
                for pp in pipeline_parallel_sizes:
                    # Calculate required data parallelism
                    base_parallelism = tp * ep * pp
                    if base_parallelism > hardware_profile.total_gpus:
                        continue

                    dp = hardware_profile.total_gpus // base_parallelism

                    # Create configuration
                    config = ParallelismConfiguration(
                        tensor_parallel_size=tp,
                        pipeline_parallel_size=pp,
                        expert_parallel_size=ep,
                        data_parallel_size=dp,
                    )

                    valid_configs.append(config)

        return valid_configs

    def _score_configuration(
        self,
        config: ParallelismConfiguration,
        model_config: dict[str, Any],
        hardware_profile: HardwareProfile,
        workload_profile: WorkloadProfile,
    ) -> ScoredConfiguration | None:
        """Score a configuration and validate its feasibility."""
        try:
            # Estimate memory usage for this configuration
            memory_components = self.memory_estimator.estimate_memory(
                model_config=model_config,
                sequence_length=workload_profile.sequence_length,
                batch_size=workload_profile.batch_size,
                tensor_parallel_size=config.tensor_parallel_size,
                pipeline_parallel_size=config.pipeline_parallel_size,
                expert_parallel_size=config.expert_parallel_size,
                is_training=workload_profile.is_training,
            )

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                config, memory_components, hardware_profile, workload_profile
            )

            # Validate configuration
            validation_result = self._validate_configuration(
                config, memory_components, hardware_profile
            )

            return ScoredConfiguration(
                configuration=config,
                performance_metrics=performance_metrics,
                memory_components=memory_components,
                validation_result=validation_result,
            )

        except Exception:
            # Skip configurations that cause estimation errors
            return None

    def _calculate_performance_metrics(
        self,
        config: ParallelismConfiguration,
        memory_components: MemoryComponents,
        hardware_profile: HardwareProfile,
        workload_profile: WorkloadProfile,
    ) -> PerformanceMetrics:
        """Calculate performance metrics for a configuration."""
        # Memory utilization
        memory_per_gpu_gb = memory_components.total_memory / (1024**3)
        is_feasible = memory_per_gpu_gb <= hardware_profile.gpu_memory_gb

        # Throughput estimation (simplified model)
        base_throughput = 1000.0  # Base tokens/sec
        tp_scaling = config.tensor_parallel_size**0.8  # Sublinear scaling
        dp_scaling = config.data_parallel_size  # Linear scaling for batch processing
        pp_overhead = max(0.5, 1.0 - (config.pipeline_parallel_size - 1) * 0.05)

        estimated_throughput = base_throughput * tp_scaling * dp_scaling * pp_overhead
        throughput_score = min(estimated_throughput / 2000.0, 1.0)  # Normalize to [0,1]

        # Latency estimation (simplified model)
        base_latency = 50.0  # Base latency in ms
        tp_latency_factor = 1.0 + (config.tensor_parallel_size - 1) * 0.1
        pp_latency_factor = 1.0 + (config.pipeline_parallel_size - 1) * 0.2

        estimated_latency = base_latency * tp_latency_factor * pp_latency_factor
        latency_score = max(0.0, 1.0 - (estimated_latency - 50.0) / 200.0)

        # Memory efficiency
        memory_utilization = memory_per_gpu_gb / hardware_profile.gpu_memory_gb
        memory_efficiency_score = min(memory_utilization / 0.8, 1.0)  # Target 80% util

        # Cost estimation (based on GPU usage efficiency)
        total_gpus_used = config.total_gpus_needed
        gpu_efficiency = min(total_gpus_used / hardware_profile.total_gpus, 1.0)
        cost_score = gpu_efficiency  # Higher efficiency = better cost

        # Communication efficiency
        communication_efficiency = hardware_profile.get_communication_efficiency(
            config.tensor_parallel_size, config.pipeline_parallel_size
        )

        # GPU utilization
        gpu_utilization = gpu_efficiency * memory_efficiency_score

        return PerformanceMetrics(
            throughput_score=throughput_score,
            latency_score=latency_score,
            memory_efficiency_score=memory_efficiency_score,
            cost_score=cost_score,
            communication_efficiency=communication_efficiency,
            gpu_utilization=gpu_utilization,
            memory_utilization_gb_per_gpu=memory_per_gpu_gb,
            is_feasible=is_feasible,
        )

    def _validate_configuration(
        self,
        config: ParallelismConfiguration,
        memory_components: MemoryComponents,
        hardware_profile: HardwareProfile,
    ) -> ValidationResult:
        """Validate a parallelism configuration."""
        errors = []
        warnings = []

        # Check GPU count
        total_gpus_required = config.total_gpus_needed

        if total_gpus_required > hardware_profile.total_gpus:
            errors.append(
                f"Configuration requires {total_gpus_required} GPUs but only "
                f"{hardware_profile.total_gpus} available"
            )

        # Check memory constraints
        memory_per_gpu_gb = memory_components.total_memory / (1024**3)

        if memory_per_gpu_gb > hardware_profile.gpu_memory_gb:
            errors.append(
                f"Memory requirement {memory_per_gpu_gb:.1f}GB exceeds "
                f"{hardware_profile.gpu_memory_gb:.1f}GB per GPU"
            )

        # Memory efficiency warnings
        memory_efficiency = memory_per_gpu_gb / hardware_profile.gpu_memory_gb
        if memory_efficiency < 0.3:
            warnings.append(
                f"Low memory utilization ({memory_efficiency:.1%}), "
                "consider reducing parallelism"
            )
        elif memory_efficiency > 0.9:
            warnings.append(
                f"High memory utilization ({memory_efficiency:.1%}), "
                "may cause OOM errors"
            )

        # GPU utilization warnings
        gpu_efficiency = total_gpus_required / hardware_profile.total_gpus
        if gpu_efficiency < 0.5:
            warnings.append(
                f"Low GPU utilization ({gpu_efficiency:.1%}), consider using fewer GPUs"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=[],
        )

    def _rank_configurations(
        self,
        scored_configs: list[ScoredConfiguration],
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective,
    ) -> list[ScoredConfiguration]:
        """Rank configurations based on optimization objective."""
        if optimization_objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            scored_configs.sort(
                key=lambda x: x.performance_metrics.throughput_score, reverse=True
            )
        elif optimization_objective == OptimizationObjective.MINIMIZE_LATENCY:
            scored_configs.sort(
                key=lambda x: x.performance_metrics.latency_score, reverse=True
            )
        elif optimization_objective == OptimizationObjective.MINIMIZE_COST:
            scored_configs.sort(
                key=lambda x: x.performance_metrics.cost_score, reverse=True
            )
        elif optimization_objective == OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY:
            scored_configs.sort(
                key=lambda x: x.performance_metrics.memory_efficiency_score,
                reverse=True,
            )
        else:  # BALANCE_EFFICIENCY
            weights = workload_profile.get_priority_weights()
            scored_configs.sort(
                key=lambda x: x.performance_metrics.get_weighted_score(weights),
                reverse=True,
            )

        return scored_configs

    def _create_fallback_constraints(
        self, model_config: dict[str, Any]
    ) -> ModelConstraints:
        """Create fallback constraints when model analysis fails."""
        # Extract basic parameters with sensible defaults
        num_attention_heads = model_config.get("num_attention_heads", 32)
        num_layers = model_config.get("num_hidden_layers", 32)

        # Conservative constraints
        max_tp = min(num_attention_heads, 64)  # Limited by attention heads
        tp_divisors = {i for i in range(1, max_tp + 1) if num_attention_heads % i == 0}

        max_pp = min(num_layers // 2, 16)  # At least 2 layers per stage

        # Check if MoE model
        num_experts = model_config.get("num_experts", 0)
        if num_experts > 0:
            max_ep = min(num_experts, 32)
            ep_divisors = {i for i in range(1, max_ep + 1) if num_experts % i == 0}
        else:
            max_ep = 1
            ep_divisors = {1}

        return ModelConstraints(
            max_tensor_parallel=max_tp,
            tensor_parallel_divisors=tp_divisors,
            max_expert_parallel=max_ep,
            expert_parallel_divisors=ep_divisors,
            max_pipeline_parallel=max_pp,
            min_layers_per_stage=2,
            requires_tied_embeddings=model_config.get("tie_word_embeddings", False),
            supports_grouped_query_attention=False,  # Conservative default
            vocabulary_sharding=4,  # Default divisibility requirement
        )


def generate_configurations_for_workload(
    model_config: dict[str, Any],
    hardware_profile: HardwareProfile,
    workload_type: WorkloadType,
    sequence_length: int = 2048,
    batch_size: int = 1,
    target_latency_ms: float | None = None,
    optimization_objective: OptimizationObjective = (
        OptimizationObjective.MAXIMIZE_THROUGHPUT
    ),
    max_configurations: int = 50,
) -> ConfigurationGenerationResult:
    """Generate configurations optimized for a specific workload type.

    This is a convenience function for common use cases.

    Args:
        model_config: Model configuration dict
        hardware_profile: Hardware specifications
        workload_type: Type of workload (inference, training, etc.)
        sequence_length: Input sequence length
        batch_size: Batch size
        target_latency_ms: Target latency constraint (optional)
        optimization_objective: Primary optimization objective
        max_configurations: Maximum configurations to return

    Returns:
        ConfigurationGenerationResult with workload-optimized configurations
    """
    # Create workload profile based on type
    workload_profile = WorkloadProfile(
        workload_type=workload_type,
        sequence_length=sequence_length,
        batch_size=batch_size if workload_type != WorkloadType.CHATBOT else 1,
        latency_budget_ms=target_latency_ms
        or _get_default_latency_budget(workload_type),
        is_training=(workload_type == WorkloadType.TRAINING),
    )

    # Use appropriate optimization objective based on workload
    if workload_type == WorkloadType.CHATBOT:
        objective = OptimizationObjective.MINIMIZE_LATENCY
    elif workload_type in [WorkloadType.TRAINING, WorkloadType.BATCH_PROCESSING]:
        objective = OptimizationObjective.MAXIMIZE_THROUGHPUT
    else:
        objective = optimization_objective

    # Generate configurations
    generator = ConfigurationGenerator()
    return generator.generate_valid_configs(
        model_config=model_config,
        hardware_profile=hardware_profile,
        workload_profile=workload_profile,
        max_configurations=max_configurations,
        optimization_objective=objective,
    )


def _get_default_latency_budget(workload_type: WorkloadType) -> float:
    """Get default latency budget for workload type."""
    if workload_type == WorkloadType.CHATBOT:
        return 100.0
    elif workload_type == WorkloadType.INTERACTIVE:
        return 200.0
    elif workload_type == WorkloadType.TRAINING:
        return 1000.0
    elif workload_type == WorkloadType.BATCH_PROCESSING:
        return 5000.0
    else:
        return 500.0  # Default inference
