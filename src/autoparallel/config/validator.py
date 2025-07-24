"""Configuration validation for autoparallel parallelism configurations.

This module provides comprehensive validation that checks parallelism configurations
against architectural constraints, memory requirements, cross-component interactions,
and provides detailed error messages and recommendations for optimization.
"""

from dataclasses import dataclass
from typing import Any, cast

try:
    from transformers import PretrainedConfig
except ImportError:
    PretrainedConfig = object  # type: ignore

from autoparallel.constraints.analyzer import (
    analyze_model_constraints,
)
from autoparallel.constraints.validation import (
    ClusterSpec,
    ConstraintValidator,
    ParallelismConfig,
    ValidationError,
    ValidationResult,
)
from autoparallel.memory.config import MemoryConfig
from autoparallel.memory.estimator import MemoryEstimator, TransformersMemoryEstimator


class ConfigurationValidationError(ValidationError):
    """Exception for configuration validation errors."""

    pass


class ResourceValidationError(ValidationError):
    """Exception for resource validation errors."""

    pass


class CompatibilityValidationError(ValidationError):
    """Exception for framework compatibility validation errors."""

    pass


class PerformanceValidationError(ValidationError):
    """Exception for performance validation errors."""

    pass


@dataclass
class ConfigurationSpec:
    """Complete configuration specification for validation.

    Attributes:
        parallelism_config: Parallelism configuration
        cluster_spec: Cluster specification
        workload_spec: Workload specification
        framework_config: Framework-specific configuration
    """

    parallelism_config: ParallelismConfig
    cluster_spec: ClusterSpec
    workload_spec: dict[str, Any]
    framework_config: dict[str, Any] | None = None


@dataclass
class WorkloadSpec:
    """Workload specification for validation.

    Attributes:
        sequence_length: Input sequence length
        batch_size: Batch size per device
        max_sequence_length: Maximum sequence length
        is_training: Whether this is for training
        workload_type: Type of workload (inference, training, etc.)
        throughput_target: Target throughput (tokens/second)
        latency_target: Target latency (milliseconds)
    """

    sequence_length: int = 2048
    batch_size: int = 1
    max_sequence_length: int = 4096
    is_training: bool = False
    workload_type: str = "inference"
    throughput_target: float | None = None
    latency_target: float | None = None


class ConfigurationValidator:
    """Comprehensive configuration validator for autoparallel.

    This validator provides comprehensive validation that:
    1. Validates parallelism configurations against architectural constraints
    2. Validates memory requirements against available resources
    3. Validates cross-component interactions (e.g., TP*PP*EP*DP = total_gpus)
    4. Provides detailed error messages and recommendations
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        memory_config: MemoryConfig | None = None,
        memory_estimator: MemoryEstimator | None = None,
        enable_performance_validation: bool = True,
        enable_framework_validation: bool = True,
    ):
        """Initialize configuration validator.

        Args:
            model_config: Transformer model configuration dictionary
            memory_config: Memory configuration for validation
            memory_estimator: Memory estimation component
            enable_performance_validation: Whether to perform performance validation
            enable_framework_validation: Whether to perform framework validation
        """
        self.model_config = model_config
        self.memory_config = memory_config or MemoryConfig()
        self.memory_estimator = memory_estimator or TransformersMemoryEstimator(
            self.memory_config
        )
        self.enable_performance_validation = enable_performance_validation
        self.enable_framework_validation = enable_framework_validation

        # Initialize constraint validator
        self.constraint_validator = ConstraintValidator(
            model_config, self.memory_config, self.memory_estimator
        )

        # Analyze model constraints - handle dict/PretrainedConfig conversion
        if isinstance(model_config, dict):
            # Create a simple object that acts like PretrainedConfig for dict access
            config_obj = type("ModelConfig", (), model_config)()
            for key, value in model_config.items():
                setattr(config_obj, key, value)
        else:
            config_obj = model_config

        self.model_constraints = analyze_model_constraints(
            cast(PretrainedConfig, config_obj)
        )

    def validate_configuration(
        self,
        configuration_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec | None = None,
    ) -> ValidationResult:
        """Validate a complete configuration specification.

        Args:
            configuration_spec: Complete configuration to validate
            workload_spec: Workload specification for validation

        Returns:
            ValidationResult with comprehensive validation feedback
        """
        workload_spec = workload_spec or WorkloadSpec()

        errors = []
        warnings = []
        recommendations = []

        # 1. Basic constraint validation using existing validator
        basic_validation = self.constraint_validator.validate_configuration(
            configuration_spec.parallelism_config,
            configuration_spec.cluster_spec,
            workload_spec.sequence_length,
            workload_spec.batch_size,
            workload_spec.is_training,
        )

        errors.extend(basic_validation.errors)
        warnings.extend(basic_validation.warnings)
        recommendations.extend(basic_validation.recommendations)

        # 2. Enhanced architectural constraint validation
        try:
            self._validate_enhanced_architectural_constraints(
                configuration_spec.parallelism_config,
                workload_spec,
                errors,
                warnings,
                recommendations,
            )
        except ConfigurationValidationError as e:
            errors.append(str(e))

        # 3. Resource availability validation
        try:
            self._validate_resource_availability(
                configuration_spec,
                workload_spec,
                errors,
                warnings,
                recommendations,
            )
        except ResourceValidationError as e:
            errors.append(str(e))

        # 4. Cross-component interaction validation
        try:
            self._validate_cross_component_interactions(
                configuration_spec.parallelism_config,
                configuration_spec.cluster_spec,
                workload_spec,
                errors,
                warnings,
                recommendations,
            )
        except ConfigurationValidationError as e:
            errors.append(str(e))

        # 5. Framework compatibility validation
        if self.enable_framework_validation and configuration_spec.framework_config:
            try:
                self._validate_framework_compatibility(
                    configuration_spec,
                    workload_spec,
                    errors,
                    warnings,
                    recommendations,
                )
            except CompatibilityValidationError as e:
                errors.append(str(e))

        # 6. Performance validation
        if self.enable_performance_validation:
            try:
                self._validate_performance_requirements(
                    configuration_spec,
                    workload_spec,
                    errors,
                    warnings,
                    recommendations,
                )
            except PerformanceValidationError as e:
                errors.append(str(e))

        # 7. Configuration completeness validation
        try:
            self._validate_configuration_completeness(
                configuration_spec,
                workload_spec,
                errors,
                warnings,
                recommendations,
            )
        except ConfigurationValidationError as e:
            errors.append(str(e))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_enhanced_architectural_constraints(
        self,
        config: ParallelismConfig,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate enhanced architectural constraints beyond basic validation."""
        # Check against model-specific constraints
        if config.tensor_parallel_size > self.model_constraints.max_tensor_parallel:
            raise ConfigurationValidationError(
                f"Tensor parallel size ({config.tensor_parallel_size}) exceeds "
                f"model maximum ({self.model_constraints.max_tensor_parallel})"
            )

        if config.pipeline_parallel_size > self.model_constraints.max_pipeline_parallel:
            raise ConfigurationValidationError(
                f"Pipeline parallel size ({config.pipeline_parallel_size}) exceeds "
                f"model maximum ({self.model_constraints.max_pipeline_parallel})"
            )

        if config.expert_parallel_size not in self.model_constraints.expert_parallel_divisors:
            raise ConfigurationValidationError(
                f"Expert parallel size ({config.expert_parallel_size}) not in "
                f"valid sizes {sorted(self.model_constraints.expert_parallel_divisors)}"
            )

        # Validate vocabulary sharding requirements
        if self.model_constraints.vocabulary_sharding:
            vocab_size = self.model_config.get("vocab_size", 32000)
            if vocab_size % config.tensor_parallel_size != 0:
                msg = (
                    f"Vocabulary size ({vocab_size}) not divisible by tensor "
                    f"parallel size ({config.tensor_parallel_size}). "
                    "This will cause padding overhead."
                )
                warnings.append(msg)

        # Validate sequence length constraints
        max_pos_emb = self.model_config.get("max_position_embeddings", 2048)
        if workload_spec.max_sequence_length > max_pos_emb:
            msg = (
                f"Maximum sequence length ({workload_spec.max_sequence_length}) "
                f"exceeds model position embeddings ({max_pos_emb})"
            )
            errors.append(msg)

        # Validate GQA constraints for grouped query attention
        num_attention_heads = self.model_config.get("num_attention_heads", 32)
        num_key_value_heads = self.model_config.get(
            "num_key_value_heads", num_attention_heads
        )

        if num_key_value_heads != num_attention_heads:  # GQA model
            if num_attention_heads % config.tensor_parallel_size != 0:
                raise ConfigurationValidationError(
                    f"For GQA models, attention heads ({num_attention_heads}) must be "
                    f"divisible by tensor parallel size ({config.tensor_parallel_size})"
                )

            kv_heads_per_device = num_key_value_heads // config.tensor_parallel_size
            if kv_heads_per_device < 1:
                tp_size = config.tensor_parallel_size
                msg = (
                    f"GQA model requires at least 1 KV head per device, but "
                    f"KV heads ({num_key_value_heads}) / TP size ({tp_size}) = "
                    f"{kv_heads_per_device:.2f}"
                )
                raise ConfigurationValidationError(msg)

    def _validate_resource_availability(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate resource availability beyond basic memory checks."""
        config = config_spec.parallelism_config
        cluster = config_spec.cluster_spec

        # Enhanced memory validation with different workload scenarios
        memory_components = self.memory_estimator.estimate_memory(
            model_config=self.model_config,
            sequence_length=workload_spec.max_sequence_length,  # Use max for safety
            batch_size=workload_spec.batch_size,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            expert_parallel_size=config.expert_parallel_size,
            is_training=workload_spec.is_training,
        )

        memory_per_gpu_gb = memory_components.total_memory / (1024**3)
        utilization = memory_per_gpu_gb / cluster.gpu_memory_gb

        # Check if memory exceeds physical limits
        if memory_per_gpu_gb > cluster.gpu_memory_gb:
            raise ResourceValidationError(
                f"Configuration requires {memory_per_gpu_gb:.2f} GB per GPU but "
                f"cluster GPUs only have {cluster.gpu_memory_gb:.2f} GB available"
            )

        # Enhanced utilization checks
        if utilization > 0.95:
            errors.append(
                f"Extremely high memory utilization ({utilization:.1%}). "
                "Consider increasing parallelism or reducing batch size."
            )
        elif utilization > self.memory_config.utilization_bound:
            warnings.append(
                f"High memory utilization ({utilization:.1%}) exceeds safety bound "
                f"({self.memory_config.utilization_bound:.1%})"
            )

        # Memory efficiency recommendations
        if utilization < 0.3:
            recommendations.append(
                f"Low memory utilization ({utilization:.1%}). Consider increasing "
                "batch size or reducing parallelism for better resource efficiency."
            )

        # Validate KV cache scaling for long sequences
        if workload_spec.max_sequence_length > 8192:
            kv_cache_gb = memory_components.kv_cache / (1024**3)
            kv_cache_ratio = kv_cache_gb / memory_per_gpu_gb

            if kv_cache_ratio > 0.5:
                warnings.append(
                    f"KV cache uses {kv_cache_ratio:.1%} of memory for long sequences. "
                    "Consider increasing tensor parallelism to reduce KV cache per GPU."
                )

        # Validate compute-to-memory ratio
        total_params = self._estimate_parameter_count()
        params_per_gpu = total_params / config.tensor_parallel_size

        if params_per_gpu < 1e9:  # Less than 1B parameters per GPU
            recommendations.append(
                f"Low parameter count per GPU ({params_per_gpu / 1e9:.1f}B). "
                "Consider reducing tensor parallelism for better compute utilization."
            )

    def _validate_cross_component_interactions(
        self,
        config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate complex cross-component interactions."""
        # Validate communication patterns
        self._validate_communication_patterns(
            config, cluster_spec, errors, warnings, recommendations
        )

        # Validate load balancing
        self._validate_load_balancing(
            config, workload_spec, errors, warnings, recommendations
        )

        # Validate scalability patterns
        self._validate_scalability_patterns(
            config, cluster_spec, errors, warnings, recommendations
        )

    def _validate_communication_patterns(
        self,
        config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate communication patterns and bandwidth requirements."""
        # Check for suboptimal communication patterns
        if config.tensor_parallel_size > cluster_spec.gpus_per_node:
            inter_node_tp_groups = (
                config.tensor_parallel_size + cluster_spec.gpus_per_node - 1
            ) // cluster_spec.gpus_per_node

            if inter_node_tp_groups > 1:
                msg = (
                    f"Tensor parallelism spans {inter_node_tp_groups} nodes. "
                    "This requires high inter-node bandwidth and may reduce "
                    "performance."
                )
                warnings.append(msg)

        # Check expert parallelism communication
        if config.expert_parallel_size > 1:
            # MoE models require all-to-all communication
            expert_comm_volume = self._estimate_expert_communication_volume(config)

            if expert_comm_volume > cluster_spec.inter_node_bandwidth_gbps * 0.8:
                msg = (
                    f"Expert parallelism communication ({expert_comm_volume:.1f} Gbps) "
                    f"may saturate inter-node bandwidth "
                    f"({cluster_spec.inter_node_bandwidth_gbps} Gbps)"
                )
                warnings.append(msg)

        # Recommend optimal communication topology
        optimal_tp = config.tensor_parallel_size <= cluster_spec.gpus_per_node
        optimal_pp = config.pipeline_parallel_size <= cluster_spec.num_nodes
        if optimal_tp and optimal_pp:
            recommendations.append(
                "Configuration optimally utilizes network topology with "
                "tensor parallelism within nodes and pipeline parallelism across nodes."
            )

    def _validate_load_balancing(
        self,
        config: ParallelismConfig,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate load balancing across parallelism dimensions."""
        num_layers = self.model_config.get("num_hidden_layers", 32)

        # Pipeline parallelism load balancing
        if config.pipeline_parallel_size > 1:
            layers_per_stage = num_layers / config.pipeline_parallel_size
            layer_imbalance = num_layers % config.pipeline_parallel_size

            if layer_imbalance > 0:
                imbalance_ratio = layer_imbalance / layers_per_stage
                if imbalance_ratio > 0.5:
                    warnings.append(
                        f"Pipeline stages have uneven load: {layer_imbalance} stages "
                        f"have extra layers, causing {imbalance_ratio:.1%} imbalance."
                    )

        # Expert parallelism load balancing for MoE
        num_experts = self.model_config.get("num_experts", 0)
        if (
            num_experts > 0
            and config.expert_parallel_size > 1
            and num_experts % config.expert_parallel_size != 0
        ):
            warnings.append(
                f"Experts ({num_experts}) not evenly distributed across "
                f"expert parallel devices ({config.expert_parallel_size}). "
                "Some devices will handle more experts than others."
            )

        # Data parallelism efficiency
        if workload_spec.batch_size < config.data_parallel_size:
            errors.append(
                f"Batch size ({workload_spec.batch_size}) is smaller than "
                f"data parallel size ({config.data_parallel_size}). "
                "Some data parallel workers will be idle."
            )

    def _validate_scalability_patterns(
        self,
        config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate scalability and efficiency patterns."""
        total_gpus = config.total_gpus

        # Check for over-parallelization
        if total_gpus > 64:
            efficiency_concerns = []

            if config.tensor_parallel_size > 8:
                concern = f"high tensor parallelism ({config.tensor_parallel_size})"
                efficiency_concerns.append(concern)

            if config.pipeline_parallel_size > 8:
                concern = f"high pipeline parallelism ({config.pipeline_parallel_size})"
                efficiency_concerns.append(concern)

            if efficiency_concerns:
                warnings.append(
                    f"Large-scale configuration with {', '.join(efficiency_concerns)} "
                    "may have diminishing returns due to communication overhead."
                )

        # Check for under-utilization
        gpu_utilization = total_gpus / cluster_spec.total_gpus
        if gpu_utilization < 0.8:
            recommendations.append(
                f"Configuration uses {gpu_utilization:.1%} of available GPUs. "
                "Consider increasing data parallelism to improve resource utilization."
            )

    def _validate_framework_compatibility(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate framework-specific compatibility requirements."""
        framework_config = config_spec.framework_config or {}
        framework_name = framework_config.get("framework", "unknown")

        # vLLM-specific validation
        if framework_name.lower() == "vllm":
            self._validate_vllm_compatibility(
                config_spec, workload_spec, errors, warnings, recommendations
            )

        # DeepSpeed-specific validation
        elif framework_name.lower() == "deepspeed":
            self._validate_deepspeed_compatibility(
                config_spec, workload_spec, errors, warnings, recommendations
            )

        # Framework-agnostic compatibility checks
        self._validate_general_framework_compatibility(
            config_spec, workload_spec, errors, warnings, recommendations
        )

    def _validate_vllm_compatibility(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate vLLM-specific requirements."""
        config = config_spec.parallelism_config
        framework_config = config_spec.framework_config or {}

        # vLLM has specific tensor parallelism limitations
        if config.tensor_parallel_size > 8:
            warnings.append(
                f"vLLM tensor parallelism ({config.tensor_parallel_size}) > 8 "
                "may have suboptimal performance. Consider using pipeline parallelism."
            )

        # Check CUDA graphs compatibility
        enable_cuda_graphs = framework_config.get("enable_cuda_graphs", True)
        if enable_cuda_graphs and workload_spec.sequence_length > 8192:
            warnings.append(
                "CUDA graphs with long sequences may increase memory usage. "
                "Consider disabling CUDA graphs for sequences > 8K tokens."
            )

        # Check quantization compatibility
        quantization = framework_config.get("quantization", None)
        if quantization and config.tensor_parallel_size > 4:
            msg = (
                f"Quantization with high tensor parallelism "
                f"({config.tensor_parallel_size}) "
                "may cause performance degradation in vLLM."
            )
            warnings.append(msg)

    def _validate_deepspeed_compatibility(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate DeepSpeed-specific requirements."""
        config = config_spec.parallelism_config
        framework_config = config_spec.framework_config or {}

        # DeepSpeed ZeRO stage validation
        zero_stage = framework_config.get("zero_stage", 0)
        if zero_stage > 0 and not workload_spec.is_training:
            warnings.append(
                f"ZeRO stage {zero_stage} is enabled for inference. "
                "ZeRO is primarily beneficial for training workloads."
            )

        # DeepSpeed 3D parallelism validation
        if config.pipeline_parallel_size > 1 and zero_stage >= 2:
            warnings.append(
                "Pipeline parallelism with ZeRO-2/3 may have complex interactions. "
                "Validate memory and communication patterns carefully."
            )

    def _validate_general_framework_compatibility(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate general framework compatibility requirements."""
        # Check for common framework limitations
        is_training = workload_spec.is_training
        ep_size = config_spec.parallelism_config.expert_parallel_size
        if is_training and ep_size > 1:
            msg = (
                "Training with expert parallelism requires careful gradient "
                "synchronization. Ensure framework supports MoE training properly."
            )
            warnings.append(msg)

    def _validate_performance_requirements(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate performance requirements and targets."""
        if not workload_spec.throughput_target and not workload_spec.latency_target:
            return  # No performance targets to validate

        # Estimate performance characteristics
        estimated_throughput = self._estimate_throughput(config_spec, workload_spec)
        estimated_latency = self._estimate_latency(config_spec, workload_spec)

        # Validate throughput targets
        if (
            workload_spec.throughput_target
            and estimated_throughput < workload_spec.throughput_target * 0.8
        ):
            msg = (
                f"Estimated throughput ({estimated_throughput:.0f} tokens/s) "
                f"may not meet target ({workload_spec.throughput_target:.0f} tokens/s)"
            )
            warnings.append(msg)

        # Validate latency targets
        if (
            workload_spec.latency_target
            and estimated_latency > workload_spec.latency_target * 1.2
        ):
            msg = (
                f"Estimated latency ({estimated_latency:.0f} ms) "
                f"may exceed target ({workload_spec.latency_target:.0f} ms)"
            )
            warnings.append(msg)

    def _validate_configuration_completeness(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate configuration completeness and consistency."""
        # Check for missing critical configuration
        if workload_spec.workload_type == "training" and not workload_spec.is_training:
            warnings.append("Workload type is 'training' but is_training=False")

        # Check for configuration consistency
        has_large_batch = workload_spec.batch_size > 32
        has_strict_latency = (
            workload_spec.latency_target and workload_spec.latency_target < 100
        )
        if has_large_batch and has_strict_latency:
            batch = workload_spec.batch_size
            latency = workload_spec.latency_target
            msg = (
                f"Large batch size ({batch}) with strict latency target "
                f"({latency} ms) may be incompatible"
            )
            warnings.append(msg)

        # Provide optimization recommendations
        if not recommendations:  # Only if no other recommendations provided
            self._generate_optimization_recommendations(
                config_spec, workload_spec, recommendations
            )

    def _generate_optimization_recommendations(
        self,
        config_spec: ConfigurationSpec,
        workload_spec: WorkloadSpec,
        recommendations: list[str],
    ) -> None:
        """Generate optimization recommendations for the configuration."""
        config = config_spec.parallelism_config

        # Memory optimization recommendations
        memory_components = self.memory_estimator.estimate_memory(
            model_config=self.model_config,
            sequence_length=workload_spec.sequence_length,
            batch_size=workload_spec.batch_size,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            expert_parallel_size=config.expert_parallel_size,
            is_training=workload_spec.is_training,
        )

        memory_per_gpu_gb = memory_components.total_memory / (1024**3)
        utilization = memory_per_gpu_gb / config_spec.cluster_spec.gpu_memory_gb

        if utilization < 0.6:
            msg = (
                "Consider increasing batch size or reducing parallelism "
                "to improve memory efficiency"
            )
            recommendations.append(msg)

        # Communication optimization recommendations
        if config.tensor_parallel_size > config_spec.cluster_spec.gpus_per_node:
            msg = (
                "Consider using pipeline parallelism instead of "
                "cross-node tensor parallelism"
            )
            recommendations.append(msg)

        # General optimization recommendations
        dp_size_is_one = config.data_parallel_size == 1
        has_unused_gpus = config_spec.cluster_spec.total_gpus > config.total_gpus
        if dp_size_is_one and has_unused_gpus:
            recommendations.append(
                "Consider increasing data parallelism to utilize additional GPUs"
            )

    def _estimate_parameter_count(self) -> int:
        """Estimate model parameter count."""
        vocab_size = self.model_config.get("vocab_size", 32000)
        hidden_size = self.model_config.get("hidden_size", 4096)
        num_layers = self.model_config.get("num_hidden_layers", 32)
        intermediate_size = self.model_config.get("intermediate_size", 11008)

        # Approximate parameter count calculation
        embedding_params = vocab_size * hidden_size
        layer_params = (
            4 * hidden_size * hidden_size + 2 * hidden_size * intermediate_size
        )
        total_params = (
            embedding_params + num_layers * layer_params + vocab_size * hidden_size
        )

        return total_params

    def _estimate_expert_communication_volume(self, config: ParallelismConfig) -> float:
        """Estimate expert communication volume for MoE models."""
        if not self.model_config.get("num_experts", 0):
            return 0.0

        # Simplified estimation - actual volume depends on routing patterns
        hidden_size = self.model_config.get("hidden_size", 4096)
        intermediate_size = self.model_config.get("intermediate_size", 11008)

        # Estimate communication volume per token (in GB/s)
        expert_comm_gb_per_token = (hidden_size * intermediate_size * 4) / (1024**3)

        # Assume moderate communication frequency
        return expert_comm_gb_per_token * 100  # Simplified estimate

    def _estimate_throughput(
        self, config_spec: ConfigurationSpec, workload_spec: WorkloadSpec
    ) -> float:
        """Estimate throughput for the configuration."""
        # Simplified throughput estimation
        config = config_spec.parallelism_config

        # Base throughput estimate (tokens/second)
        base_throughput = 1000 * workload_spec.batch_size

        # Scale by parallelism efficiency
        parallel_efficiency = 0.9 / max(1, config.pipeline_parallel_size * 0.1)

        return base_throughput * parallel_efficiency

    def _estimate_latency(
        self, config_spec: ConfigurationSpec, workload_spec: WorkloadSpec
    ) -> float:
        """Estimate latency for the configuration."""
        # Simplified latency estimation
        config = config_spec.parallelism_config

        # Base latency estimate (milliseconds)
        base_latency = 50

        # Pipeline parallelism adds latency due to bubbles
        pipeline_overhead = config.pipeline_parallel_size * 5

        return base_latency + pipeline_overhead


def validate_configuration_compatibility(
    config1: ParallelismConfig,
    config2: ParallelismConfig,
) -> ValidationResult:
    """Validate compatibility between two parallelism configurations.

    Args:
        config1: First parallelism configuration
        config2: Second parallelism configuration

    Returns:
        ValidationResult indicating compatibility
    """
    errors = []
    warnings = []
    recommendations = []

    # Check if configurations can coexist (e.g., for ensemble serving)
    if config1.total_gpus + config2.total_gpus > 64:
        warnings.append(
            f"Combined GPU usage ({config1.total_gpus + config2.total_gpus}) "
            "may require large cluster resources"
        )

    # Check for resource conflicts
    same_tp = config1.tensor_parallel_size == config2.tensor_parallel_size
    same_pp = config1.pipeline_parallel_size == config2.pipeline_parallel_size
    if same_tp and same_pp:
        recommendations.append(
            "Configurations have identical parallelism patterns. "
            "Consider diversifying for different workload optimization."
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        recommendations=recommendations,
    )
