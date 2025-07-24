"""Constraint validation for autoparallel configurations.

This module provides comprehensive validation and sanity checking functions
for parallelism configurations, ensuring they respect both model architecture
constraints and hardware limitations.
"""

import math
from dataclasses import dataclass
from typing import Any

from autoparallel.memory.config import MemoryConfig
from autoparallel.memory.estimator import MemoryEstimator, TransformersMemoryEstimator


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class TensorParallelValidationError(ValidationError):
    """Exception for tensor parallelism validation errors."""

    pass


class PipelineParallelValidationError(ValidationError):
    """Exception for pipeline parallelism validation errors."""

    pass


class ExpertParallelValidationError(ValidationError):
    """Exception for expert parallelism validation errors."""

    pass


class CrossConstraintValidationError(ValidationError):
    """Exception for cross-component constraint validation errors."""

    pass


class ClusterResourceValidationError(ValidationError):
    """Exception for cluster resource constraint validation errors."""

    pass


@dataclass
class ValidationResult:
    """Result of constraint validation.

    Attributes:
        is_valid: Whether the configuration is valid
        errors: List of validation error messages
        warnings: List of validation warning messages
        recommendations: List of optimization recommendations
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    recommendations: list[str]

    def __bool__(self) -> bool:
        """Return True if validation passed."""
        return self.is_valid


@dataclass
class ParallelismConfig:
    """Parallelism configuration for validation.

    Attributes:
        tensor_parallel_size: Tensor parallelism size
        pipeline_parallel_size: Pipeline parallelism size
        expert_parallel_size: Expert parallelism size (for MoE models)
        data_parallel_size: Data parallelism size
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1

    @property
    def total_gpus(self) -> int:
        """Calculate total GPUs required for this configuration."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.data_parallel_size
        )


@dataclass
class ClusterSpec:
    """GPU cluster specification for validation.

    Attributes:
        total_gpus: Total number of GPUs in cluster
        gpus_per_node: Number of GPUs per node
        gpu_memory_gb: GPU memory per device in GB
        inter_node_bandwidth_gbps: Inter-node bandwidth in Gbps
        intra_node_bandwidth_gbps: Intra-node bandwidth in Gbps
    """

    total_gpus: int
    gpus_per_node: int
    gpu_memory_gb: float
    inter_node_bandwidth_gbps: float = 200.0
    intra_node_bandwidth_gbps: float = 900.0

    @property
    def num_nodes(self) -> int:
        """Calculate number of nodes."""
        return math.ceil(self.total_gpus / self.gpus_per_node)


class ConstraintValidator:
    """Comprehensive constraint validator for parallelism configurations.

    This class validates parallelism configurations against model architecture
    constraints, hardware limitations, and cross-component compatibility.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        memory_config: MemoryConfig | None = None,
        memory_estimator: MemoryEstimator | None = None,
    ):
        """Initialize constraint validator.

        Args:
            model_config: Transformer model configuration dictionary
            memory_config: Memory configuration for validation
            memory_estimator: Memory estimation component
        """
        self.model_config = model_config
        self.memory_config = memory_config or MemoryConfig()
        self.memory_estimator = memory_estimator or TransformersMemoryEstimator(
            self.memory_config
        )

        # Extract model architecture parameters
        self._extract_model_parameters()

    def _extract_model_parameters(self) -> None:
        """Extract key model parameters for validation."""
        self.hidden_size = self.model_config.get("hidden_size", 4096)
        self.num_attention_heads = self.model_config.get("num_attention_heads", 32)
        self.num_key_value_heads = self.model_config.get(
            "num_key_value_heads", self.num_attention_heads
        )
        self.num_layers = self.model_config.get("num_hidden_layers", 32)
        self.vocab_size = self.model_config.get("vocab_size", 32000)
        self.intermediate_size = self.model_config.get(
            "intermediate_size", 4 * self.hidden_size
        )

        # MoE parameters
        self.num_experts = self.model_config.get("num_experts", 0)
        self.num_experts_per_token = self.model_config.get("num_experts_per_token", 2)

        # Architecture flags
        self.tie_word_embeddings = self.model_config.get("tie_word_embeddings", False)
        self.is_moe = self.num_experts > 0

    def validate_configuration(
        self,
        parallelism_config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        sequence_length: int = 2048,
        batch_size: int = 1,
        is_training: bool = False,
    ) -> ValidationResult:
        """Validate a complete parallelism configuration.

        Args:
            parallelism_config: Parallelism configuration to validate
            cluster_spec: Target cluster specification
            sequence_length: Input sequence length
            batch_size: Batch size
            is_training: Whether configuration is for training

        Returns:
            ValidationResult with detailed validation feedback
        """
        errors = []
        warnings = []
        recommendations = []

        # Run individual constraint checks
        try:
            self._validate_tensor_parallel_constraints(
                parallelism_config, errors, warnings, recommendations
            )
        except TensorParallelValidationError as e:
            errors.append(str(e))

        try:
            self._validate_pipeline_parallel_constraints(
                parallelism_config, errors, warnings, recommendations
            )
        except PipelineParallelValidationError as e:
            errors.append(str(e))

        try:
            self._validate_expert_parallel_constraints(
                parallelism_config, errors, warnings, recommendations
            )
        except ExpertParallelValidationError as e:
            errors.append(str(e))

        try:
            self._validate_cross_constraints(
                parallelism_config, cluster_spec, errors, warnings, recommendations
            )
        except (CrossConstraintValidationError, ClusterResourceValidationError) as e:
            errors.append(str(e))

        try:
            self._validate_memory_constraints(
                parallelism_config,
                cluster_spec,
                sequence_length,
                batch_size,
                is_training,
                errors,
                warnings,
                recommendations,
            )
        except ValidationError as e:
            errors.append(str(e))

        try:
            self._validate_topology_constraints(
                parallelism_config, cluster_spec, errors, warnings, recommendations
            )
        except ValidationError as e:
            errors.append(str(e))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_tensor_parallel_constraints(
        self,
        config: ParallelismConfig,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate tensor parallelism constraints."""
        tp_size = config.tensor_parallel_size

        if tp_size <= 0:
            raise TensorParallelValidationError(
                f"Tensor parallel size must be positive, got {tp_size}"
            )

        # Check attention head divisibility
        if self.num_attention_heads % tp_size != 0:
            raise TensorParallelValidationError(
                f"Number of attention heads ({self.num_attention_heads}) must be "
                f"divisible by tensor parallel size ({tp_size})"
            )

        # Check key-value head divisibility for GQA models
        if self.num_key_value_heads % tp_size != 0:
            raise TensorParallelValidationError(
                f"Number of key-value heads ({self.num_key_value_heads}) must be "
                f"divisible by tensor parallel size ({tp_size})"
            )

        # Check hidden size divisibility
        if self.hidden_size % tp_size != 0:
            errors.append(
                f"Hidden size ({self.hidden_size}) should be divisible by "
                f"tensor parallel size ({tp_size}) for optimal performance"
            )

        # Check intermediate size divisibility
        if self.intermediate_size % tp_size != 0:
            errors.append(
                f"Intermediate size ({self.intermediate_size}) should be divisible by "
                f"tensor parallel size ({tp_size}) for optimal MLP sharding"
            )

        # Check vocabulary size divisibility for embedding sharding
        if self.vocab_size % tp_size != 0:
            warnings.append(
                f"Vocabulary size ({self.vocab_size}) is not divisible by "
                f"tensor parallel size ({tp_size}). This may cause padding overhead."
            )

        # Performance recommendations
        if tp_size > 8 and not self._is_power_of_2(tp_size):
            recommendations.append(
                f"Consider using power-of-2 tensor parallel size instead of {tp_size} "
                "for better communication efficiency"
            )

    def _validate_pipeline_parallel_constraints(
        self,
        config: ParallelismConfig,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate pipeline parallelism constraints."""
        pp_size = config.pipeline_parallel_size

        if pp_size <= 0:
            raise PipelineParallelValidationError(
                f"Pipeline parallel size must be positive, got {pp_size}"
            )

        if pp_size > self.num_layers:
            raise PipelineParallelValidationError(
                f"Pipeline parallel size ({pp_size}) cannot exceed number of "
                f"layers ({self.num_layers})"
            )

        # Check minimum layers per stage
        min_layers_per_stage = 2  # Configurable parameter
        layers_per_stage = self.num_layers / pp_size

        if layers_per_stage < min_layers_per_stage:
            warnings.append(
                f"Pipeline parallel size ({pp_size}) results in "
                f"{layers_per_stage:.1f} layers per stage, which is below "
                f"recommended minimum of {min_layers_per_stage}"
            )

        # Check for load balancing issues
        if self.num_layers % pp_size != 0:
            warnings.append(
                f"Number of layers ({self.num_layers}) is not evenly divisible by "
                f"pipeline parallel size ({pp_size}). Some stages will have "
                "more layers than others, causing load imbalance."
            )

        # Performance recommendations
        if pp_size > 4:
            recommendations.append(
                f"Pipeline parallel size of {pp_size} may introduce significant "
                "pipeline bubble overhead. Consider reducing PP size and "
                "increasing TP or DP instead."
            )

    def _validate_expert_parallel_constraints(
        self,
        config: ParallelismConfig,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate expert parallelism constraints for MoE models."""
        ep_size = config.expert_parallel_size

        if ep_size <= 0:
            raise ExpertParallelValidationError(
                f"Expert parallel size must be positive, got {ep_size}"
            )

        # Non-MoE models should use EP=1
        if not self.is_moe and ep_size > 1:
            raise ExpertParallelValidationError(
                f"Expert parallel size ({ep_size}) must be 1 for non-MoE models"
            )

        # MoE-specific constraints
        if self.is_moe:
            if self.num_experts % ep_size != 0:
                raise ExpertParallelValidationError(
                    f"Number of experts ({self.num_experts}) must be divisible by "
                    f"expert parallel size ({ep_size})"
                )

            # Check minimum experts per device
            min_experts_per_device = 1
            experts_per_device = self.num_experts / ep_size

            if experts_per_device < min_experts_per_device:
                raise ExpertParallelValidationError(
                    f"Expert parallel size ({ep_size}) results in "
                    f"{experts_per_device:.1f} experts per device, which is below "
                    f"minimum of {min_experts_per_device}"
                )

            # Performance recommendations
            if ep_size > 8:
                recommendations.append(
                    f"Expert parallel size of {ep_size} may cause excessive "
                    "all-to-all communication overhead. Consider reducing EP size."
                )

    def _validate_cross_constraints(
        self,
        config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate cross-component constraints."""
        total_gpus_needed = config.total_gpus

        # Check total GPU requirement
        if total_gpus_needed > cluster_spec.total_gpus:
            raise ClusterResourceValidationError(
                f"Configuration requires {total_gpus_needed} GPUs but cluster "
                f"only has {cluster_spec.total_gpus} GPUs available"
            )

        if total_gpus_needed != cluster_spec.total_gpus:
            warnings.append(
                f"Configuration uses {total_gpus_needed} GPUs out of "
                f"{cluster_spec.total_gpus} available. Consider increasing "
                "data parallelism to utilize all resources."
            )

        # Validate parallelism product
        expected_total = (
            config.tensor_parallel_size
            * config.pipeline_parallel_size
            * config.expert_parallel_size
            * config.data_parallel_size
        )

        if expected_total != total_gpus_needed:
            raise CrossConstraintValidationError(
                f"Parallelism sizes don't multiply correctly: "
                f"{config.tensor_parallel_size} * {config.pipeline_parallel_size} * "
                f"{config.expert_parallel_size} * {config.data_parallel_size} = "
                f"{expected_total}, but total_gpus = {total_gpus_needed}"
            )

    def _validate_memory_constraints(
        self,
        config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        sequence_length: int,
        batch_size: int,
        is_training: bool,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate memory constraints."""
        # Estimate memory usage
        memory_components = self.memory_estimator.estimate_memory(
            model_config=self.model_config,
            sequence_length=sequence_length,
            batch_size=batch_size,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            expert_parallel_size=config.expert_parallel_size,
            is_training=is_training,
        )

        # Convert bytes to GB
        memory_per_gpu_gb = memory_components.total_memory / (1024**3)
        available_gpu_memory_gb = cluster_spec.gpu_memory_gb

        # Check if memory fits
        if memory_per_gpu_gb > available_gpu_memory_gb:
            raise ValidationError(
                f"Estimated memory usage ({memory_per_gpu_gb:.2f} GB) exceeds "
                f"available GPU memory ({available_gpu_memory_gb:.2f} GB)"
            )

        # Check memory utilization bounds
        utilization = memory_per_gpu_gb / available_gpu_memory_gb

        if utilization > self.memory_config.utilization_bound:
            errors.append(
                f"Memory utilization ({utilization:.1%}) exceeds safety bound "
                f"({self.memory_config.utilization_bound:.1%})"
            )

        if utilization < 0.5:
            warnings.append(
                f"Low memory utilization ({utilization:.1%}). Consider increasing "
                "batch size or reducing parallelism to better utilize GPU memory."
            )

        # Memory efficiency recommendations
        if utilization > 0.9:
            recommendations.append(
                f"High memory utilization ({utilization:.1%}). Consider increasing "
                "tensor or pipeline parallelism to reduce memory per GPU."
            )

    def _validate_topology_constraints(
        self,
        config: ParallelismConfig,
        cluster_spec: ClusterSpec,
        errors: list[str],
        warnings: list[str],
        recommendations: list[str],
    ) -> None:
        """Validate network topology constraints."""
        tp_size = config.tensor_parallel_size
        pp_size = config.pipeline_parallel_size
        ep_size = config.expert_parallel_size

        # Check if tensor parallelism fits within nodes
        if tp_size > cluster_spec.gpus_per_node:
            warnings.append(
                f"Tensor parallel size ({tp_size}) exceeds GPUs per node "
                f"({cluster_spec.gpus_per_node}). This will require inter-node "
                "communication and may reduce performance."
            )

        # Check if expert parallelism fits within nodes (recommended)
        if ep_size > cluster_spec.gpus_per_node:
            warnings.append(
                f"Expert parallel size ({ep_size}) exceeds GPUs per node "
                f"({cluster_spec.gpus_per_node}). Expert routing across nodes "
                "may introduce communication overhead."
            )

        # Check pipeline parallelism vs node count
        if pp_size > cluster_spec.num_nodes:
            errors.append(
                f"Pipeline parallel size ({pp_size}) exceeds number of nodes "
                f"({cluster_spec.num_nodes}). Cannot place multiple pipeline "
                "stages on the same node efficiently."
            )

        # Topology optimization recommendations
        if tp_size <= cluster_spec.gpus_per_node and pp_size <= cluster_spec.num_nodes:
            recommendations.append(
                "Configuration respects topology constraints. Tensor parallelism "
                "will use intra-node communication for optimal performance."
            )

    def get_valid_tensor_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid tensor parallel sizes for the model.

        Args:
            max_gpus: Maximum number of GPUs to consider

        Returns:
            List of valid tensor parallel sizes
        """
        valid_sizes = []

        for tp_size in range(1, max_gpus + 1):
            if (
                self.num_attention_heads % tp_size == 0
                and self.num_key_value_heads % tp_size == 0
            ):
                valid_sizes.append(tp_size)

        return valid_sizes

    def get_valid_expert_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid expert parallel sizes for MoE models.

        Args:
            max_gpus: Maximum number of GPUs to consider

        Returns:
            List of valid expert parallel sizes
        """
        if not self.is_moe:
            return [1]

        valid_sizes = []

        for ep_size in range(1, min(max_gpus, self.num_experts) + 1):
            if self.num_experts % ep_size == 0:
                valid_sizes.append(ep_size)

        return valid_sizes

    def get_valid_pipeline_parallel_sizes(self, max_stages: int) -> list[int]:
        """Get valid pipeline parallel sizes for the model.

        Args:
            max_stages: Maximum number of pipeline stages to consider

        Returns:
            List of valid pipeline parallel sizes
        """
        valid_sizes = []
        min_layers_per_stage = 2  # Configurable

        for pp_size in range(1, min(max_stages, self.num_layers) + 1):
            if self.num_layers / pp_size >= min_layers_per_stage:
                valid_sizes.append(pp_size)

        return valid_sizes

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """Check if a number is a power of 2."""
        return n > 0 and (n & (n - 1)) == 0


def validate_parallelism_combination(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    data_parallel_size: int,
    total_gpus: int,
) -> ValidationResult:
    """Validate a parallelism combination for basic consistency.

    This is a lightweight validation function that checks basic mathematical
    constraints without requiring model configuration.

    Args:
        tensor_parallel_size: Tensor parallelism size
        pipeline_parallel_size: Pipeline parallelism size
        expert_parallel_size: Expert parallelism size
        data_parallel_size: Data parallelism size
        total_gpus: Total available GPUs

    Returns:
        ValidationResult indicating if combination is valid
    """
    errors = []
    warnings = []
    recommendations = []

    # Check positive values
    sizes = [
        ("tensor_parallel_size", tensor_parallel_size),
        ("pipeline_parallel_size", pipeline_parallel_size),
        ("expert_parallel_size", expert_parallel_size),
        ("data_parallel_size", data_parallel_size),
    ]

    for name, size in sizes:
        if size <= 0:
            errors.append(f"{name} must be positive, got {size}")

    # Check GPU allocation
    required_gpus = (
        tensor_parallel_size
        * pipeline_parallel_size
        * expert_parallel_size
        * data_parallel_size
    )

    if required_gpus != total_gpus:
        errors.append(
            f"Parallelism sizes multiply to {required_gpus} but "
            f"total GPUs is {total_gpus}"
        )

    # Performance warnings
    if pipeline_parallel_size > 4:
        warnings.append(
            f"High pipeline parallelism ({pipeline_parallel_size}) may "
            "cause significant bubble overhead"
        )

    if tensor_parallel_size > 8:
        warnings.append(
            f"High tensor parallelism ({tensor_parallel_size}) may "
            "cause communication bottlenecks"
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        recommendations=recommendations,
    )


def get_divisors(n: int, max_divisor: int | None = None) -> list[int]:
    """Get all divisors of a number up to max_divisor.

    Args:
        n: Number to find divisors for
        max_divisor: Maximum divisor to consider

    Returns:
        List of divisors in ascending order
    """
    if max_divisor is None:
        max_divisor = n

    divisors = []
    for i in range(1, min(int(n**0.5) + 1, max_divisor + 1)):
        if n % i == 0:
            divisors.append(i)
            if i != n // i and n // i <= max_divisor:
                divisors.append(n // i)

    return sorted(divisors)


def get_power_of_2_divisors(n: int, max_divisor: int | None = None) -> list[int]:
    """Get power-of-2 divisors of a number.

    Args:
        n: Number to find divisors for
        max_divisor: Maximum divisor to consider

    Returns:
        List of power-of-2 divisors in ascending order
    """
    all_divisors = get_divisors(n, max_divisor)
    return [d for d in all_divisors if d > 0 and (d & (d - 1)) == 0]
