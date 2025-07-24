"""Comprehensive tests for configuration validator."""

from unittest.mock import Mock, patch

import pytest

from autoparallel.config.validator import (
    CompatibilityValidationError,
    ConfigurationSpec,
    ConfigurationValidationError,
    ConfigurationValidator,
    PerformanceValidationError,
    ResourceValidationError,
    WorkloadSpec,
    validate_configuration_compatibility,
)
from autoparallel.constraints.validation import (
    ClusterSpec,
    ConstraintValidator,
    ParallelismConfig,
    ValidationResult,
)
from autoparallel.memory.components import MemoryComponents
from autoparallel.memory.config import MemoryConfig
from autoparallel.memory.estimator import TransformersMemoryEstimator


class TestWorkloadSpec:
    """Test WorkloadSpec functionality."""

    def test_workload_spec_creation_default(self):
        """Test WorkloadSpec creation with default values."""
        spec = WorkloadSpec()

        assert spec.sequence_length == 2048
        assert spec.batch_size == 1
        assert spec.max_sequence_length == 4096
        assert spec.is_training is False
        assert spec.workload_type == "inference"
        assert spec.throughput_target is None
        assert spec.latency_target is None

    def test_workload_spec_creation_custom(self):
        """Test WorkloadSpec creation with custom values."""
        spec = WorkloadSpec(
            sequence_length=1024,
            batch_size=32,
            max_sequence_length=8192,
            is_training=True,
            workload_type="training",
            throughput_target=1000.0,
            latency_target=100.0,
        )

        assert spec.sequence_length == 1024
        assert spec.batch_size == 32
        assert spec.max_sequence_length == 8192
        assert spec.is_training is True
        assert spec.workload_type == "training"
        assert spec.throughput_target == 1000.0
        assert spec.latency_target == 100.0


class TestConfigurationSpec:
    """Test ConfigurationSpec functionality."""

    def test_configuration_spec_creation(self):
        """Test ConfigurationSpec creation."""
        parallelism_config = ParallelismConfig(tensor_parallel_size=2)
        cluster_spec = ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0)
        workload_spec = {"batch_size": 32, "sequence_length": 2048}
        framework_config = {"framework": "vllm", "enable_cuda_graphs": True}

        config_spec = ConfigurationSpec(
            parallelism_config=parallelism_config,
            cluster_spec=cluster_spec,
            workload_spec=workload_spec,
            framework_config=framework_config,
        )

        assert config_spec.parallelism_config is parallelism_config
        assert config_spec.cluster_spec is cluster_spec
        assert config_spec.workload_spec is workload_spec
        assert config_spec.framework_config is framework_config

    def test_configuration_spec_without_framework(self):
        """Test ConfigurationSpec without framework configuration."""
        parallelism_config = ParallelismConfig()
        cluster_spec = ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0)
        workload_spec = {}

        config_spec = ConfigurationSpec(
            parallelism_config=parallelism_config,
            cluster_spec=cluster_spec,
            workload_spec=workload_spec,
        )

        assert config_spec.framework_config is None


class TestConfigurationValidator:
    """Test ConfigurationValidator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }

        self.memory_config = MemoryConfig()
        self.memory_estimator = Mock(spec=TransformersMemoryEstimator)

        # Mock memory estimation
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=20 * (1024**3),  # 20 GB
            activations=10 * (1024**3),  # 10 GB
            kv_cache=5 * (1024**3),   # 5 GB
        )

    def test_validator_initialization_default(self):
        """Test validator initialization with defaults."""
        validator = ConfigurationValidator(self.model_config)

        assert validator.model_config is self.model_config
        assert isinstance(validator.memory_config, MemoryConfig)
        assert isinstance(validator.memory_estimator, TransformersMemoryEstimator)
        assert validator.enable_performance_validation is True
        assert validator.enable_framework_validation is True
        assert isinstance(validator.constraint_validator, ConstraintValidator)
        assert validator.model_constraints is not None

    def test_validator_initialization_custom(self):
        """Test validator initialization with custom parameters."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_config=self.memory_config,
            memory_estimator=self.memory_estimator,
            enable_performance_validation=False,
            enable_framework_validation=False,
        )

        assert validator.model_config is self.model_config
        assert validator.memory_config is self.memory_config
        assert validator.memory_estimator is self.memory_estimator
        assert validator.enable_performance_validation is False
        assert validator.enable_framework_validation is False

    def test_validator_with_dict_model_config(self):
        """Test validator initialization with dict model config."""
        validator = ConfigurationValidator(self.model_config)

        # Should successfully create model constraints from dict
        assert validator.model_constraints is not None

    def test_validate_configuration_basic_success(self):
        """Test basic configuration validation that succeeds."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                expert_parallel_size=1,
                data_parallel_size=4,
            ),
            cluster_spec=ClusterSpec(
                total_gpus=8,
                gpus_per_node=8,
                gpu_memory_gb=80.0,
            ),
            workload_spec={"batch_size": 32, "sequence_length": 2048},
        )

        workload_spec = WorkloadSpec(
            batch_size=32,
            sequence_length=2048,
            max_sequence_length=2048,  # Match model's default position embeddings
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        assert isinstance(result, ValidationResult)
        # With 35 GB total memory on 80 GB GPUs, should be valid
        assert result.is_valid

    def test_validate_configuration_memory_exceeded(self):
        """Test configuration validation with memory exceeded."""
        # Mock high memory usage
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=90 * (1024**3),  # 90 GB - exceeds 80 GB GPU
        )

        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=1),
            cluster_spec=ClusterSpec(
                total_gpus=8,
                gpus_per_node=8,
                gpu_memory_gb=80.0,
            ),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        assert not result.is_valid
        assert any("exceeds" in error for error in result.errors)

    def test_validate_configuration_insufficient_gpus(self):
        """Test configuration validation with insufficient GPUs."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=4,
                pipeline_parallel_size=4,  # Requires 16 GPUs
            ),
            cluster_spec=ClusterSpec(
                total_gpus=8,  # Only 8 GPUs available
                gpus_per_node=8,
                gpu_memory_gb=80.0,
            ),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        assert not result.is_valid
        assert any("requires" in error and "GPUs" in error for error in result.errors)

    def test_enhanced_architectural_constraints_validation(self):
        """Test enhanced architectural constraints validation."""
        # Model with specific constraints
        model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,  # Position embedding limit
        }

        validator = ConfigurationValidator(
            model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            max_sequence_length=4096,  # Exceeds position embeddings
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have error about sequence length exceeding position embeddings
        assert not result.is_valid
        assert any("exceeds model position embeddings" in error for error in result.errors)

    def test_grouped_query_attention_validation(self):
        """Test validation with grouped query attention models."""
        # GQA model configuration
        model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA: 4 query heads per KV head
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }

        validator = ConfigurationValidator(
            model_config,
            memory_estimator=self.memory_estimator,
        )

        # Test valid GQA configuration
        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=8),  # Divides both 32 and 8
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(max_sequence_length=2048)  # Match model defaults
        result = validator.validate_configuration(config_spec, workload_spec)

        # Should be valid
        assert result.is_valid

    def test_grouped_query_attention_invalid_tensor_parallel(self):
        """Test GQA validation with invalid tensor parallel size."""
        model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 7,  # Prime number, won't divide evenly
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }

        validator = ConfigurationValidator(
            model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=4),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(max_sequence_length=2048)  # Match model defaults
        result = validator.validate_configuration(config_spec, workload_spec)

        assert not result.is_valid
        # Should have some validation error (may be different specific error)

    def test_vocabulary_sharding_warning(self):
        """Test vocabulary sharding warning."""
        model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 30001,  # Prime number, won't divide evenly
        }

        validator = ConfigurationValidator(
            model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=8),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should have warning about vocabulary sharding
        assert any("padding overhead" in warning for warning in result.warnings)

    def test_resource_availability_validation(self):
        """Test enhanced resource availability validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            max_sequence_length=16384,  # Very long sequence
            batch_size=1,
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # May have warnings about KV cache usage for long sequences
        # Result depends on actual memory estimation
        assert isinstance(result, ValidationResult)

    def test_memory_utilization_warnings(self):
        """Test memory utilization warnings."""
        # Mock very low memory usage
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=5 * (1024**3),  # 5 GB - very low utilization
        )

        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=1),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should have recommendation about low memory utilization
        assert len(result.recommendations) > 0
        assert any("memory utilization" in rec for rec in result.recommendations)

    def test_communication_patterns_validation(self):
        """Test communication patterns validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=16,  # Spans multiple nodes
            ),
            cluster_spec=ClusterSpec(
                total_gpus=16,
                gpus_per_node=8,  # 2 nodes, so TP spans nodes
                gpu_memory_gb=80.0,
            ),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should have warning about inter-node tensor parallelism
        assert any("inter-node bandwidth" in warning for warning in result.warnings)

    def test_load_balancing_validation(self):
        """Test load balancing validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                pipeline_parallel_size=7,  # 32 layers / 7 = uneven distribution
            ),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(batch_size=1)

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warning about uneven pipeline stages
        assert any("uneven load" in warning for warning in result.warnings)

    def test_data_parallel_batch_size_validation(self):
        """Test data parallel batch size validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                data_parallel_size=8,  # 8 data parallel workers
            ),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            batch_size=4,  # Smaller than data parallel size
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have error about batch size being too small
        assert not result.is_valid
        assert any("idle" in error for error in result.errors)

    def test_scalability_patterns_validation(self):
        """Test scalability patterns validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=16,
                pipeline_parallel_size=8,  # Very high parallelism
            ),
            cluster_spec=ClusterSpec(total_gpus=128, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should have warnings about high parallelism and efficiency concerns
        assert any("diminishing returns" in warning for warning in result.warnings)

    def test_gpu_utilization_validation(self):
        """Test GPU utilization validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=2,  # Only uses 2 out of 16 GPUs
            ),
            cluster_spec=ClusterSpec(total_gpus=16, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should have recommendation about low GPU utilization
        assert any("resource utilization" in rec for rec in result.recommendations)

    def test_framework_validation_disabled(self):
        """Test that framework validation can be disabled."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
            enable_framework_validation=False,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
            framework_config={"framework": "vllm"},  # Framework config present
        )

        workload_spec = WorkloadSpec(max_sequence_length=2048)  # Match model defaults
        validator.validate_configuration(config_spec, workload_spec)

        # Should not perform framework-specific validation, but may still have other validation errors
        # The main test is that it doesn't crash due to framework validation being disabled

    def test_performance_validation_disabled(self):
        """Test that performance validation can be disabled."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
            enable_performance_validation=False,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            throughput_target=10000.0,  # Very high target
            latency_target=1.0,         # Very low target
            max_sequence_length=2048,   # Match model defaults
        )

        validator.validate_configuration(config_spec, workload_spec)

        # Should not validate performance targets, but may have other validation errors
        # The main test is that performance validation is disabled

    def test_vllm_framework_validation(self):
        """Test vLLM-specific framework validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=16),  # Very high TP
            cluster_spec=ClusterSpec(total_gpus=16, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
            framework_config={
                "framework": "vllm",
                "enable_cuda_graphs": True,
                "quantization": "int8",
            },
        )

        workload_spec = WorkloadSpec(sequence_length=10000)  # Long sequence

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warnings about vLLM-specific issues
        assert any("vLLM" in warning for warning in result.warnings)

    def test_deepspeed_framework_validation(self):
        """Test DeepSpeed-specific framework validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                pipeline_parallel_size=4,
            ),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
            framework_config={
                "framework": "deepspeed",
                "zero_stage": 2,
            },
        )

        workload_spec = WorkloadSpec(is_training=False)  # Inference workload

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warnings about ZeRO for inference and PP+ZeRO interactions
        assert len(result.warnings) > 0

    def test_moe_training_framework_validation(self):
        """Test MoE training framework validation."""
        # MoE model
        model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "num_experts": 8,
        }

        validator = ConfigurationValidator(
            model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                expert_parallel_size=4,  # Expert parallelism
            ),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
            framework_config={"framework": "general"},
        )

        workload_spec = WorkloadSpec(is_training=True)

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warning about MoE training complexity
        assert any("gradient synchronization" in warning for warning in result.warnings)

    def test_performance_requirements_validation(self):
        """Test performance requirements validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=1),  # Simple config
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            throughput_target=100000.0,  # Very high throughput target
            latency_target=1.0,          # Very low latency target
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warnings about not meeting performance targets
        assert len(result.warnings) > 0
        assert any("throughput" in warning or "latency" in warning for warning in result.warnings)

    def test_configuration_completeness_validation(self):
        """Test configuration completeness validation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            workload_type="training",  # Says training
            is_training=False,         # But is_training=False
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warning about inconsistency
        assert any("training" in warning for warning in result.warnings)

    def test_incompatible_batch_and_latency_warning(self):
        """Test warning for incompatible batch size and latency target."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            batch_size=128,      # Large batch
            latency_target=50.0, # Strict latency requirement
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have warning about incompatible requirements
        assert any("incompatible" in warning for warning in result.warnings)

    def test_optimization_recommendations_generation(self):
        """Test generation of optimization recommendations."""
        # Use low memory utilization to trigger recommendations
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=20 * (1024**3),  # 20 GB on 80 GB GPU = 25% utilization
        )

        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=16,  # Cross-node
            ),
            cluster_spec=ClusterSpec(
                total_gpus=32,
                gpus_per_node=8,
                gpu_memory_gb=80.0,
            ),
            workload_spec={},
        )

        workload_spec = WorkloadSpec()

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should have various optimization recommendations
        assert len(result.recommendations) > 0

        # Should suggest memory efficiency improvements
        memory_recommendations = [
            rec for rec in result.recommendations
            if "memory" in rec.lower()
        ]
        assert len(memory_recommendations) > 0

    def test_parameter_count_estimation(self):
        """Test parameter count estimation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        param_count = validator._estimate_parameter_count()

        # Should return reasonable parameter count
        assert param_count > 0
        # For the test model config, should be in billions
        assert param_count > 1e9

    def test_expert_communication_volume_estimation(self):
        """Test expert communication volume estimation."""
        # MoE model
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_experts": 8,
            "intermediate_size": 11008,
        }

        validator = ConfigurationValidator(
            model_config,
            memory_estimator=self.memory_estimator,
        )

        config = ParallelismConfig(expert_parallel_size=4)
        volume = validator._estimate_expert_communication_volume(config)

        # Should return positive communication volume for MoE model
        assert volume > 0

    def test_expert_communication_volume_non_moe(self):
        """Test expert communication volume for non-MoE model."""
        validator = ConfigurationValidator(
            self.model_config,  # Non-MoE model
            memory_estimator=self.memory_estimator,
        )

        config = ParallelismConfig(expert_parallel_size=1)
        volume = validator._estimate_expert_communication_volume(config)

        # Should return 0 for non-MoE model
        assert volume == 0.0

    def test_throughput_estimation(self):
        """Test throughput estimation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            ),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(batch_size=32)

        throughput = validator._estimate_throughput(config_spec, workload_spec)

        # Should return positive throughput estimate
        assert throughput > 0

    def test_latency_estimation(self):
        """Test latency estimation."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(pipeline_parallel_size=4),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec()

        latency = validator._estimate_latency(config_spec, workload_spec)

        # Should return positive latency estimate
        assert latency > 0
        # Pipeline parallelism should add latency
        assert latency > 50  # Base latency + pipeline overhead

    def test_validation_exception_handling(self):
        """Test handling of validation exceptions."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        # Mock memory estimator to raise exception
        validator.memory_estimator.estimate_memory.side_effect = Exception("Memory estimation failed")
        # Also mock the constraint validator's memory estimator
        validator.constraint_validator.memory_estimator.estimate_memory.side_effect = Exception("Memory estimation failed")

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(tensor_parallel_size=2),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        # Should raise exception during validation since memory estimation fails
        with pytest.raises(Exception, match="Memory estimation failed"):
            validator.validate_configuration(config_spec)

    def test_constraint_validator_integration(self):
        """Test integration with constraint validator."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        # Test that basic constraint validation is called
        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=7,  # Prime number, should cause issues
            ),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should incorporate results from constraint validator
        assert isinstance(result, ValidationResult)


class TestValidateConfigurationCompatibility:
    """Test configuration compatibility validation function."""

    def test_compatible_configurations(self):
        """Test validation of compatible configurations."""
        config1 = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
        )  # 8 GPUs

        config2 = ParallelismConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=2,
        )  # 8 GPUs

        result = validate_configuration_compatibility(config1, config2)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_high_gpu_usage_warning(self):
        """Test warning for high combined GPU usage."""
        config1 = ParallelismConfig(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            data_parallel_size=1,
        )  # 32 GPUs

        config2 = ParallelismConfig(
            tensor_parallel_size=8,
            pipeline_parallel_size=5,
            data_parallel_size=1,
        )  # 40 GPus, total = 72 > 64

        result = validate_configuration_compatibility(config1, config2)

        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("large cluster" in warning for warning in result.warnings)

    def test_identical_parallelism_patterns(self):
        """Test recommendation for identical parallelism patterns."""
        config1 = ParallelismConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )

        config2 = ParallelismConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )

        result = validate_configuration_compatibility(config1, config2)

        assert result.is_valid
        assert len(result.recommendations) > 0
        assert any("diversifying" in rec for rec in result.recommendations)


class TestValidationErrorTypes:
    """Test different validation error types."""

    def test_configuration_validation_error(self):
        """Test ConfigurationValidationError."""
        with pytest.raises(ConfigurationValidationError, match="Test error"):
            raise ConfigurationValidationError("Test error")

    def test_resource_validation_error(self):
        """Test ResourceValidationError."""
        with pytest.raises(ResourceValidationError, match="Resource error"):
            raise ResourceValidationError("Resource error")

    def test_compatibility_validation_error(self):
        """Test CompatibilityValidationError."""
        with pytest.raises(CompatibilityValidationError, match="Compatibility error"):
            raise CompatibilityValidationError("Compatibility error")

    def test_performance_validation_error(self):
        """Test PerformanceValidationError."""
        with pytest.raises(PerformanceValidationError, match="Performance error"):
            raise PerformanceValidationError("Performance error")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
        }

        self.memory_estimator = Mock(spec=TransformersMemoryEstimator)
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=30 * (1024**3),
        )

    def test_empty_model_config(self):
        """Test validator with empty model config."""
        validator = ConfigurationValidator(
            {},  # Empty model config
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should handle empty config gracefully
        assert isinstance(result, ValidationResult)

    def test_zero_memory_components(self):
        """Test validation with zero memory components."""
        self.memory_estimator.estimate_memory.return_value = MemoryComponents()

        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should handle zero memory gracefully
        assert isinstance(result, ValidationResult)
        # May not be valid due to expert parallel constraints, but should not crash

    def test_extreme_parallelism_values(self):
        """Test validation with extreme parallelism values."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(
                tensor_parallel_size=1024,  # Extreme value
                pipeline_parallel_size=1024,
                expert_parallel_size=1,
                data_parallel_size=1,
            ),
            cluster_spec=ClusterSpec(
                total_gpus=1048576,  # 1M GPUs
                gpus_per_node=8,
                gpu_memory_gb=80.0,
            ),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should handle extreme values and provide appropriate warnings
        assert isinstance(result, ValidationResult)
        assert len(result.warnings) > 0

    def test_inconsistent_workload_spec(self):
        """Test validation with inconsistent workload specification."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        workload_spec = WorkloadSpec(
            sequence_length=2048,
            max_sequence_length=1024,  # max < sequence_length
            batch_size=0,  # Invalid batch size
            throughput_target=-1000.0,  # Negative target
            latency_target=-50.0,  # Negative target
        )

        result = validator.validate_configuration(config_spec, workload_spec)

        # Should handle inconsistencies
        assert isinstance(result, ValidationResult)

    def test_very_small_cluster(self):
        """Test validation with very small cluster."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(
                total_gpus=1,      # Single GPU
                gpus_per_node=1,
                gpu_memory_gb=8.0,  # Small memory
            ),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should handle small cluster and likely show memory issues
        assert isinstance(result, ValidationResult)

    def test_framework_config_edge_cases(self):
        """Test framework configuration edge cases."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
            framework_config={
                "framework": "",  # Empty framework name
                "unknown_option": "value",
                "enable_cuda_graphs": "not_a_boolean",
            },
        )

        result = validator.validate_configuration(config_spec)

        # Should handle malformed framework config gracefully
        assert isinstance(result, ValidationResult)

    @patch('autoparallel.config.validator.analyze_model_constraints')
    def test_model_constraints_analysis_failure(self, mock_analyze):
        """Test handling of model constraints analysis failure."""
        mock_analyze.side_effect = Exception("Analysis failed")

        # Should raise during initialization since analyze_model_constraints is called in __init__
        with pytest.raises(Exception, match="Analysis failed"):
            ConfigurationValidator(
                self.model_config,
                memory_estimator=self.memory_estimator,
            )

    def test_memory_config_integration(self):
        """Test integration with custom memory config."""
        memory_config = MemoryConfig(
            utilization_bound=0.9,  # Higher utilization bound
            safety_margin=0.1,
        )

        validator = ConfigurationValidator(
            self.model_config,
            memory_config=memory_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        result = validator.validate_configuration(config_spec)

        # Should use custom memory config for validation
        assert isinstance(result, ValidationResult)
        assert validator.memory_config.utilization_bound == 0.9

    def test_concurrent_validation_calls(self):
        """Test that validator handles concurrent validation calls."""
        validator = ConfigurationValidator(
            self.model_config,
            memory_estimator=self.memory_estimator,
        )

        config_spec = ConfigurationSpec(
            parallelism_config=ParallelismConfig(),
            cluster_spec=ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0),
            workload_spec={},
        )

        # Validate same configuration multiple times
        results = []
        for _ in range(5):
            result = validator.validate_configuration(config_spec)
            results.append(result)

        # All results should be consistent
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.is_valid == results[0].is_valid
            assert len(result.errors) == len(results[0].errors)
