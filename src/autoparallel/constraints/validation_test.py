"""Tests for constraint validation functions."""

import pytest

from autoparallel.constraints.validation import (
    ClusterResourceValidationError,
    ClusterSpec,
    ConstraintValidator,
    CrossConstraintValidationError,
    ExpertParallelValidationError,
    ParallelismConfig,
    PipelineParallelValidationError,
    TensorParallelValidationError,
    ValidationResult,
    get_divisors,
    get_power_of_2_divisors,
    validate_parallelism_combination,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation and properties."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["warning"],
            recommendations=["recommendation"]
        )

        assert result.is_valid
        assert bool(result) is True
        assert result.errors == []
        assert result.warnings == ["warning"]
        assert result.recommendations == ["recommendation"]

    def test_validation_result_invalid(self):
        """Test ValidationResult with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=["error1", "error2"],
            warnings=[],
            recommendations=[]
        )

        assert not result.is_valid
        assert bool(result) is False
        assert result.errors == ["error1", "error2"]


class TestParallelismConfig:
    """Tests for ParallelismConfig class."""

    def test_parallelism_config_default(self):
        """Test default ParallelismConfig values."""
        config = ParallelismConfig()

        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.expert_parallel_size == 1
        assert config.data_parallel_size == 1
        assert config.total_gpus == 1

    def test_parallelism_config_custom(self):
        """Test custom ParallelismConfig values."""
        config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            expert_parallel_size=2,
            data_parallel_size=2
        )

        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 4
        assert config.expert_parallel_size == 2
        assert config.data_parallel_size == 2
        assert config.total_gpus == 32  # 2 * 4 * 2 * 2


class TestClusterSpec:
    """Tests for ClusterSpec class."""

    def test_cluster_spec_creation(self):
        """Test ClusterSpec creation and properties."""
        spec = ClusterSpec(
            total_gpus=16,
            gpus_per_node=8,
            gpu_memory_gb=80.0
        )

        assert spec.total_gpus == 16
        assert spec.gpus_per_node == 8
        assert spec.gpu_memory_gb == 80.0
        assert spec.num_nodes == 2

    def test_cluster_spec_num_nodes_calculation(self):
        """Test num_nodes calculation."""
        # Exact division
        spec = ClusterSpec(total_gpus=16, gpus_per_node=8, gpu_memory_gb=80.0)
        assert spec.num_nodes == 2

        # Ceiling division
        spec = ClusterSpec(total_gpus=17, gpus_per_node=8, gpu_memory_gb=80.0)
        assert spec.num_nodes == 3


class TestConstraintValidator:
    """Tests for ConstraintValidator class."""

    def test_validator_initialization(self):
        """Test ConstraintValidator initialization."""
        model_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000
        }

        validator = ConstraintValidator(model_config)

        assert validator.hidden_size == 4096
        assert validator.num_attention_heads == 32
        assert validator.num_layers == 32
        assert validator.vocab_size == 32000
        assert validator.intermediate_size == 16384  # 4 * hidden_size

    def test_validator_with_defaults(self):
        """Test ConstraintValidator with default values."""
        model_config = {}
        validator = ConstraintValidator(model_config)

        # Should use defaults
        assert validator.hidden_size == 4096
        assert validator.num_attention_heads == 32
        assert validator.num_layers == 32
        assert validator.vocab_size == 32000

    def test_validator_moe_detection(self):
        """Test MoE model detection."""
        # Non-MoE model
        model_config = {"num_experts": 0}
        validator = ConstraintValidator(model_config)
        assert not validator.is_moe

        # MoE model
        model_config = {"num_experts": 8}
        validator = ConstraintValidator(model_config)
        assert validator.is_moe
        assert validator.num_experts == 8

    def test_get_valid_tensor_parallel_sizes(self):
        """Test valid tensor parallel size calculation."""
        model_config = {
            "num_attention_heads": 32,
            "num_key_value_heads": 8  # GQA
        }
        validator = ConstraintValidator(model_config)

        valid_sizes = validator.get_valid_tensor_parallel_sizes(max_gpus=16)

        # Should be limited by KV heads (8)
        assert all(size <= 8 for size in valid_sizes)
        assert all(32 % size == 0 and 8 % size == 0 for size in valid_sizes)
        assert 1 in valid_sizes
        assert 2 in valid_sizes
        assert 4 in valid_sizes
        assert 8 in valid_sizes

    def test_get_valid_expert_parallel_sizes_non_moe(self):
        """Test expert parallel sizes for non-MoE models."""
        model_config = {"num_experts": 0}
        validator = ConstraintValidator(model_config)

        valid_sizes = validator.get_valid_expert_parallel_sizes(max_gpus=8)
        assert valid_sizes == [1]

    def test_get_valid_expert_parallel_sizes_moe(self):
        """Test expert parallel sizes for MoE models."""
        model_config = {"num_experts": 8}
        validator = ConstraintValidator(model_config)

        valid_sizes = validator.get_valid_expert_parallel_sizes(max_gpus=16)

        # Should include divisors of 8
        assert 1 in valid_sizes
        assert 2 in valid_sizes
        assert 4 in valid_sizes
        assert 8 in valid_sizes
        assert all(8 % size == 0 for size in valid_sizes)

    def test_get_valid_pipeline_parallel_sizes(self):
        """Test valid pipeline parallel size calculation."""
        model_config = {"num_hidden_layers": 24}
        validator = ConstraintValidator(model_config)

        valid_sizes = validator.get_valid_pipeline_parallel_sizes(max_stages=16)

        # With 24 layers and min 2 layers per stage, max is 12
        assert all(size <= 12 for size in valid_sizes)
        assert all(24 / size >= 2 for size in valid_sizes)
        assert 1 in valid_sizes


class TestTensorParallelValidation:
    """Tests for tensor parallel constraint validation."""

    def test_valid_tensor_parallel_config(self):
        """Test valid tensor parallel configuration."""
        model_config = {
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "vocab_size": 32000
        }
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(tensor_parallel_size=8)
        cluster_spec = ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert result.is_valid

    def test_invalid_tensor_parallel_attention_heads(self):
        """Test tensor parallel validation with incompatible attention heads."""
        model_config = {
            "num_attention_heads": 7,  # Prime number
            "num_key_value_heads": 7
        }
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(tensor_parallel_size=4)
        cluster_spec = ClusterSpec(total_gpus=4, gpus_per_node=4, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("attention heads" in error for error in result.errors)

    def test_invalid_tensor_parallel_kv_heads(self):
        """Test tensor parallel validation with incompatible KV heads."""
        model_config = {
            "num_attention_heads": 32,
            "num_key_value_heads": 7  # Prime number
        }
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(tensor_parallel_size=4)
        cluster_spec = ClusterSpec(total_gpus=4, gpus_per_node=4, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("key-value heads" in error for error in result.errors)

    def test_tensor_parallel_zero_size(self):
        """Test tensor parallel validation with zero size."""
        model_config = {"num_attention_heads": 32}
        validator = ConstraintValidator(model_config)

        # Test the tensor parallel constraint validation directly
        errors = []
        warnings = []
        recommendations = []

        config = ParallelismConfig(tensor_parallel_size=0)

        # This should raise TensorParallelValidationError
        with pytest.raises(TensorParallelValidationError, match="must be positive"):
            validator._validate_tensor_parallel_constraints(
                config, errors, warnings, recommendations
            )


class TestPipelineParallelValidation:
    """Tests for pipeline parallel constraint validation."""

    def test_valid_pipeline_parallel_config(self):
        """Test valid pipeline parallel configuration."""
        model_config = {"num_hidden_layers": 24}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(pipeline_parallel_size=4)
        cluster_spec = ClusterSpec(total_gpus=4, gpus_per_node=1, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        # Should have warnings about layers per stage but still be valid
        assert result.is_valid or len(result.warnings) > 0

    def test_pipeline_parallel_exceeds_layers(self):
        """Test pipeline parallel size exceeding number of layers."""
        model_config = {"num_hidden_layers": 4}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(pipeline_parallel_size=8)
        cluster_spec = ClusterSpec(total_gpus=8, gpus_per_node=1, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("cannot exceed number of layers" in error for error in result.errors)

    def test_pipeline_parallel_zero_size(self):
        """Test pipeline parallel validation with zero size."""
        model_config = {"num_hidden_layers": 24}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(pipeline_parallel_size=0)
        cluster_spec = ClusterSpec(total_gpus=1, gpus_per_node=1, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("must be positive" in error for error in result.errors)


class TestExpertParallelValidation:
    """Tests for expert parallel constraint validation."""

    def test_valid_expert_parallel_moe_config(self):
        """Test valid expert parallel configuration for MoE models."""
        model_config = {"num_experts": 8}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(expert_parallel_size=4)
        cluster_spec = ClusterSpec(total_gpus=4, gpus_per_node=4, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert result.is_valid

    def test_expert_parallel_non_moe_invalid(self):
        """Test expert parallel size > 1 for non-MoE models."""
        model_config = {"num_experts": 0}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(expert_parallel_size=4)
        cluster_spec = ClusterSpec(total_gpus=4, gpus_per_node=4, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("must be 1 for non-MoE models" in error for error in result.errors)

    def test_expert_parallel_not_divisible(self):
        """Test expert parallel size not dividing number of experts."""
        model_config = {"num_experts": 7}  # Prime number
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(expert_parallel_size=4)
        cluster_spec = ClusterSpec(total_gpus=4, gpus_per_node=4, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("must be divisible by" in error for error in result.errors)

    def test_expert_parallel_zero_size(self):
        """Test expert parallel validation with zero size."""
        model_config = {"num_experts": 8}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(expert_parallel_size=0)
        cluster_spec = ClusterSpec(total_gpus=1, gpus_per_node=1, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("must be positive" in error for error in result.errors)


class TestCrossConstraintValidation:
    """Tests for cross-component constraint validation."""

    def test_insufficient_gpus(self):
        """Test configuration requiring more GPUs than available."""
        model_config = {}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=4
        )  # Requires 16 GPUs
        cluster_spec = ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("requires" in error and "GPUs" in error for error in result.errors)

    def test_parallelism_product_mismatch(self):
        """Test parallelism sizes not matching total GPUs."""
        model_config = {}
        validator = ConstraintValidator(model_config)

        # This should be caught by the parallelism config itself
        config = ParallelismConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1  # 2 * 2 * 1 * 1 = 4
        )
        cluster_spec = ClusterSpec(total_gpus=8, gpus_per_node=8, gpu_memory_gb=80.0)

        result = validator.validate_configuration(config, cluster_spec)

        # This should trigger a warning about not using all GPUs
        assert len(result.warnings) > 0 or not result.is_valid


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_divisors_basic(self):
        """Test basic divisor calculation."""
        divisors = get_divisors(12)
        assert set(divisors) == {1, 2, 3, 4, 6, 12}
        assert divisors == sorted(divisors)

    def test_get_divisors_with_max(self):
        """Test divisor calculation with maximum."""
        divisors = get_divisors(12, max_divisor=4)
        assert set(divisors) == {1, 2, 3, 4}

    def test_get_divisors_prime(self):
        """Test divisor calculation for prime numbers."""
        divisors = get_divisors(7)
        assert set(divisors) == {1, 7}

    def test_get_divisors_edge_cases(self):
        """Test divisor calculation edge cases."""
        # n = 1
        divisors = get_divisors(1)
        assert divisors == [1]

        # max_divisor = 0 (should return empty)
        divisors = get_divisors(12, max_divisor=0)
        assert divisors == []

    def test_get_power_of_2_divisors(self):
        """Test power-of-2 divisor calculation."""
        # Number with power-of-2 divisors
        divisors = get_power_of_2_divisors(16)
        assert set(divisors) == {1, 2, 4, 8, 16}

        # Number with non-power-of-2 divisors
        divisors = get_power_of_2_divisors(12)
        assert set(divisors) == {1, 2, 4}  # Only power-of-2 divisors

    def test_validate_parallelism_combination_valid(self):
        """Test valid parallelism combination validation."""
        result = validate_parallelism_combination(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            data_parallel_size=2,
            total_gpus=8
        )

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_parallelism_combination_invalid_product(self):
        """Test invalid parallelism product."""
        result = validate_parallelism_combination(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            data_parallel_size=1,  # 2*2*1*1 = 4, but total_gpus = 8
            total_gpus=8
        )

        assert not result.is_valid
        assert any("multiply to" in error for error in result.errors)

    def test_validate_parallelism_combination_negative_values(self):
        """Test parallelism combination with negative values."""
        result = validate_parallelism_combination(
            tensor_parallel_size=-1,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            data_parallel_size=1,
            total_gpus=2
        )

        assert not result.is_valid
        assert any("must be positive" in error for error in result.errors)

    def test_validate_parallelism_combination_warnings(self):
        """Test parallelism combination warnings."""
        result = validate_parallelism_combination(
            tensor_parallel_size=16,  # High TP
            pipeline_parallel_size=8,  # High PP
            expert_parallel_size=1,
            data_parallel_size=1,
            total_gpus=128
        )

        # Should have warnings about high parallelism
        assert len(result.warnings) > 0
        assert any("High tensor parallelism" in warning for warning in result.warnings)
        assert any(
            "High pipeline parallelism" in warning for warning in result.warnings
        )


class TestValidationExceptions:
    """Tests for validation exception handling."""

    def test_tensor_parallel_validation_error(self):
        """Test TensorParallelValidationError raising."""
        with pytest.raises(TensorParallelValidationError):
            raise TensorParallelValidationError("Test error")

    def test_pipeline_parallel_validation_error(self):
        """Test PipelineParallelValidationError raising."""
        with pytest.raises(PipelineParallelValidationError):
            raise PipelineParallelValidationError("Test error")

    def test_expert_parallel_validation_error(self):
        """Test ExpertParallelValidationError raising."""
        with pytest.raises(ExpertParallelValidationError):
            raise ExpertParallelValidationError("Test error")

    def test_cross_constraint_validation_error(self):
        """Test CrossConstraintValidationError raising."""
        with pytest.raises(CrossConstraintValidationError):
            raise CrossConstraintValidationError("Test error")

    def test_cluster_resource_validation_error(self):
        """Test ClusterResourceValidationError raising."""
        with pytest.raises(ClusterResourceValidationError):
            raise ClusterResourceValidationError("Test error")


class TestTopologyValidation:
    """Tests for network topology constraint validation."""

    def test_tensor_parallel_exceeds_gpus_per_node(self):
        """Test tensor parallel size exceeding GPUs per node."""
        model_config = {
            "num_attention_heads": 32,
            "num_key_value_heads": 32
        }
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(tensor_parallel_size=8)
        cluster_spec = ClusterSpec(
            total_gpus=16,
            gpus_per_node=4,  # TP > GPUs per node
            gpu_memory_gb=80.0
        )

        result = validator.validate_configuration(config, cluster_spec)
        # Should have warnings about inter-node communication
        assert len(result.warnings) > 0
        topology_warnings = [w for w in result.warnings if "inter-node" in w.lower()]
        assert len(topology_warnings) > 0

    def test_pipeline_parallel_exceeds_nodes(self):
        """Test pipeline parallel size exceeding number of nodes."""
        model_config = {"num_hidden_layers": 32}
        validator = ConstraintValidator(model_config)

        config = ParallelismConfig(pipeline_parallel_size=8)
        cluster_spec = ClusterSpec(
            total_gpus=16,
            gpus_per_node=8,  # Only 2 nodes, but PP=8
            gpu_memory_gb=80.0
        )

        result = validator.validate_configuration(config, cluster_spec)
        assert not result.is_valid
        assert any("exceeds number of nodes" in error for error in result.errors)
