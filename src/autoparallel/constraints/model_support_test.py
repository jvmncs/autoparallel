"""Tests for model support utilities and constraint analysis."""

from unittest.mock import Mock

from transformers import PretrainedConfig

from .analyzer import (
    ModelConstraints,
    ParallelismConstraintParameters,
    _analyze_expert_parallel_constraints,
    _analyze_pipeline_parallel_constraints,
    _analyze_tensor_parallel_constraints,
    _check_tied_embeddings,
    _determine_vocab_divisibility_requirement,
    _get_divisors,
    _get_efficient_divisors,
    _is_efficient_divisor,
    analyze_model_constraints,
)


class TestModelArchitectureDetection:
    """Test model architecture detection and parameter extraction."""

    def test_standard_transformer_detection(self):
        """Test detection of standard transformer architecture parameters."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.vocab_size = 50257
        config.intermediate_size = 3072

        constraints = analyze_model_constraints(config)

        assert constraints.max_tensor_parallel <= 12  # Limited by attention heads
        assert 1 in constraints.tensor_parallel_divisors
        assert constraints.max_expert_parallel == 0  # Not MoE
        assert constraints.max_pipeline_parallel == 6  # 12 layers / 2 min per stage
        assert not constraints.supports_grouped_query_attention

    def test_moe_model_detection(self):
        """Test detection of MoE (Mixtral-style) model parameters."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_hidden_layers = 32
        config.vocab_size = 32000
        config.intermediate_size = 14336
        config.num_local_experts = 8  # Mixtral-style MoE

        constraints = analyze_model_constraints(config)

        assert constraints.max_expert_parallel == 8
        assert 1 in constraints.expert_parallel_divisors
        assert 2 in constraints.expert_parallel_divisors
        assert 4 in constraints.expert_parallel_divisors
        assert 8 in constraints.expert_parallel_divisors

    def test_gqa_model_detection(self):
        """Test detection of Grouped Query Attention (GQA) models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_key_value_heads = 8  # GQA: fewer KV heads
        config.num_hidden_layers = 32
        config.vocab_size = 32000
        config.intermediate_size = 11008

        constraints = analyze_model_constraints(config)

        assert constraints.supports_grouped_query_attention
        assert constraints.max_tensor_parallel <= 8  # Limited by KV heads

    def test_tied_embeddings_detection(self):
        """Test detection of tied embeddings."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.vocab_size = 50257
        config.tie_word_embeddings = True

        constraints = analyze_model_constraints(config)

        assert constraints.requires_tied_embeddings

    def test_missing_optional_parameters(self):
        """Test handling of missing optional configuration parameters."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        # Missing vocab_size and intermediate_size

        constraints = analyze_model_constraints(config)

        # Should use defaults
        assert constraints.vocabulary_sharding >= 1
        assert constraints.max_tensor_parallel >= 1


class TestConstraintValidation:
    """Test constraint validation across different architectures."""

    def test_tensor_parallel_constraints_standard_model(self):
        """Test tensor parallel constraints for standard models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.vocab_size = 50257
        config.intermediate_size = 3072

        constraints = analyze_model_constraints(config)

        # Test valid TP sizes
        valid_tp_sizes = constraints.get_valid_tensor_parallel_sizes(max_gpus=8)
        assert 1 in valid_tp_sizes
        assert all(size <= 8 for size in valid_tp_sizes)
        assert all(12 % size == 0 for size in valid_tp_sizes if size <= 12)

    def test_expert_parallel_constraints_moe_model(self):
        """Test expert parallel constraints for MoE models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_hidden_layers = 32
        config.vocab_size = 32000
        config.num_local_experts = 8

        constraints = analyze_model_constraints(config)

        # Test valid EP sizes
        valid_ep_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=16)
        assert 1 in valid_ep_sizes
        assert 8 in valid_ep_sizes
        assert all(8 % size == 0 for size in valid_ep_sizes)

    def test_pipeline_parallel_constraints(self):
        """Test pipeline parallel constraints."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_hidden_layers = 48  # Large model
        config.vocab_size = 32000

        constraints = analyze_model_constraints(config)

        # Test valid PP sizes
        valid_pp_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        assert 1 in valid_pp_sizes
        assert all(48 / size >= 2 for size in valid_pp_sizes)  # Min 2 layers per stage

    def test_custom_constraint_parameters(self):
        """Test custom constraint parameters."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 24
        config.vocab_size = 50257

        custom_params = ParallelismConstraintParameters(
            default_min_layers_per_stage=4,  # Higher minimum
            default_max_tensor_parallel=8,  # Lower maximum
        )

        constraints = analyze_model_constraints(config, custom_params)

        assert constraints.min_layers_per_stage == 4
        assert constraints.max_pipeline_parallel == 6  # 24 layers / 4 min per stage


class TestEdgeCases:
    """Test edge cases like single layer models and massive MoE."""

    def test_single_layer_model(self):
        """Test constraint analysis for single layer models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 1  # Single layer
        config.vocab_size = 50257

        constraints = analyze_model_constraints(config)

        # Pipeline parallelism should be limited
        assert constraints.max_pipeline_parallel == 0  # 1 layer / 2 min per stage
        valid_pp_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        assert valid_pp_sizes == []  # No valid PP sizes

    def test_tiny_model_architecture(self):
        """Test constraint analysis for very small models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 32
        config.num_attention_heads = 2
        config.num_hidden_layers = 2
        config.vocab_size = 1000

        constraints = analyze_model_constraints(config)

        assert constraints.max_tensor_parallel <= 2
        assert 1 in constraints.tensor_parallel_divisors
        assert 2 in constraints.tensor_parallel_divisors

    def test_massive_moe_model(self):
        """Test constraint analysis for massive MoE models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 8192
        config.num_attention_heads = 64
        config.num_hidden_layers = 64
        config.vocab_size = 100000
        config.num_local_experts = 64  # Large number of experts

        constraints = analyze_model_constraints(config)

        assert constraints.max_expert_parallel == 64
        valid_ep_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=128)
        assert 64 in valid_ep_sizes
        assert all(64 % size == 0 for size in valid_ep_sizes)

    def test_irregular_architecture_dimensions(self):
        """Test models with irregular dimension sizes."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 1337  # Prime number
        config.num_attention_heads = 17  # Prime number
        config.num_hidden_layers = 23  # Prime number
        config.vocab_size = 30001  # Prime number

        constraints = analyze_model_constraints(config)

        # Should still work, but with limited parallelization
        assert constraints.max_tensor_parallel >= 1
        assert 1 in constraints.tensor_parallel_divisors


class TestModelFamilyDetection:
    """Test model family detection and constraint mapping."""

    def test_llama_style_architecture(self):
        """Test detection of LLaMA-style architecture constraints."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_key_value_heads = 32  # Multi-head attention
        config.num_hidden_layers = 32
        config.vocab_size = 32000
        config.intermediate_size = 11008
        config.tie_word_embeddings = False

        constraints = analyze_model_constraints(config)

        assert not constraints.supports_grouped_query_attention
        assert not constraints.requires_tied_embeddings
        assert constraints.max_tensor_parallel <= 32

    def test_gpt_style_architecture(self):
        """Test detection of GPT-style architecture constraints."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 1600
        config.num_attention_heads = 25
        config.num_hidden_layers = 48
        config.vocab_size = 50257
        config.intermediate_size = 6400
        config.tie_word_embeddings = True

        constraints = analyze_model_constraints(config)

        assert constraints.requires_tied_embeddings
        assert not constraints.supports_grouped_query_attention

    def test_t5_style_architecture(self):
        """Test detection of T5-style architecture constraints."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.vocab_size = 32128
        config.intermediate_size = 2048
        config.tie_word_embeddings = True

        constraints = analyze_model_constraints(config)

        assert constraints.requires_tied_embeddings


class TestArchitectureSpecificParameterExtraction:
    """Test architecture-specific parameter extraction."""

    def test_tensor_parallel_parameter_extraction(self):
        """Test tensor parallel parameter extraction."""
        result = _analyze_tensor_parallel_constraints(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,
            vocab_size=50257,
            intermediate_size=3072,
            constraint_params=ParallelismConstraintParameters(),
        )

        assert "max_size" in result
        assert "valid_sizes" in result
        assert result["max_size"] >= 1
        assert 1 in result["valid_sizes"]

    def test_expert_parallel_parameter_extraction(self):
        """Test expert parallel parameter extraction."""
        # Non-MoE model
        config = Mock(spec=PretrainedConfig)
        # No num_local_experts or num_experts attribute

        result = _analyze_expert_parallel_constraints(
            config, ParallelismConstraintParameters()
        )

        assert result["max_size"] == 0
        assert result["valid_sizes"] == {1}

        # MoE model
        config.num_local_experts = 8
        result = _analyze_expert_parallel_constraints(
            config, ParallelismConstraintParameters()
        )

        assert result["max_size"] == 8
        assert 8 in result["valid_sizes"]

    def test_pipeline_parallel_parameter_extraction(self):
        """Test pipeline parallel parameter extraction."""
        result = _analyze_pipeline_parallel_constraints(
            num_layers=24,
            constraint_params=ParallelismConstraintParameters(
                default_min_layers_per_stage=3
            ),
        )

        assert result["max_size"] == 8  # 24 layers / 3 min per stage
        assert result["min_layers_per_stage"] == 3

    def test_vocabulary_divisibility_requirements(self):
        """Test vocabulary divisibility requirement calculation."""
        params = ParallelismConstraintParameters()

        # Small vocabulary
        small_req = _determine_vocab_divisibility_requirement(10000, params)
        assert small_req == params.vocab_small_divisibility

        # Medium vocabulary
        medium_req = _determine_vocab_divisibility_requirement(75000, params)
        assert medium_req == params.vocab_medium_divisibility

        # Large vocabulary
        large_req = _determine_vocab_divisibility_requirement(150000, params)
        assert large_req == params.vocab_large_divisibility


class TestUtilityFunctions:
    """Test utility functions for constraint analysis."""

    def test_get_divisors(self):
        """Test divisor calculation."""
        assert _get_divisors(12) == [1, 2, 3, 4, 6, 12]
        assert _get_divisors(8) == [1, 2, 4, 8]
        assert _get_divisors(7) == [1, 7]  # Prime number
        assert _get_divisors(12, max_divisor=6) == [1, 2, 3, 4, 6]

    def test_get_efficient_divisors(self):
        """Test efficient divisor calculation."""
        # Powers of 2 should be included
        divisors = _get_efficient_divisors(16)
        assert 1 in divisors
        assert 2 in divisors
        assert 4 in divisors
        assert 8 in divisors
        assert 16 in divisors

        # Numbers with small prime factors
        divisors = _get_efficient_divisors(12)
        assert 1 in divisors
        assert 2 in divisors
        assert 3 in divisors
        assert 4 in divisors
        assert 6 in divisors
        assert 12 in divisors

    def test_is_efficient_divisor(self):
        """Test efficient divisor detection."""
        # Powers of 2
        assert _is_efficient_divisor(1)
        assert _is_efficient_divisor(2)
        assert _is_efficient_divisor(4)
        assert _is_efficient_divisor(16)
        assert _is_efficient_divisor(64)

        # Numbers with small prime factors (2, 3, 5, 7)
        assert _is_efficient_divisor(6)  # 2 * 3
        assert _is_efficient_divisor(10)  # 2 * 5
        assert _is_efficient_divisor(14)  # 2 * 7
        assert _is_efficient_divisor(21)  # 3 * 7

        # Numbers with large prime factors
        assert not _is_efficient_divisor(11)  # Prime > 7
        assert not _is_efficient_divisor(13)  # Prime > 7
        assert not _is_efficient_divisor(33)  # 3 * 11

    def test_check_tied_embeddings(self):
        """Test tied embeddings detection."""
        config = Mock(spec=PretrainedConfig)
        config.tie_word_embeddings = True
        assert _check_tied_embeddings(config)

        config.tie_word_embeddings = False
        assert not _check_tied_embeddings(config)

        # Missing attribute should default to False
        del config.tie_word_embeddings
        assert not _check_tied_embeddings(config)


class TestModelConstraintsClass:
    """Test ModelConstraints class methods."""

    def test_get_valid_tensor_parallel_sizes(self):
        """Test tensor parallel size validation."""
        constraints = ModelConstraints(
            max_tensor_parallel=8,
            tensor_parallel_divisors={1, 2, 4, 8},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=12,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        # Should return sizes up to max_gpus limit
        valid_sizes = constraints.get_valid_tensor_parallel_sizes(max_gpus=4)
        assert valid_sizes == [1, 2, 4]

        # Should respect model's max_tensor_parallel
        valid_sizes = constraints.get_valid_tensor_parallel_sizes(max_gpus=16)
        assert valid_sizes == [1, 2, 4, 8]

    def test_get_valid_expert_parallel_sizes(self):
        """Test expert parallel size validation."""
        # Non-MoE model
        constraints = ModelConstraints(
            max_tensor_parallel=8,
            tensor_parallel_divisors={1, 2, 4, 8},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=12,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        valid_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=8)
        assert valid_sizes == [1]

        # MoE model
        constraints.max_expert_parallel = 8
        constraints.expert_parallel_divisors = {1, 2, 4, 8}

        valid_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=4)
        assert valid_sizes == [1, 2, 4]

    def test_get_valid_pipeline_parallel_sizes(self):
        """Test pipeline parallel size validation."""
        constraints = ModelConstraints(
            max_tensor_parallel=8,
            tensor_parallel_divisors={1, 2, 4, 8},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=12,  # 24 layers
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        # Should ensure at least min_layers_per_stage per stage
        expected_sizes = []
        for size in range(1, min(8, 12) + 1):
            if 12 / size >= 2:  # min_layers_per_stage
                expected_sizes.append(size)

        actual_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        assert actual_sizes == expected_sizes
