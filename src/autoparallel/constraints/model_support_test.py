"""Tests for model support utilities and constraint analysis."""

from unittest.mock import Mock

from transformers import PretrainedConfig

from autoparallel.constraints.model_support import (
    ArchitectureConstraints,
    ModelArchitectureInfo,
    ModelArchitectureType,
    QuantizationType,
    _calculate_expert_parallel_constraints,
    _calculate_pipeline_parallel_constraints,
    _calculate_tensor_parallel_constraints,
    _determine_vocab_divisibility_requirement,
    _extract_additional_features,
    _get_divisors,
    _get_efficient_divisors,
    _is_power_of_2_or_small_factors,
    calculate_architecture_constraints,
    detect_model_architecture,
    get_model_family_constraints,
    get_quantization_memory_multiplier,
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
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "mixtral"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "llama"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

        assert constraints.requires_tied_embeddings

    def test_missing_optional_parameters(self):
        """Test handling of missing optional configuration parameters."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.model_type = "gpt2"
        # Missing vocab_size and intermediate_size

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

        # Should use defaults
        assert constraints.vocab_divisibility_requirement >= 1
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
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "mixtral"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "llama"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

        # Test valid PP sizes
        valid_pp_sizes = constraints.get_valid_pipeline_parallel_sizes(max_stages=8)
        assert 1 in valid_pp_sizes
        assert all(48 / size >= 2 for size in valid_pp_sizes)  # Min 2 layers per stage

    def test_custom_constraint_parameters(self):
        """Test custom constraint parameters."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = 24
        config.vocab_size = 50257
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(
            arch_info, min_layers_per_stage=4
        )

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
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

        # Pipeline parallelism should be limited
        assert constraints.max_pipeline_parallel == 0  # 1 layer / 2 min per stage
        valid_pp_sizes = constraints.get_valid_pipeline_parallel_sizes(max_stages=8)
        assert valid_pp_sizes == []  # No valid PP sizes

    def test_tiny_model_architecture(self):
        """Test constraint analysis for very small models."""
        config = Mock(spec=PretrainedConfig)
        config.hidden_size = 32
        config.num_attention_heads = 2
        config.num_hidden_layers = 2
        config.vocab_size = 1000
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "mixtral"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "llama"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "gpt2"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

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
        config.model_type = "t5"

        arch_info = detect_model_architecture(config)
        constraints = calculate_architecture_constraints(arch_info)

        assert constraints.requires_tied_embeddings


class TestArchitectureSpecificParameterExtraction:
    """Test architecture-specific parameter extraction."""

    def test_tensor_parallel_parameter_extraction(self):
        """Test tensor parallel parameter extraction."""
        arch_info = ModelArchitectureInfo(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,
            num_layers=12,
            vocab_size=50257,
            intermediate_size=3072,
            architecture_type=ModelArchitectureType.DENSE_TRANSFORMER,
            model_type="gpt2",
        )

        result = _calculate_tensor_parallel_constraints(
            arch_info, max_tensor_parallel=64
        )

        assert "max_size" in result
        assert "valid_sizes" in result
        assert result["max_size"] >= 1
        assert 1 in result["valid_sizes"]

    def test_expert_parallel_parameter_extraction(self):
        """Test expert parallel parameter extraction."""
        # Non-MoE model
        arch_info = ModelArchitectureInfo(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_layers=32,
            vocab_size=32000,
            intermediate_size=14336,
            architecture_type=ModelArchitectureType.DENSE_TRANSFORMER,
            model_type="llama",
            num_experts=0,
        )

        result = _calculate_expert_parallel_constraints(
            arch_info, min_experts_per_device=1
        )

        assert result["max_size"] == 0
        assert result["valid_sizes"] == {1}

        # MoE model
        arch_info.num_experts = 8
        result = _calculate_expert_parallel_constraints(
            arch_info, min_experts_per_device=1
        )

        assert result["max_size"] == 8
        assert 8 in result["valid_sizes"]

    def test_pipeline_parallel_parameter_extraction(self):
        """Test pipeline parallel parameter extraction."""
        arch_info = ModelArchitectureInfo(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            num_layers=24,
            vocab_size=32000,
            intermediate_size=14336,
            architecture_type=ModelArchitectureType.DENSE_TRANSFORMER,
            model_type="llama",
        )

        result = _calculate_pipeline_parallel_constraints(
            arch_info, min_layers_per_stage=3
        )

        assert result["max_size"] == 8  # 24 layers / 3 min per stage
        assert result["min_layers_per_stage"] == 3

    def test_vocabulary_divisibility_requirements(self):
        """Test vocabulary divisibility requirement calculation."""
        # Small vocabulary
        small_req = _determine_vocab_divisibility_requirement(10000)
        assert small_req == 2

        # Medium vocabulary
        medium_req = _determine_vocab_divisibility_requirement(75000)
        assert medium_req == 4

        # Large vocabulary
        large_req = _determine_vocab_divisibility_requirement(150000)
        assert large_req == 8


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

    def test_is_power_of_2_or_small_factors(self):
        """Test efficient divisor detection."""
        # Powers of 2
        assert _is_power_of_2_or_small_factors(1)
        assert _is_power_of_2_or_small_factors(2)
        assert _is_power_of_2_or_small_factors(4)
        assert _is_power_of_2_or_small_factors(16)
        assert _is_power_of_2_or_small_factors(64)

        # Numbers with small prime factors (2, 3, 5, 7)
        assert _is_power_of_2_or_small_factors(6)  # 2 * 3
        assert _is_power_of_2_or_small_factors(10)  # 2 * 5
        assert _is_power_of_2_or_small_factors(14)  # 2 * 7
        assert _is_power_of_2_or_small_factors(21)  # 3 * 7

        # Numbers with large prime factors
        assert not _is_power_of_2_or_small_factors(11)  # Prime > 7
        assert not _is_power_of_2_or_small_factors(13)  # Prime > 7
        assert not _is_power_of_2_or_small_factors(33)  # 3 * 11

    def test_extract_additional_features(self):
        """Test additional features extraction."""
        config = Mock(spec=PretrainedConfig)
        config.tie_word_embeddings = True
        config.model_type = "llama"

        features = _extract_additional_features(config)
        assert features["has_tied_embeddings"]

        config.tie_word_embeddings = False
        features = _extract_additional_features(config)
        assert not features["has_tied_embeddings"]

        # Missing attribute should default to False
        del config.tie_word_embeddings
        features = _extract_additional_features(config)
        assert not features["has_tied_embeddings"]


class TestArchitectureConstraintsClass:
    """Test ArchitectureConstraints class methods."""

    def test_get_valid_tensor_parallel_sizes(self):
        """Test tensor parallel size validation."""
        constraints = ArchitectureConstraints(
            max_tensor_parallel=8,
            tensor_parallel_divisors={1, 2, 4, 8},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=12,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocab_divisibility_requirement=2,
            preferred_attention_head_divisors={1, 2, 4, 8},
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
        constraints = ArchitectureConstraints(
            max_tensor_parallel=8,
            tensor_parallel_divisors={1, 2, 4, 8},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=12,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocab_divisibility_requirement=2,
            preferred_attention_head_divisors={1, 2, 4, 8},
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
        constraints = ArchitectureConstraints(
            max_tensor_parallel=8,
            tensor_parallel_divisors={1, 2, 4, 8},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=12,  # 24 layers
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocab_divisibility_requirement=2,
            preferred_attention_head_divisors={1, 2, 4, 8},
        )

        # Should ensure at least min_layers_per_stage per stage
        expected_sizes = []
        for size in range(1, min(8, 12) + 1):
            if 12 / size >= 2:  # min_layers_per_stage
                expected_sizes.append(size)

        actual_sizes = constraints.get_valid_pipeline_parallel_sizes(max_stages=8)
        assert actual_sizes == expected_sizes


class TestModelFamilyConstraints:
    """Test model family specific constraints."""

    def test_get_model_family_constraints_llama(self):
        """Test LLaMA family constraints."""
        constraints = get_model_family_constraints("meta-llama/Llama-2-7b-hf")
        assert constraints["family"] == "llama"
        assert constraints["supports_flash_attention"]
        assert 1 in constraints["preferred_tp_sizes"]

    def test_get_model_family_constraints_mixtral(self):
        """Test Mixtral family constraints."""
        constraints = get_model_family_constraints("mistralai/Mixtral-8x7B-v0.1")
        assert constraints["family"] == "mixtral"
        assert constraints["supports_flash_attention"]
        assert "preferred_ep_sizes" in constraints

    def test_get_model_family_constraints_unknown(self):
        """Test unknown model family constraints."""
        constraints = get_model_family_constraints("unknown/model")
        assert constraints["family"] == "unknown"
        assert not constraints["supports_flash_attention"]


class TestQuantizationSupport:
    """Test quantization memory calculations."""

    def test_get_quantization_memory_multiplier(self):
        """Test quantization memory multipliers."""
        assert (
            get_quantization_memory_multiplier(QuantizationType.FULL_PRECISION) == 1.0
        )
        assert get_quantization_memory_multiplier(QuantizationType.GPTQ) == 0.25
        assert get_quantization_memory_multiplier(QuantizationType.AWQ) == 0.25
        assert get_quantization_memory_multiplier(QuantizationType.BITSANDBYTES) == 0.5
        assert get_quantization_memory_multiplier(QuantizationType.INT8) == 0.5
        assert get_quantization_memory_multiplier(QuantizationType.UNKNOWN) == 1.0
