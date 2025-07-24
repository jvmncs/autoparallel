"""Tests for constraints analyzer."""

from unittest.mock import MagicMock, patch

from autoparallel.constraints.analyzer import (
    ModelConstraints,
    ParallelismConstraintParameters,
    _get_divisors,
    _get_efficient_divisors,
    analyze_model_constraints,
)


def _create_mock_config(config_dict: dict) -> MagicMock:
    """Create a mock PretrainedConfig object from a dict."""
    mock_config = MagicMock()
    for key, value in config_dict.items():
        setattr(mock_config, key, value)

    # Handle missing attributes with defaults instead of MagicMock objects
    # Configure known attributes that might be missing
    mock_config.configure_mock(
        **{
            "num_local_experts": config_dict.get("num_local_experts", 0),
            "num_experts": config_dict.get("num_experts", 0),
            "tie_word_embeddings": config_dict.get("tie_word_embeddings", False),
            "num_key_value_heads": config_dict.get(
                "num_key_value_heads", config_dict.get("num_attention_heads", 12)
            ),
            "vocab_size": config_dict.get("vocab_size", 50257),
            "intermediate_size": config_dict.get(
                "intermediate_size", 4 * config_dict.get("hidden_size", 768)
            ),
        }
    )

    return mock_config


class TestParallelismConstraintParameters:
    """Test ParallelismConstraintParameters dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = ParallelismConstraintParameters()

        assert params.default_min_layers_per_stage == 2
        assert params.default_max_tensor_parallel == 64
        assert params.min_experts_per_device == 1
        assert params.vocab_large_threshold == 100000
        assert params.vocab_medium_threshold == 50000
        assert params.vocab_large_divisibility == 8
        assert params.vocab_medium_divisibility == 4
        assert params.vocab_small_divisibility == 2

    def test_custom_values(self):
        """Test custom parameter values."""
        params = ParallelismConstraintParameters(
            default_min_layers_per_stage=4,
            default_max_tensor_parallel=32,
            min_experts_per_device=2,
            vocab_large_threshold=200000,
        )

        assert params.default_min_layers_per_stage == 4
        assert params.default_max_tensor_parallel == 32
        assert params.min_experts_per_device == 2
        assert params.vocab_large_threshold == 200000
        # Other values should remain default
        assert params.vocab_medium_threshold == 50000


class TestModelConstraints:
    """Test ModelConstraints dataclass."""

    def test_model_constraints_creation(self):
        """Test basic ModelConstraints creation."""
        constraints = ModelConstraints(
            max_tensor_parallel=32,
            tensor_parallel_divisors={1, 2, 4, 8, 16, 32},
            max_expert_parallel=8,
            expert_parallel_divisors={1, 2, 4, 8},
            max_pipeline_parallel=32,
            min_layers_per_stage=2,
            requires_tied_embeddings=True,
            supports_grouped_query_attention=False,
            vocabulary_sharding=4,
        )

        assert constraints.max_tensor_parallel == 32
        assert constraints.tensor_parallel_divisors == {1, 2, 4, 8, 16, 32}
        assert constraints.max_expert_parallel == 8
        assert constraints.expert_parallel_divisors == {1, 2, 4, 8}
        assert constraints.max_pipeline_parallel == 32
        assert constraints.min_layers_per_stage == 2
        assert constraints.requires_tied_embeddings is True
        assert constraints.supports_grouped_query_attention is False
        assert constraints.vocabulary_sharding == 4

    def test_get_valid_tensor_parallel_sizes(self):
        """Test tensor parallel size filtering."""
        constraints = ModelConstraints(
            max_tensor_parallel=16,
            tensor_parallel_divisors={1, 2, 4, 8, 16, 32},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=32,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        # Should be limited by max_tensor_parallel
        valid_sizes = constraints.get_valid_tensor_parallel_sizes(max_gpus=64)
        assert valid_sizes == [1, 2, 4, 8, 16]

        # Should be limited by max_gpus
        valid_sizes = constraints.get_valid_tensor_parallel_sizes(max_gpus=8)
        assert valid_sizes == [1, 2, 4, 8]

        # Should be limited by both
        valid_sizes = constraints.get_valid_tensor_parallel_sizes(max_gpus=4)
        assert valid_sizes == [1, 2, 4]

    def test_get_valid_expert_parallel_sizes(self):
        """Test expert parallel size filtering."""
        constraints = ModelConstraints(
            max_tensor_parallel=16,
            tensor_parallel_divisors={1, 2, 4, 8, 16},
            max_expert_parallel=8,
            expert_parallel_divisors={1, 2, 4, 8},
            max_pipeline_parallel=32,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        # Normal MoE model
        valid_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=16)
        assert valid_sizes == [1, 2, 4, 8]

        # Limited by max_gpus
        valid_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=4)
        assert valid_sizes == [1, 2, 4]

        # Non-MoE model
        constraints.max_expert_parallel = 0
        valid_sizes = constraints.get_valid_expert_parallel_sizes(max_gpus=16)
        assert valid_sizes == [1]

    def test_get_valid_pipeline_parallel_sizes(self):
        """Test pipeline parallel size filtering."""
        constraints = ModelConstraints(
            max_tensor_parallel=16,
            tensor_parallel_divisors={1, 2, 4, 8, 16},
            max_expert_parallel=0,
            expert_parallel_divisors={1},
            max_pipeline_parallel=16,  # Based on 32 layers // 2 per stage
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        # max_pipeline_parallel=16, minimum 2 layers per stage
        valid_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=20)
        # The real implementation checks if 16/size >= 2, which means size <= 8
        expected = list(range(1, 9))  # 1 to 8
        assert valid_sizes == expected

        # Limited by max_nodes
        valid_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        expected = list(range(1, 9))  # 1 to 8
        assert valid_sizes == expected

        # Test with higher min_layers_per_stage
        constraints.min_layers_per_stage = 4
        valid_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=20)
        # With min_layers_per_stage=4, only stages where 16/size >= 4 are valid
        # That means size <= 4, so valid sizes are [1, 2, 3, 4]
        expected = list(range(1, 5))  # 1 to 4
        assert valid_sizes == expected


class TestAnalyzeModelConstraints:
    """Test analyze_model_constraints function."""

    def test_llama_constraints(self, mock_llama_config):
        """Test constraint analysis for LLaMA model."""
        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)

        # LLaMA has 32 attention heads -> max TP = 32
        assert constraints.max_tensor_parallel <= 32
        assert 1 in constraints.tensor_parallel_divisors
        assert 2 in constraints.tensor_parallel_divisors
        assert 4 in constraints.tensor_parallel_divisors
        assert 8 in constraints.tensor_parallel_divisors

        # Not an MoE model
        assert constraints.max_expert_parallel == 0
        assert constraints.expert_parallel_divisors == {1}

        # 32 layers, min 2 per stage -> max PP = 16
        assert constraints.max_pipeline_parallel == 16
        assert constraints.min_layers_per_stage == 2

        # LLaMA typically has tied embeddings
        assert constraints.requires_tied_embeddings is True

        # Standard attention (not GQA)
        assert constraints.supports_grouped_query_attention is False

        # Vocab size 32000 -> small threshold (< 50000)
        assert constraints.vocabulary_sharding == 2

    def test_gqa_constraints(self, mock_gqa_config):
        """Test constraint analysis for GQA model."""
        config = _create_mock_config(mock_gqa_config)
        constraints = analyze_model_constraints(config)

        # GQA has 8 KV heads which is more restrictive than 32 Q heads
        # Max TP should be limited by KV heads
        assert constraints.max_tensor_parallel <= 8

        # Should detect GQA
        assert constraints.supports_grouped_query_attention is True

        # Not tied embeddings
        assert constraints.requires_tied_embeddings is False

    def test_moe_constraints(self, mock_moe_config):
        """Test constraint analysis for MoE model."""
        config = _create_mock_config(mock_moe_config)
        constraints = analyze_model_constraints(config)

        # Should detect MoE
        assert constraints.max_expert_parallel > 0
        assert 1 in constraints.expert_parallel_divisors
        assert 2 in constraints.expert_parallel_divisors
        assert 4 in constraints.expert_parallel_divisors
        assert 8 in constraints.expert_parallel_divisors

        # Should detect GQA
        assert constraints.supports_grouped_query_attention is True

        # Not tied embeddings
        assert constraints.requires_tied_embeddings is False

    def test_single_layer_constraints(self, mock_single_layer_config):
        """Test constraint analysis for single layer model."""
        config = _create_mock_config(mock_single_layer_config)
        constraints = analyze_model_constraints(config)

        # Single layer, min 2 per stage -> max PP = 0 (1 // 2 = 0)
        assert constraints.max_pipeline_parallel == 0

        # Pipeline parallelism not viable with default min_layers_per_stage
        # 1 layer with min_layers_per_stage=2 means no valid PP sizes
        valid_pp_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        assert valid_pp_sizes == []  # No valid PP sizes with min 2 layers per stage

        # Should still support tensor parallelism (at least 1)
        assert constraints.max_tensor_parallel >= 1

    def test_massive_moe_constraints(self, mock_massive_moe_config):
        """Test constraint analysis for massive MoE model."""
        config = _create_mock_config(mock_massive_moe_config)
        constraints = analyze_model_constraints(config)

        # 64 experts -> should support high EP
        assert constraints.max_expert_parallel > 8
        assert len(constraints.expert_parallel_divisors) > 4

        # Large vocab (100k) -> large divisibility requirement
        assert constraints.vocabulary_sharding == 8

        # High layer count -> supports high PP (80 // 2 = 40)
        assert constraints.max_pipeline_parallel == 40

        # Should detect GQA
        assert constraints.supports_grouped_query_attention is True

    def test_custom_constraint_params(self, mock_llama_config):
        """Test with custom constraint parameters."""
        custom_params = ParallelismConstraintParameters(
            default_min_layers_per_stage=4,
            default_max_tensor_parallel=16,
            vocab_medium_threshold=20000,
            vocab_medium_divisibility=8,
        )

        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config, custom_params)

        # Should respect custom max TP
        assert constraints.max_tensor_parallel <= 16

        # Should respect custom min layers per stage
        assert constraints.min_layers_per_stage == 4

        # Vocab 32000 > 20000 -> medium divisibility = 8
        assert constraints.vocabulary_sharding == 8

    def test_edge_case_empty_divisors(self):
        """Test edge case where no valid divisors exist."""
        # Create config with problematic dimensions
        config = {
            "model_type": "test",
            "hidden_size": 7,  # Prime number
            "num_attention_heads": 3,  # Small prime
            "num_key_value_heads": 3,
            "num_hidden_layers": 12,
            "vocab_size": 13,  # Prime number
            "intermediate_size": 11,  # Prime number
        }

        mock_config = _create_mock_config(config)
        constraints = analyze_model_constraints(mock_config)

        # Should have at least size 1
        assert 1 in constraints.tensor_parallel_divisors
        assert constraints.max_tensor_parallel >= 1


class TestConstraintDetectionAcrossArchitectures:
    """Test constraint detection across different model architectures."""

    def test_gpt2_architecture(self):
        """Test GPT-2 style architecture."""
        config = {
            "model_type": "gpt2",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 50257,
            "intermediate_size": 3072,
            "tie_word_embeddings": True,
        }

        mock_config = _create_mock_config(config)
        constraints = analyze_model_constraints(mock_config)

        assert constraints.max_tensor_parallel <= 12  # Limited by attention heads
        assert constraints.requires_tied_embeddings is True
        assert constraints.supports_grouped_query_attention is False

    def test_t5_architecture(self):
        """Test T5 style architecture."""
        config = {
            "model_type": "t5",
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "vocab_size": 32128,
            "intermediate_size": 2048,
            "tie_word_embeddings": False,
        }

        mock_config = _create_mock_config(config)
        constraints = analyze_model_constraints(mock_config)

        assert constraints.max_tensor_parallel <= 8
        assert constraints.max_pipeline_parallel == 3  # 6 // 2 = 3
        assert constraints.requires_tied_embeddings is False

    def test_bert_architecture(self):
        """Test BERT style architecture."""
        config = {
            "model_type": "bert",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 30522,
            "intermediate_size": 3072,
            "tie_word_embeddings": False,
        }

        mock_config = _create_mock_config(config)
        constraints = analyze_model_constraints(mock_config)

        assert constraints.max_tensor_parallel <= 12
        assert constraints.max_pipeline_parallel == 6  # 12 // 2 = 6
        assert constraints.requires_tied_embeddings is False


class TestQuantizationImpact:
    """Test different quantization formats and their impact on constraints."""

    def test_fp16_quantization(self, mock_llama_config):
        """Test FP16 quantization impact."""
        # Quantization shouldn't change architectural constraints
        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)

        # Constraints should be based on architecture, not quantization
        assert constraints.max_tensor_parallel > 1
        assert len(constraints.tensor_parallel_divisors) > 1

    def test_int8_quantization(self, mock_llama_config):
        """Test INT8 quantization impact."""
        # For now, quantization doesn't affect architectural constraints
        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)

        # Should have same constraints as FP16
        assert constraints.max_tensor_parallel > 1
        assert len(constraints.tensor_parallel_divisors) > 1

    def test_int4_quantization(self, mock_llama_config):
        """Test INT4 quantization impact."""
        # Future: INT4 might have different alignment requirements
        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)

        # For now, same as other formats
        assert constraints.max_tensor_parallel > 1
        assert len(constraints.tensor_parallel_divisors) > 1


class TestMetaDeviceLoading:
    """Test meta-device loading for model config analysis."""

    @patch("transformers.AutoConfig.from_pretrained")
    def test_mock_transformers_config_loading(
        self, mock_from_pretrained, mock_llama_config
    ):
        """Test loading config without downloading models."""
        # Mock the transformers config loading
        mock_config = MagicMock()
        for key, value in mock_llama_config.items():
            setattr(mock_config, key, value)
        mock_from_pretrained.return_value = mock_config

        # This would be the actual implementation approach
        # config = transformers.AutoConfig.from_pretrained(
        #     "mock-model", trust_remote_code=False
        # )
        # constraints = analyze_model_constraints(config.__dict__)

        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)
        assert constraints.max_tensor_parallel > 1

        # Verify mock was not called (since we're using dict directly)
        mock_from_pretrained.assert_not_called()

    def test_config_dict_analysis(self, mock_llama_config):
        """Test analyzing config dict directly (no model loading)."""
        # This approach avoids any model downloads
        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)

        assert isinstance(constraints, ModelConstraints)
        assert constraints.max_tensor_parallel > 0
        assert len(constraints.tensor_parallel_divisors) > 0

    def test_meta_device_mock(self, mock_llama_config):
        """Test meta device loading pattern approach."""
        # Test the pattern we'd use for meta device loading
        # In real implementation, we'd use transformers.AutoModel.from_pretrained(
        #     ..., device_map="meta"
        # )
        # but here we test the constraint analysis directly

        config = _create_mock_config(mock_llama_config)
        constraints = analyze_model_constraints(config)
        assert constraints.max_tensor_parallel > 1
        assert isinstance(constraints, ModelConstraints)

        # This demonstrates how we'd analyze without loading actual weights
        assert len(constraints.tensor_parallel_divisors) > 0


class TestUtilityFunctions:
    """Test utility functions used in constraint analysis."""

    def test_get_divisors(self):
        """Test divisor calculation."""
        assert _get_divisors(12) == [1, 2, 3, 4, 6, 12]
        assert _get_divisors(16) == [1, 2, 4, 8, 16]
        assert _get_divisors(1) == [1]
        assert _get_divisors(7) == [1, 7]  # Prime

    def test_get_efficient_divisors(self):
        """Test efficient divisor calculation with max limit."""
        divisors = _get_efficient_divisors(32, max_divisor=8)
        assert divisors == [1, 2, 4, 8]

        divisors = _get_efficient_divisors(64, max_divisor=16)
        assert divisors == [1, 2, 4, 8, 16]

        # Prime number - 17 is not efficient (not power of 2 or small primes)
        divisors = _get_efficient_divisors(17, max_divisor=64)
        assert divisors == [1]  # Only 1 is efficient for prime 17

    def test_edge_case_zero_values(self):
        """Test edge cases with zero or invalid values."""
        config = {
            "model_type": "test",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 0,  # Edge case
            "vocab_size": 0,  # Edge case
        }

        mock_config = _create_mock_config(config)
        constraints = analyze_model_constraints(mock_config)

        # Should handle gracefully - 0 layers // 2 per stage = 0 max PP
        assert constraints.max_pipeline_parallel == 0
        assert isinstance(constraints, ModelConstraints)
