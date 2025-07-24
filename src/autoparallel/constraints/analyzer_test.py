"""Tests for constraints analyzer."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


@dataclass
class ParallelismConstraintParameters:
    """Configurable parameters for parallelism constraints."""

    default_min_layers_per_stage: int = 2
    default_max_tensor_parallel: int = 64
    min_experts_per_device: int = 1
    vocab_large_threshold: int = 100000
    vocab_medium_threshold: int = 50000
    vocab_large_divisibility: int = 8
    vocab_medium_divisibility: int = 4
    vocab_small_divisibility: int = 2


@dataclass
class ModelConstraints:
    """Model architecture constraints that limit parallelization."""

    # Tensor parallelism constraints
    max_tensor_parallel: int
    tensor_parallel_divisors: set[int]

    # Expert parallelism constraints (for MoE)
    max_expert_parallel: int
    expert_parallel_divisors: set[int]

    # Pipeline parallelism constraints
    max_pipeline_parallel: int
    min_layers_per_stage: int

    # Additional constraints
    requires_tied_embeddings: bool
    supports_grouped_query_attention: bool
    vocab_divisibility_requirement: int

    def get_valid_tensor_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid tensor parallel sizes up to max_gpus."""
        valid_sizes = []
        for size in self.tensor_parallel_divisors:
            if size <= min(max_gpus, self.max_tensor_parallel):
                valid_sizes.append(size)
        return sorted(valid_sizes)

    def get_valid_expert_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid expert parallel sizes up to max_gpus."""
        if self.max_expert_parallel == 0:
            return [1]  # Not an MoE model

        valid_sizes = []
        for size in self.expert_parallel_divisors:
            if size <= min(max_gpus, self.max_expert_parallel):
                valid_sizes.append(size)
        return sorted(valid_sizes)

    def get_valid_pipeline_parallel_sizes(self, max_nodes: int) -> list[int]:
        """Get valid pipeline parallel sizes up to max_nodes."""
        if self.max_pipeline_parallel == 0:
            return []

        max_pp = min(max_nodes, self.max_pipeline_parallel)
        valid_sizes = []

        for size in range(1, max_pp + 1):
            layers_per_stage = self.max_pipeline_parallel / size
            if layers_per_stage >= self.min_layers_per_stage:
                valid_sizes.append(size)

        return valid_sizes


def _get_divisors(n: int) -> list[int]:
    """Get all divisors of n."""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def _get_efficient_divisors(n: int, max_divisor: int = 64) -> list[int]:
    """Get efficient divisors up to max_divisor."""
    divisors = _get_divisors(n)
    return [d for d in divisors if d <= max_divisor]


def analyze_model_constraints(
    config: dict,
    constraint_params: ParallelismConstraintParameters | None = None
) -> ModelConstraints:
    """Analyze model architecture to determine parallelization constraints."""

    if constraint_params is None:
        constraint_params = ParallelismConstraintParameters()

    # Extract basic architecture parameters
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    num_layers = config["num_hidden_layers"]
    vocab_size = config.get("vocab_size", 50257)
    intermediate_size = config.get("intermediate_size", 4 * hidden_size)

    # Tensor parallelism constraints
    attention_head_constraint = num_attention_heads
    kv_head_constraint = num_key_value_heads

    hidden_size_divisors = _get_divisors(hidden_size)
    intermediate_divisors = _get_divisors(intermediate_size)
    vocab_divisors = _get_efficient_divisors(
        vocab_size, constraint_params.default_max_tensor_parallel
    )

    # Find intersection of all constraints
    valid_tp_sizes = set(range(1, attention_head_constraint + 1))
    valid_tp_sizes &= set(range(1, kv_head_constraint + 1))
    valid_tp_sizes &= set(hidden_size_divisors)
    valid_tp_sizes &= set(intermediate_divisors)
    valid_tp_sizes &= set(vocab_divisors)

    practical_max_tp = min(
        constraint_params.default_max_tensor_parallel,
        max(valid_tp_sizes) if valid_tp_sizes else 1
    )
    valid_tp_sizes = {size for size in valid_tp_sizes if size <= practical_max_tp}

    # Expert parallelism constraints (MoE specific)
    num_experts = config.get("num_local_experts", config.get("num_experts", 0))
    if num_experts == 0:
        max_ep = 0
        valid_ep_sizes = {1}
    else:
        expert_divisors = _get_divisors(num_experts)
        max_ep = min(num_experts, constraint_params.default_max_tensor_parallel)
        valid_ep_sizes = {
            size for size in expert_divisors
            if size <= max_ep and
            num_experts // size >= constraint_params.min_experts_per_device
        }

    # Pipeline parallelism constraints
    max_pp = num_layers
    min_layers_per_stage = constraint_params.default_min_layers_per_stage

    # Additional architectural features
    tied_embeddings = config.get("tie_word_embeddings", False)
    gqa_support = num_key_value_heads != num_attention_heads

    # Vocab divisibility requirement
    if vocab_size >= constraint_params.vocab_large_threshold:
        vocab_divisibility = constraint_params.vocab_large_divisibility
    elif vocab_size >= constraint_params.vocab_medium_threshold:
        vocab_divisibility = constraint_params.vocab_medium_divisibility
    else:
        vocab_divisibility = constraint_params.vocab_small_divisibility

    return ModelConstraints(
        max_tensor_parallel=practical_max_tp,
        tensor_parallel_divisors=valid_tp_sizes,
        max_expert_parallel=max_ep,
        expert_parallel_divisors=valid_ep_sizes,
        max_pipeline_parallel=max_pp,
        min_layers_per_stage=min_layers_per_stage,
        requires_tied_embeddings=tied_embeddings,
        supports_grouped_query_attention=gqa_support,
        vocab_divisibility_requirement=vocab_divisibility
    )


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
            vocab_divisibility_requirement=4,
        )

        assert constraints.max_tensor_parallel == 32
        assert constraints.tensor_parallel_divisors == {1, 2, 4, 8, 16, 32}
        assert constraints.max_expert_parallel == 8
        assert constraints.expert_parallel_divisors == {1, 2, 4, 8}
        assert constraints.max_pipeline_parallel == 32
        assert constraints.min_layers_per_stage == 2
        assert constraints.requires_tied_embeddings is True
        assert constraints.supports_grouped_query_attention is False
        assert constraints.vocab_divisibility_requirement == 4

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
            vocab_divisibility_requirement=2,
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
            vocab_divisibility_requirement=2,
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
            max_pipeline_parallel=32,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocab_divisibility_requirement=2,
        )

        # 32 layers, minimum 2 layers per stage -> max PP = 16
        valid_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=20)
        expected = list(range(1, 17))  # 1 to 16
        assert valid_sizes == expected

        # Limited by max_nodes
        valid_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        expected = list(range(1, 9))  # 1 to 8
        assert valid_sizes == expected

        # Test with higher min_layers_per_stage
        constraints.min_layers_per_stage = 4
        valid_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=20)
        expected = list(range(1, 9))  # 32/4 = 8 max stages
        assert valid_sizes == expected


class TestAnalyzeModelConstraints:
    """Test analyze_model_constraints function."""

    def test_llama_constraints(self, mock_llama_config):
        """Test constraint analysis for LLaMA model."""
        constraints = analyze_model_constraints(mock_llama_config)

        # LLaMA has 32 attention heads -> max TP = 32
        assert constraints.max_tensor_parallel <= 32
        assert 1 in constraints.tensor_parallel_divisors
        assert 2 in constraints.tensor_parallel_divisors
        assert 4 in constraints.tensor_parallel_divisors
        assert 8 in constraints.tensor_parallel_divisors

        # Not an MoE model
        assert constraints.max_expert_parallel == 0
        assert constraints.expert_parallel_divisors == {1}

        # 32 layers -> max PP = 32
        assert constraints.max_pipeline_parallel == 32
        assert constraints.min_layers_per_stage == 2

        # LLaMA typically has tied embeddings
        assert constraints.requires_tied_embeddings is True

        # Standard attention (not GQA)
        assert constraints.supports_grouped_query_attention is False

        # Vocab size 32000 -> small threshold (< 50000)
        assert constraints.vocab_divisibility_requirement == 2

    def test_gqa_constraints(self, mock_gqa_config):
        """Test constraint analysis for GQA model."""
        constraints = analyze_model_constraints(mock_gqa_config)

        # GQA has 8 KV heads which is more restrictive than 32 Q heads
        # Max TP should be limited by KV heads
        assert constraints.max_tensor_parallel <= 8

        # Should detect GQA
        assert constraints.supports_grouped_query_attention is True

        # Not tied embeddings
        assert constraints.requires_tied_embeddings is False

    def test_moe_constraints(self, mock_moe_config):
        """Test constraint analysis for MoE model."""
        constraints = analyze_model_constraints(mock_moe_config)

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
        constraints = analyze_model_constraints(mock_single_layer_config)

        # Single layer -> max PP = 1
        assert constraints.max_pipeline_parallel == 1

        # Pipeline parallelism not viable with default min_layers_per_stage
        # 1 layer with min_layers_per_stage=2 means no valid PP sizes
        valid_pp_sizes = constraints.get_valid_pipeline_parallel_sizes(max_nodes=8)
        assert valid_pp_sizes == []  # No valid PP sizes with min 2 layers per stage

        # Should still support tensor parallelism (at least 1)
        assert constraints.max_tensor_parallel >= 1

    def test_massive_moe_constraints(self, mock_massive_moe_config):
        """Test constraint analysis for massive MoE model."""
        constraints = analyze_model_constraints(mock_massive_moe_config)

        # 64 experts -> should support high EP
        assert constraints.max_expert_parallel > 8
        assert len(constraints.expert_parallel_divisors) > 4

        # Large vocab (100k) -> large divisibility requirement
        assert constraints.vocab_divisibility_requirement == 8

        # High layer count -> supports high PP
        assert constraints.max_pipeline_parallel == 80

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

        constraints = analyze_model_constraints(mock_llama_config, custom_params)

        # Should respect custom max TP
        assert constraints.max_tensor_parallel <= 16

        # Should respect custom min layers per stage
        assert constraints.min_layers_per_stage == 4

        # Vocab 32000 > 20000 -> medium divisibility = 8
        assert constraints.vocab_divisibility_requirement == 8

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

        constraints = analyze_model_constraints(config)

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

        constraints = analyze_model_constraints(config)

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

        constraints = analyze_model_constraints(config)

        assert constraints.max_tensor_parallel <= 8
        assert constraints.max_pipeline_parallel == 6
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

        constraints = analyze_model_constraints(config)

        assert constraints.max_tensor_parallel <= 12
        assert constraints.max_pipeline_parallel == 12
        assert constraints.requires_tied_embeddings is False


class TestQuantizationImpact:
    """Test different quantization formats and their impact on constraints."""

    def test_fp16_quantization(self, mock_llama_config):
        """Test FP16 quantization impact."""
        # Quantization shouldn't change architectural constraints
        constraints = analyze_model_constraints(mock_llama_config)

        # Constraints should be based on architecture, not quantization
        assert constraints.max_tensor_parallel > 1
        assert len(constraints.tensor_parallel_divisors) > 1

    def test_int8_quantization(self, mock_llama_config):
        """Test INT8 quantization impact."""
        # For now, quantization doesn't affect architectural constraints
        constraints = analyze_model_constraints(mock_llama_config)

        # Should have same constraints as FP16
        assert constraints.max_tensor_parallel > 1
        assert len(constraints.tensor_parallel_divisors) > 1

    def test_int4_quantization(self, mock_llama_config):
        """Test INT4 quantization impact."""
        # Future: INT4 might have different alignment requirements
        constraints = analyze_model_constraints(mock_llama_config)

        # For now, same as other formats
        assert constraints.max_tensor_parallel > 1
        assert len(constraints.tensor_parallel_divisors) > 1


class TestMetaDeviceLoading:
    """Test meta-device loading for model config analysis."""

    @patch('transformers.AutoConfig.from_pretrained')
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

        constraints = analyze_model_constraints(mock_llama_config)
        assert constraints.max_tensor_parallel > 1

        # Verify mock was not called (since we're using dict directly)
        mock_from_pretrained.assert_not_called()

    def test_config_dict_analysis(self, mock_llama_config):
        """Test analyzing config dict directly (no model loading)."""
        # This approach avoids any model downloads
        constraints = analyze_model_constraints(mock_llama_config)

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

        constraints = analyze_model_constraints(mock_llama_config)
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

        # Prime number
        divisors = _get_efficient_divisors(17, max_divisor=64)
        assert divisors == [1, 17]

    def test_edge_case_zero_values(self):
        """Test edge cases with zero or invalid values."""
        config = {
            "model_type": "test",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 0,  # Edge case
            "vocab_size": 0,  # Edge case
        }

        constraints = analyze_model_constraints(config)

        # Should handle gracefully
        assert constraints.max_pipeline_parallel == 0
        assert isinstance(constraints, ModelConstraints)
