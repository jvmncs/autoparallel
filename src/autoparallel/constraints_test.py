"""Tests for simplified constraints analysis module."""

from autoparallel.constraints import (
    check_expert_parallel_efficiency,
    check_pipeline_parallel_efficiency,
    check_tensor_parallel_efficiency,
    get_divisors,
    get_model_architecture_info,
    is_power_of_2,
    valid_expert_parallel_sizes,
    valid_pipeline_parallel_sizes,
    valid_tensor_parallel_sizes,
    validate_parallelism_config,
)


class MockConfig(dict):
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_valid_tensor_parallel_sizes():
    """Test tensor parallel size validation."""
    # Standard model (32 attention heads)
    config = MockConfig(num_attention_heads=32, num_key_value_heads=32)
    valid_sizes = valid_tensor_parallel_sizes(config, 16)

    assert 1 in valid_sizes
    assert 2 in valid_sizes
    assert 4 in valid_sizes
    assert 8 in valid_sizes
    assert 16 in valid_sizes
    assert 32 not in valid_sizes  # max_size limit

    # GQA model (32 heads, 8 KV heads)
    gqa_config = MockConfig(num_attention_heads=32, num_key_value_heads=8)
    gqa_valid = valid_tensor_parallel_sizes(gqa_config, 16)

    assert 1 in gqa_valid
    assert 2 in gqa_valid
    assert 4 in gqa_valid
    assert 8 in gqa_valid
    assert 16 not in gqa_valid  # KV heads constraint (8 % 16 != 0)


def test_valid_pipeline_parallel_sizes():
    """Test pipeline parallel size validation."""
    # 32 layer model
    config = MockConfig(num_hidden_layers=32)
    valid_sizes = valid_pipeline_parallel_sizes(config, 20)

    assert 1 in valid_sizes
    assert 2 in valid_sizes
    assert 4 in valid_sizes
    assert 8 in valid_sizes
    assert 16 in valid_sizes
    assert 32 not in valid_sizes  # Would result in 1 layer per stage (< min 2)

    # Small model with 6 layers
    small_config = MockConfig(num_hidden_layers=6)
    small_valid = valid_pipeline_parallel_sizes(small_config, 10)

    assert 1 in small_valid
    assert 2 in small_valid
    assert 3 in small_valid
    assert 4 not in small_valid  # Would result in 1.5 layers per stage


def test_valid_expert_parallel_sizes():
    """Test expert parallel size validation."""
    # Dense model (no experts)
    dense_config = MockConfig(num_experts=0)
    dense_valid = valid_expert_parallel_sizes(dense_config, 8)

    assert dense_valid == [1]

    # MoE model with 8 experts
    moe_config = MockConfig(num_local_experts=8)
    moe_valid = valid_expert_parallel_sizes(moe_config, 16)

    assert 1 in moe_valid
    assert 2 in moe_valid
    assert 4 in moe_valid
    assert 8 in moe_valid
    assert 16 not in moe_valid  # Exceeds number of experts


def test_get_divisors():
    """Test divisor calculation."""
    divisors_12 = get_divisors(12)
    assert divisors_12 == [1, 2, 3, 4, 6, 12]

    divisors_12_limited = get_divisors(12, max_divisor=6)
    assert divisors_12_limited == [1, 2, 3, 4, 6]

    divisors_16 = get_divisors(16)
    assert divisors_16 == [1, 2, 4, 8, 16]


def test_is_power_of_2():
    """Test power of 2 detection."""
    assert is_power_of_2(1) is True
    assert is_power_of_2(2) is True
    assert is_power_of_2(4) is True
    assert is_power_of_2(8) is True
    assert is_power_of_2(16) is True

    assert is_power_of_2(3) is False
    assert is_power_of_2(6) is False
    assert is_power_of_2(12) is False
    assert is_power_of_2(0) is False


def test_validate_parallelism_config():
    """Test parallelism configuration validation."""
    # Valid configuration
    is_valid, errors = validate_parallelism_config(2, 2, 1, 2, 8)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid - doesn't multiply correctly
    is_valid, errors = validate_parallelism_config(2, 2, 1, 2, 10)
    assert is_valid is False
    assert len(errors) == 1
    assert "multiply to 8 but total GPUs is 10" in errors[0]

    # Invalid - negative values
    is_valid, errors = validate_parallelism_config(-1, 2, 1, 2, 8)
    assert is_valid is False
    assert "must be positive" in errors[0]


def test_get_model_architecture_info():
    """Test model architecture information extraction."""
    config = MockConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        vocab_size=32000,
        intermediate_size=11008,
        num_local_experts=8,
        model_type="llama",
    )

    info = get_model_architecture_info(config)

    assert info["hidden_size"] == 4096
    assert info["num_attention_heads"] == 32
    assert info["num_key_value_heads"] == 8
    assert info["num_hidden_layers"] == 32
    assert info["vocab_size"] == 32000
    assert info["intermediate_size"] == 11008
    assert info["num_experts"] == 8
    assert info["model_type"] == "llama"
    assert info["is_moe"] is True


def test_check_tensor_parallel_efficiency():
    """Test tensor parallel efficiency warnings."""
    config = MockConfig(
        hidden_size=4096,
        intermediate_size=11008,
        vocab_size=32000,
        num_attention_heads=32,
    )

    # Efficient configuration (all divisible)
    warnings = check_tensor_parallel_efficiency(config, 4)
    assert len(warnings) == 0

    # Inefficient configuration (hidden size not divisible)
    config_bad = MockConfig(
        hidden_size=4097,  # Not divisible by 4
        intermediate_size=11008,
        vocab_size=32000,
        num_attention_heads=32,
    )
    warnings = check_tensor_parallel_efficiency(config_bad, 4)
    assert len(warnings) > 0
    assert "Hidden size" in warnings[0]


def test_check_pipeline_parallel_efficiency():
    """Test pipeline parallel efficiency warnings."""
    config = MockConfig(num_hidden_layers=32)

    # Efficient configuration
    warnings = check_pipeline_parallel_efficiency(config, 4)
    assert len(warnings) == 0

    # Too many stages
    warnings = check_pipeline_parallel_efficiency(config, 8)
    assert len(warnings) > 0
    assert "bubble overhead" in warnings[0]

    # Uneven distribution
    config_uneven = MockConfig(num_hidden_layers=30)
    warnings = check_pipeline_parallel_efficiency(config_uneven, 4)
    assert len(warnings) > 0
    assert "not evenly divisible" in warnings[0]


def test_check_expert_parallel_efficiency():
    """Test expert parallel efficiency warnings."""
    # Dense model
    dense_config = MockConfig(num_experts=0)
    warnings = check_expert_parallel_efficiency(dense_config, 2)
    assert len(warnings) > 0
    assert "should be 1 for non-MoE models" in warnings[0]

    # MoE model - efficient
    moe_config = MockConfig(num_local_experts=8, num_experts=8)
    warnings = check_expert_parallel_efficiency(moe_config, 4)
    assert len(warnings) == 0

    # MoE model - too many EP groups
    warnings = check_expert_parallel_efficiency(moe_config, 16)
    assert len(warnings) > 0
    assert "communication overhead" in warnings[0]


def test_edge_cases():
    """Test edge cases and error handling."""
    # Config with missing attributes
    minimal_config = MockConfig()

    # Should use defaults
    valid_tp = valid_tensor_parallel_sizes(minimal_config, 8)
    assert len(valid_tp) > 0

    valid_pp = valid_pipeline_parallel_sizes(minimal_config, 8)
    assert len(valid_pp) > 0

    valid_ep = valid_expert_parallel_sizes(minimal_config, 8)
    assert valid_ep == [1]  # Default for non-MoE


def test_integration_scenario():
    """Test realistic model configuration scenario."""
    # Llama-2-7B like configuration
    llama_config = MockConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=32,
        vocab_size=32000,
        intermediate_size=11008,
        model_type="llama",
    )

    # Test with 8 GPUs
    max_gpus = 8

    valid_tp = valid_tensor_parallel_sizes(llama_config, max_gpus)
    valid_pp = valid_pipeline_parallel_sizes(llama_config, max_gpus)
    valid_ep = valid_expert_parallel_sizes(llama_config, max_gpus)

    # Should have multiple valid options
    assert len(valid_tp) >= 3  # At least 1, 2, 4, 8
    assert len(valid_pp) >= 4  # At least 1, 2, 4, 8
    assert valid_ep == [1]  # Dense model

    # Test a valid combination
    tp, pp, ep = 2, 2, 1
    dp = max_gpus // (tp * pp * ep)

    is_valid, errors = validate_parallelism_config(tp, pp, ep, dp, max_gpus)
    assert is_valid is True
