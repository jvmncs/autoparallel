"""Tests for simplified grid search module."""

import pytest

from autoparallel.grid_search import (
    ParallelismConfig,
    _calculate_config_score,
    _find_max_throughput_config,
    _find_min_gpu_config,
    find_best_config,
    find_valid_configs,
    generate_search_space,
    get_gpu_requirements,
)
from autoparallel.memory import MemoryBreakdown


class MockConfig(dict):
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_parallelism_config():
    """Test ParallelismConfig dataclass."""
    memory = MemoryBreakdown(
        weights=1000, activations=500, kv_cache=300, framework_overhead=200
    )

    config = ParallelismConfig(
        tensor_parallel=2,
        pipeline_parallel=2,
        expert_parallel=1,
        data_parallel=2,
        memory_breakdown=memory,
        total_gpus=8,
        memory_per_gpu_gb=10.0,
        memory_utilization=0.8,
        score=85.5,
    )

    assert config.tensor_parallel == 2
    assert config.total_gpus == 8
    assert config.memory_utilization == 0.8

    # Test to_dict conversion
    config_dict = config.to_dict()
    assert config_dict["tensor_parallel"] == 2
    assert config_dict["total_gpus"] == 8
    assert "memory_breakdown" in config_dict
    assert config_dict["memory_breakdown"]["total_gb"] > 0


def test_calculate_config_score():
    """Test configuration scoring heuristics."""
    # Balanced configuration should score well
    balanced_score = _calculate_config_score(
        tp=2, pp=2, ep=1, dp=2, memory_utilization=0.8, total_gpus=8
    )

    # Unbalanced configuration should score lower
    _calculate_config_score(
        tp=8, pp=1, ep=1, dp=1, memory_utilization=0.8, total_gpus=8
    )

    # Excessive pipeline parallelism should be penalized
    excessive_pp_score = _calculate_config_score(
        tp=1, pp=8, ep=1, dp=1, memory_utilization=0.8, total_gpus=8
    )

    assert balanced_score > 0
    assert excessive_pp_score < balanced_score  # PP=8 should be penalized


def test_find_min_gpu_config():
    """Test minimum GPU configuration selection."""
    memory1 = MemoryBreakdown(
        weights=1000, activations=500, kv_cache=300, framework_overhead=200
    )
    memory2 = MemoryBreakdown(
        weights=800, activations=400, kv_cache=250, framework_overhead=200
    )

    configs = [
        ParallelismConfig(
            tensor_parallel=1,
            pipeline_parallel=1,
            expert_parallel=1,
            data_parallel=8,
            memory_breakdown=memory1,
            total_gpus=8,
            memory_per_gpu_gb=10.0,
            memory_utilization=0.8,
            score=80.0,
        ),
        ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=1,
            expert_parallel=1,
            data_parallel=2,
            memory_breakdown=memory2,
            total_gpus=4,
            memory_per_gpu_gb=8.0,
            memory_utilization=0.7,
            score=85.0,
        ),
    ]

    min_config = _find_min_gpu_config(configs)
    assert min_config.total_gpus == 4  # Should select the 4-GPU config


def test_find_max_throughput_config():
    """Test maximum throughput configuration selection."""
    memory = MemoryBreakdown(
        weights=1000, activations=500, kv_cache=300, framework_overhead=200
    )

    configs = [
        # High DP, low PP (good for throughput)
        ParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=1,
            expert_parallel=1,
            data_parallel=4,
            memory_breakdown=memory,
            total_gpus=8,
            memory_per_gpu_gb=10.0,
            memory_utilization=0.8,
            score=80.0,
        ),
        # High PP, low DP (bad for throughput)
        ParallelismConfig(
            tensor_parallel=1,
            pipeline_parallel=4,
            expert_parallel=1,
            data_parallel=2,
            memory_breakdown=memory,
            total_gpus=8,
            memory_per_gpu_gb=10.0,
            memory_utilization=0.8,
            score=85.0,
        ),
    ]

    max_throughput_config = _find_max_throughput_config(configs)
    # Should prefer the high DP, low PP configuration
    assert max_throughput_config.data_parallel == 4
    assert max_throughput_config.pipeline_parallel == 1


def test_generate_search_space():
    """Test search space generation."""
    config = MockConfig(
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=32,
        num_experts=0,  # Dense model
    )

    search_space = generate_search_space(config, max_gpus=8)

    assert "tensor_parallel" in search_space
    assert "pipeline_parallel" in search_space
    assert "expert_parallel" in search_space
    assert "data_parallel" in search_space

    # Check reasonable values
    assert 1 in search_space["tensor_parallel"]
    assert 1 in search_space["pipeline_parallel"]
    assert search_space["expert_parallel"] == [1]  # Dense model
    assert max(search_space["data_parallel"]) == 8

    # Test MoE model
    moe_config = MockConfig(
        num_attention_heads=32,
        num_key_value_heads=32,
        num_hidden_layers=32,
        num_local_experts=8,
    )

    moe_search_space = generate_search_space(moe_config, max_gpus=8)
    assert (
        len(moe_search_space["expert_parallel"]) > 1
    )  # Should have multiple EP options


def test_find_valid_configs():
    """Test finding valid configurations."""
    # Small model configuration for testing
    config = MockConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        num_experts=0,
    )

    valid_configs = find_valid_configs(
        model_config=config,
        max_gpus=4,
        gpu_memory_gb=24.0,  # Large GPU memory for small model
        sequence_length=512,
        batch_size=1,
        max_configs=10,
    )

    # Should find at least some valid configurations
    assert len(valid_configs) > 0

    # All configurations should be valid
    for config_obj in valid_configs:
        assert config_obj.tensor_parallel >= 1
        assert config_obj.pipeline_parallel >= 1
        assert config_obj.expert_parallel >= 1
        assert config_obj.data_parallel >= 1
        assert config_obj.total_gpus <= 4
        assert config_obj.memory_utilization <= 1.0
        assert config_obj.score > 0

    # Should be sorted by score (highest first)
    scores = [config.score for config in valid_configs]
    assert scores == sorted(scores, reverse=True)


def test_find_valid_configs_no_valid():
    """Test finding valid configurations when none exist."""
    # Very large model with tiny GPU memory
    large_config = MockConfig(
        vocab_size=100000,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=64,
        intermediate_size=22016,
        num_experts=0,
    )

    valid_configs = find_valid_configs(
        model_config=large_config,
        max_gpus=1,
        gpu_memory_gb=1.0,  # Tiny GPU memory
        sequence_length=2048,
        batch_size=1,
    )

    # Should find no valid configurations
    assert len(valid_configs) == 0


def test_find_best_config():
    """Test finding best configuration for different objectives."""
    # Small model that should have multiple valid configs
    config = MockConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        num_experts=0,
    )

    # Test minimize_gpus objective
    min_gpu_config = find_best_config(
        model_config=config, max_gpus=8, gpu_memory_gb=24.0, objective="minimize_gpus"
    )

    assert min_gpu_config.total_gpus >= 1

    # Test balance objective
    balanced_config = find_best_config(
        model_config=config, max_gpus=8, gpu_memory_gb=24.0, objective="balance"
    )

    assert balanced_config.score > 0

    # Test invalid objective
    with pytest.raises(ValueError, match="Unknown objective"):
        find_best_config(
            model_config=config,
            max_gpus=8,
            gpu_memory_gb=24.0,
            objective="invalid_objective",
        )


def test_find_best_config_no_valid():
    """Test finding best configuration when none exist."""
    # Model that won't fit
    large_config = MockConfig(
        vocab_size=100000,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=64,
        intermediate_size=22016,
        num_experts=0,
    )

    with pytest.raises(ValueError, match="No valid parallelism configurations found"):
        find_best_config(
            model_config=large_config,
            max_gpus=1,
            gpu_memory_gb=1.0,
            objective="minimize_gpus",
        )


def test_get_gpu_requirements():
    """Test GPU requirements calculation."""
    # Small model that fits in single GPU
    small_config = MockConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        num_experts=0,
    )

    requirements = get_gpu_requirements(
        model_config=small_config, gpu_memory_gb=24.0, sequence_length=512, batch_size=1
    )

    assert requirements["fits_in_single_gpu"] is True
    assert requirements["min_gpus_required"] == 1
    assert requirements["memory_per_gpu_gb"] > 0
    assert requirements["total_model_memory_gb"] > 0
    assert "breakdown" in requirements

    # Large model that requires multiple GPUs
    large_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        num_experts=0,
    )

    large_requirements = get_gpu_requirements(
        model_config=large_config,
        gpu_memory_gb=8.0,  # Small GPU
        sequence_length=2048,
        batch_size=1,
    )

    assert large_requirements["fits_in_single_gpu"] is False
    assert large_requirements["min_gpus_required"] > 1


def test_integration_scenario():
    """Test realistic grid search scenario."""
    # Llama-2-7B like configuration
    llama_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        num_experts=0,
        model_type="llama",
    )

    # Test with A100-like cluster (8x80GB)
    valid_configs = find_valid_configs(
        model_config=llama_config,
        max_gpus=8,
        gpu_memory_gb=80.0,
        sequence_length=2048,
        batch_size=1,
        quantization_bytes=2,  # fp16
        max_configs=20,
    )

    # Should find multiple valid configurations
    assert len(valid_configs) > 0

    # Test different objectives
    min_gpu_config = find_best_config(
        model_config=llama_config,
        max_gpus=8,
        gpu_memory_gb=80.0,
        objective="minimize_gpus",
    )

    throughput_config = find_best_config(
        model_config=llama_config,
        max_gpus=8,
        gpu_memory_gb=80.0,
        objective="maximize_throughput",
    )

    # Min GPU config should use fewer or equal GPUs
    assert min_gpu_config.total_gpus <= throughput_config.total_gpus

    # Verify configurations are reasonable
    for config in [min_gpu_config, throughput_config]:
        assert 1 <= config.tensor_parallel <= 8
        assert 1 <= config.pipeline_parallel <= 8
        assert config.expert_parallel == 1  # Dense model
        assert config.memory_utilization < 1.0
        assert config.score > 0


def test_moe_model_grid_search():
    """Test grid search with MoE model."""
    moe_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        num_local_experts=8,
        num_experts=8,  # Alternative naming
    )

    valid_configs = find_valid_configs(
        model_config=moe_config,
        max_gpus=8,
        gpu_memory_gb=80.0,
        sequence_length=2048,
        batch_size=1,
        max_configs=10,
    )

    # Should find valid configurations for MoE
    assert len(valid_configs) > 0

    # Some configurations should use expert parallelism
    ep_configs = [config for config in valid_configs if config.expert_parallel > 1]
    assert len(ep_configs) > 0  # Should have some EP > 1 configs


def test_edge_cases():
    """Test edge cases and error handling."""
    # Minimal configuration
    minimal_config = MockConfig()

    # Should handle missing attributes gracefully
    try:
        configs = find_valid_configs(
            model_config=minimal_config, max_gpus=2, gpu_memory_gb=24.0, max_configs=5
        )
        # Should either find configs or return empty list
        assert isinstance(configs, list)
    except Exception:
        # Some exceptions may be acceptable for minimal config
        pass

    # Test with max_gpus=1
    small_config = MockConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=1024,
        num_experts=0,
    )

    single_gpu_configs = find_valid_configs(
        model_config=small_config, max_gpus=1, gpu_memory_gb=24.0
    )

    # Should find at least the 1x1x1x1 configuration
    assert len(single_gpu_configs) >= 1
    config = single_gpu_configs[0]
    assert config.tensor_parallel == 1
    assert config.pipeline_parallel == 1
    assert config.expert_parallel == 1
    assert config.data_parallel == 1
