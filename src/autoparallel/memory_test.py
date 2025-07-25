"""Tests for simplified memory estimation module."""

from autoparallel.memory import (
    MemoryBreakdown,
    _estimate_activations,
    _estimate_kv_cache,
    _estimate_param_count,
    check_memory_feasibility,
    estimate_memory,
    estimate_memory_for_config,
    get_quantization_bytes,
)


class MockConfig(dict):
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_memory_breakdown():
    """Test MemoryBreakdown dataclass."""
    breakdown = MemoryBreakdown(
        weights=1000, activations=500, kv_cache=300, framework_overhead=200
    )

    assert breakdown.total == 2000
    assert breakdown.weights == 1000
    assert breakdown.activations == 500
    assert breakdown.kv_cache == 300
    assert breakdown.framework_overhead == 200


def test_memory_breakdown_fits_in_gpu():
    """Test GPU memory fitting calculation."""
    breakdown = MemoryBreakdown(
        weights=1024**3,  # 1GB
        activations=512 * 1024**2,  # 0.5GB
        kv_cache=256 * 1024**2,  # 0.25GB
        framework_overhead=256 * 1024**2,  # 0.25GB
    )
    # Total: 2GB

    assert breakdown.fits_in_gpu(4.0) is True  # 2GB < 90% of 4GB
    assert breakdown.fits_in_gpu(2.0) is False  # 2GB > 90% of 2GB


def test_memory_breakdown_scale_by_parallelism():
    """Test parallelism scaling."""
    breakdown = MemoryBreakdown(
        weights=8000, activations=4000, kv_cache=2000, framework_overhead=1000
    )

    # Test tensor parallelism scaling
    scaled = breakdown.scale_by_parallelism(tensor_parallel=2)
    assert scaled.weights == 4000  # weights / TP
    assert scaled.activations == 2000  # activations / TP
    assert scaled.kv_cache == 1000  # kv_cache / TP
    assert scaled.framework_overhead == 1000  # framework doesn't scale

    # Test pipeline parallelism scaling
    scaled = breakdown.scale_by_parallelism(pipeline_parallel=2)
    assert scaled.weights == 8000  # weights don't scale by PP
    assert scaled.activations == 2000  # activations / PP
    assert scaled.kv_cache == 2000  # kv_cache doesn't scale by PP

    # Test expert parallelism scaling
    scaled = breakdown.scale_by_parallelism(expert_parallel=2)
    assert scaled.weights == 4000  # weights / EP
    assert scaled.activations == 4000  # activations don't scale by EP
    assert scaled.kv_cache == 2000  # kv_cache doesn't scale by EP

    # Test combined scaling
    scaled = breakdown.scale_by_parallelism(
        tensor_parallel=2, pipeline_parallel=2, expert_parallel=2
    )
    assert scaled.weights == 2000  # 8000 / (2 * 2)
    assert scaled.activations == 1000  # 4000 / (2 * 2)
    assert scaled.kv_cache == 1000  # 2000 / 2


def test_estimate_param_count():
    """Test parameter counting for different model types."""
    # Dense transformer (Llama-like)
    dense_params = _estimate_param_count(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        intermediate_size=11008,
        num_experts=0,
    )

    # Should be roughly 5-7B parameters (simplified estimation)
    assert 5_000_000_000 < dense_params < 8_000_000_000

    # MoE transformer
    moe_params = _estimate_param_count(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        intermediate_size=11008,
        num_experts=8,
    )

    # MoE should have significantly more parameters
    assert moe_params > dense_params * 4


def test_estimate_activations():
    """Test activation memory estimation."""
    activations = _estimate_activations(
        batch_size=1,
        sequence_length=2048,
        hidden_size=4096,
        num_layers=32,
        intermediate_size=11008,
    )

    # Should be reasonable size (not too small or large)
    assert activations > 0
    assert activations < 1_000_000_000  # Less than 1B elements

    # Should scale with batch size
    activations_batch2 = _estimate_activations(
        batch_size=2,
        sequence_length=2048,
        hidden_size=4096,
        num_layers=32,
        intermediate_size=11008,
    )
    assert activations_batch2 > activations


def test_estimate_kv_cache():
    """Test KV cache estimation."""
    kv_cache = _estimate_kv_cache(
        batch_size=1,
        sequence_length=2048,
        num_layers=32,
        num_key_value_heads=32,
        head_dim=128,
    )

    # Expected: 2 * 1 * 32 * 2048 * 128 * 32 = ~536M elements
    expected = 2 * 1 * 32 * 2048 * 128 * 32
    assert abs(kv_cache - expected) < expected * 0.01  # Within 1%

    # Test GQA (fewer KV heads)
    gqa_kv_cache = _estimate_kv_cache(
        batch_size=1,
        sequence_length=2048,
        num_layers=32,
        num_key_value_heads=8,  # 1/4 of attention heads
        head_dim=128,
    )

    assert gqa_kv_cache == kv_cache // 4


def test_estimate_memory():
    """Test full memory estimation."""
    config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )

    memory = estimate_memory(
        model_config=config,
        sequence_length=2048,
        batch_size=1,
        quantization_bytes=2,
        framework_overhead_gb=2.0,
    )

    assert memory.weights > 0
    assert memory.activations > 0
    assert memory.kv_cache > 0
    assert memory.framework_overhead == 2 * (1024**3)  # 2GB
    assert memory.total > 0


def test_estimate_memory_with_dict_config():
    """Test memory estimation with dictionary config."""
    config = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 11008,
    }

    memory = estimate_memory(model_config=config, sequence_length=2048, batch_size=1)

    assert memory.total > 0


def test_estimate_memory_moe_model():
    """Test memory estimation for MoE model."""
    moe_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        num_local_experts=8,
    )

    moe_memory = estimate_memory(model_config=moe_config)

    # Compare to dense model
    dense_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )

    dense_memory = estimate_memory(model_config=dense_config)

    # MoE should have much larger weights
    assert moe_memory.weights > dense_memory.weights * 3


def test_estimate_memory_for_config():
    """Test memory estimation with parallelism configuration."""
    config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )

    # Single GPU config
    single_gpu = estimate_memory_for_config(
        model_config=config, tensor_parallel=1, pipeline_parallel=1, expert_parallel=1
    )

    # 4-way tensor parallel config
    tp4_config = estimate_memory_for_config(
        model_config=config, tensor_parallel=4, pipeline_parallel=1, expert_parallel=1
    )

    # With TP=4, weights and activations should be ~1/4, KV cache ~1/4
    assert tp4_config.weights <= single_gpu.weights // 3
    assert tp4_config.activations <= single_gpu.activations // 3
    assert tp4_config.kv_cache <= single_gpu.kv_cache // 3


def test_get_quantization_bytes():
    """Test quantization bytes mapping."""
    assert get_quantization_bytes("fp32") == 4
    assert get_quantization_bytes("float32") == 4
    assert get_quantization_bytes("fp16") == 2
    assert get_quantization_bytes("float16") == 2
    assert get_quantization_bytes("bf16") == 2
    assert get_quantization_bytes("bfloat16") == 2
    assert get_quantization_bytes("int8") == 1
    assert get_quantization_bytes("fp8") == 1
    assert get_quantization_bytes("unknown") == 2  # Default


def test_check_memory_feasibility():
    """Test memory feasibility checking."""
    # Small model that should fit
    small_config = MockConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
    )

    feasibility = check_memory_feasibility(
        model_config=small_config,
        gpu_memory_gb=24.0,  # A100-like
        sequence_length=512,
        batch_size=1,
    )

    assert feasibility["fits_in_single_gpu"] is True
    assert feasibility["memory_utilization"] < 1.0
    assert feasibility["total_memory_gb"] > 0
    assert feasibility["available_memory_gb"] == 24.0
    assert "breakdown" in feasibility

    # Large model that shouldn't fit in small GPU
    large_config = MockConfig(
        vocab_size=32000,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=64,
        intermediate_size=22016,
    )

    feasibility_large = check_memory_feasibility(
        model_config=large_config,
        gpu_memory_gb=8.0,  # Small GPU
        sequence_length=2048,
        batch_size=1,
    )

    assert feasibility_large["fits_in_single_gpu"] is False
    assert feasibility_large["memory_utilization"] > 1.0


def test_memory_estimation_defaults():
    """Test memory estimation with minimal config using defaults."""
    minimal_config = MockConfig()

    memory = estimate_memory(model_config=minimal_config)

    # Should use reasonable defaults and not crash
    assert memory.weights > 0
    assert memory.activations > 0
    assert memory.kv_cache > 0
    assert memory.framework_overhead > 0
    assert memory.total > 0


def test_memory_estimation_no_kv_cache():
    """Test memory estimation without KV cache."""
    config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )

    memory_with_kv = estimate_memory(model_config=config, enable_kv_cache=True)

    memory_without_kv = estimate_memory(model_config=config, enable_kv_cache=False)

    assert memory_without_kv.kv_cache == 0
    assert memory_with_kv.kv_cache > 0
    assert memory_without_kv.total < memory_with_kv.total


def test_integration_scenario():
    """Test realistic model memory estimation scenario."""
    # Llama-2-7B configuration
    llama_7b_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        model_type="llama",
    )

    # Test memory for different configurations
    configs = [
        {"tensor_parallel": 1, "pipeline_parallel": 1},
        {"tensor_parallel": 2, "pipeline_parallel": 1},
        {"tensor_parallel": 4, "pipeline_parallel": 1},
        {"tensor_parallel": 2, "pipeline_parallel": 2},
    ]

    memories = []
    for config in configs:
        memory = estimate_memory_for_config(
            model_config=llama_7b_config, sequence_length=2048, batch_size=1, **config
        )
        memories.append(memory)

    # More parallelism should reduce memory per GPU
    assert memories[0].total > memories[1].total  # TP=1 > TP=2
    assert memories[1].total > memories[2].total  # TP=2 > TP=4
    assert memories[0].total > memories[3].total  # No parallelism > TP=2,PP=2
