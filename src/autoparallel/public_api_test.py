"""Tests for simplified public API module."""

import contextlib
from unittest.mock import patch

import pytest

from autoparallel.public_api import (
    InsufficientMemoryError,
    ModelNotFoundError,
    _estimate_param_count_simple,
    _load_model_config,
    _validate_cluster_spec,
    analyze,
    best_config,
    check_memory_requirements,
    estimate_cost,
    find_minimum_gpus,
    get_memory_estimate,
    get_parallelism_config,
)


class MockConfig(dict):
    """Mock configuration for testing."""

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def sample_cluster():
    """Sample cluster specification."""
    return {"gpu_count": 8, "gpu_memory_gb": 80.0, "gpu_type": "A100"}


@pytest.fixture
def small_model_config():
    """Small model configuration for testing."""
    return MockConfig(
        vocab_size=1000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=2048,
        num_experts=0,
        model_type="test",
    )


def test_validate_cluster_spec():
    """Test cluster specification validation."""
    # Valid cluster
    valid_cluster = {"gpu_count": 8, "gpu_memory_gb": 80.0}
    _validate_cluster_spec(valid_cluster)  # Should not raise

    # Missing required keys
    with pytest.raises(ValueError, match="missing required key"):
        _validate_cluster_spec({"gpu_count": 8})

    with pytest.raises(ValueError, match="missing required key"):
        _validate_cluster_spec({"gpu_memory_gb": 80.0})

    # Invalid gpu_count
    with pytest.raises(ValueError, match="gpu_count must be a positive integer"):
        _validate_cluster_spec({"gpu_count": 0, "gpu_memory_gb": 80.0})

    with pytest.raises(ValueError, match="gpu_count must be a positive integer"):
        _validate_cluster_spec({"gpu_count": -1, "gpu_memory_gb": 80.0})

    # Invalid gpu_memory_gb
    with pytest.raises(ValueError, match="gpu_memory_gb must be a positive number"):
        _validate_cluster_spec({"gpu_count": 8, "gpu_memory_gb": 0})

    with pytest.raises(ValueError, match="gpu_memory_gb must be a positive number"):
        _validate_cluster_spec({"gpu_count": 8, "gpu_memory_gb": -1.0})


def test_estimate_param_count_simple():
    """Test simple parameter count estimation."""
    arch_info = {
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "num_experts": 0,
    }

    param_str = _estimate_param_count_simple(arch_info)
    assert "B" in param_str or "M" in param_str  # Should be in billions or millions

    # Test MoE model
    moe_arch_info = arch_info.copy()
    moe_arch_info["num_experts"] = 8

    moe_param_str = _estimate_param_count_simple(moe_arch_info)
    assert "B" in moe_param_str  # MoE should be larger


@patch("autoparallel.public_api.AutoConfig")
def test_load_model_config_success(mock_auto_config, small_model_config):
    """Test successful model configuration loading."""
    mock_auto_config.from_pretrained.return_value = small_model_config

    config = _load_model_config("test-model")
    assert config == small_model_config
    mock_auto_config.from_pretrained.assert_called_once_with(
        "test-model", trust_remote_code=False
    )


@patch("autoparallel.public_api.AutoConfig")
def test_load_model_config_failure(mock_auto_config):
    """Test model configuration loading failure."""
    mock_auto_config.from_pretrained.side_effect = Exception("Model not found")

    with pytest.raises(ModelNotFoundError, match="Failed to load model"):
        _load_model_config("nonexistent-model")


@patch("autoparallel.public_api.HAS_TRANSFORMERS", False)
def test_load_model_config_no_transformers():
    """Test model loading without transformers library."""
    with pytest.raises(ModelNotFoundError, match="transformers library is required"):
        _load_model_config("test-model")


@patch("autoparallel.public_api._load_model_config")
@patch("autoparallel.public_api.find_valid_configs")
def test_analyze_success(
    mock_find_configs, mock_load_config, sample_cluster, small_model_config
):
    """Test successful analyze function."""
    mock_load_config.return_value = small_model_config

    # Mock a valid configuration
    from autoparallel.grid_search import ParallelismConfig
    from autoparallel.memory import MemoryBreakdown

    mock_config = ParallelismConfig(
        tensor_parallel=2,
        pipeline_parallel=1,
        expert_parallel=1,
        data_parallel=4,
        memory_breakdown=MemoryBreakdown(
            weights=1000, activations=500, kv_cache=300, framework_overhead=200
        ),
        total_gpus=8,
        memory_per_gpu_gb=10.0,
        memory_utilization=0.8,
        score=85.0,
    )
    mock_find_configs.return_value = [mock_config]

    results = analyze("test-model", sample_cluster)

    assert len(results) == 1
    assert results[0]["tensor_parallel"] == 2
    assert results[0]["total_gpus"] == 8
    assert "memory_breakdown" in results[0]


@patch("autoparallel.public_api._load_model_config")
@patch("autoparallel.public_api.find_valid_configs")
def test_analyze_no_valid_configs(
    mock_find_configs, mock_load_config, sample_cluster, small_model_config
):
    """Test analyze function with no valid configurations."""
    mock_load_config.return_value = small_model_config
    mock_find_configs.return_value = []

    with pytest.raises(InsufficientMemoryError, match="cannot fit in cluster"):
        analyze("test-model", sample_cluster)


@patch("autoparallel.public_api._load_model_config")
@patch("autoparallel.public_api.find_best_config")
def test_best_config_success(
    mock_find_best, mock_load_config, sample_cluster, small_model_config
):
    """Test successful best_config function."""
    mock_load_config.return_value = small_model_config

    # Mock a best configuration
    from autoparallel.grid_search import ParallelismConfig
    from autoparallel.memory import MemoryBreakdown

    mock_config = ParallelismConfig(
        tensor_parallel=1,
        pipeline_parallel=1,
        expert_parallel=1,
        data_parallel=8,
        memory_breakdown=MemoryBreakdown(
            weights=800, activations=400, kv_cache=250, framework_overhead=200
        ),
        total_gpus=8,
        memory_per_gpu_gb=8.0,
        memory_utilization=0.7,
        score=90.0,
    )
    mock_find_best.return_value = mock_config

    result = best_config("test-model", sample_cluster, objective="minimize_gpus")

    assert result["tensor_parallel"] == 1
    assert result["total_gpus"] == 8
    assert result["score"] == 90.0


def test_best_config_invalid_objective(sample_cluster):
    """Test best_config with invalid objective."""
    with pytest.raises(ValueError, match="Invalid objective"):
        best_config("test-model", sample_cluster, objective="invalid")


@patch("autoparallel.public_api._load_model_config")
@patch("autoparallel.public_api.check_memory_feasibility")
@patch("autoparallel.public_api.get_model_architecture_info")
def test_check_memory_requirements(
    mock_arch_info, mock_feasibility, mock_load_config, small_model_config
):
    """Test memory requirements checking."""
    mock_load_config.return_value = small_model_config
    mock_feasibility.return_value = {
        "total_memory_gb": 15.0,
        "breakdown": {
            "weights_gb": 10.0,
            "activations_gb": 2.0,
            "kv_cache_gb": 2.0,
            "framework_overhead_gb": 1.0,
        },
    }
    mock_arch_info.return_value = {
        "model_type": "test",
        "num_hidden_layers": 6,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "vocab_size": 1000,
        "intermediate_size": 2048,
        "num_experts": 0,
        "is_moe": False,
    }

    result = check_memory_requirements("test-model")

    assert result["total_memory_gb"] == 15.0
    assert "breakdown" in result
    assert "single_gpu_requirements" in result
    assert "architecture_info" in result
    assert "A100_80GB" in result["single_gpu_requirements"]["fits_in_common_gpus"]


@patch("autoparallel.public_api.best_config")
def test_estimate_cost(mock_best_config, sample_cluster):
    """Test cost estimation."""
    mock_best_config.side_effect = [
        {"total_gpus": 2, "memory_utilization": 0.8},  # minimize_gpus
        {"total_gpus": 8, "memory_utilization": 0.6},  # maximize_throughput
        {"total_gpus": 4, "memory_utilization": 0.7},  # balance
    ]

    result = estimate_cost(
        "test-model", sample_cluster, hours_per_month=730, cost_per_gpu_hour=2.0
    )

    assert "cost_analysis" in result
    assert "minimize_gpus" in result["cost_analysis"]
    assert "maximize_throughput" in result["cost_analysis"]
    assert "balance" in result["cost_analysis"]

    # Check cost calculations
    min_gpu_cost = result["cost_analysis"]["minimize_gpus"]
    assert min_gpu_cost["gpus_used"] == 2
    assert min_gpu_cost["cost_per_hour"] == 4.0  # 2 GPUs * $2/hour
    assert min_gpu_cost["cost_per_month"] == 2920.0  # 2 * 730 * 2


def test_convenience_aliases():
    """Test convenience alias functions."""
    # These functions should exist and be callable
    assert callable(get_parallelism_config)
    assert callable(get_memory_estimate)
    assert callable(find_minimum_gpus)


@patch("autoparallel.public_api.best_config")
def test_find_minimum_gpus(mock_best_config):
    """Test find_minimum_gpus function."""
    mock_best_config.return_value = {
        "total_gpus": 4,
        "memory_per_gpu_gb": 18.5,
        "memory_utilization": 0.85,
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "expert_parallel": 1,
        "data_parallel": 2,
    }

    result = find_minimum_gpus("test-model", gpu_memory_gb=24.0)

    assert result["min_gpus"] == 4
    assert result["memory_per_gpu_gb"] == 18.5
    assert result["memory_utilization"] == 0.85
    assert "configuration" in result


@patch("autoparallel.public_api.best_config")
def test_find_minimum_gpus_failure(mock_best_config):
    """Test find_minimum_gpus with insufficient memory."""
    mock_best_config.side_effect = Exception("Cannot fit")

    with pytest.raises(InsufficientMemoryError, match="Cannot determine minimum GPUs"):
        find_minimum_gpus("test-model", gpu_memory_gb=1.0)


def test_analyze_invalid_cluster():
    """Test analyze with invalid cluster specification."""
    invalid_cluster = {"gpu_count": 0}

    with pytest.raises(ValueError, match="missing required key"):
        analyze("test-model", invalid_cluster)


@patch("autoparallel.public_api._load_model_config")
def test_analyze_model_load_failure(mock_load_config, sample_cluster):
    """Test analyze with model loading failure."""
    mock_load_config.side_effect = ModelNotFoundError("Model not found")

    with pytest.raises(ModelNotFoundError):
        analyze("nonexistent-model", sample_cluster)


def test_quantization_parameter():
    """Test that quantization parameter is handled correctly."""
    # This is more of an integration test to ensure quantization flows through
    with (
        patch("autoparallel.public_api._load_model_config") as mock_load,
        patch("autoparallel.public_api.find_valid_configs") as mock_find
    ):
        mock_load.return_value = MockConfig(vocab_size=1000, hidden_size=256)
        mock_find.return_value = []

        with contextlib.suppress(InsufficientMemoryError):
            analyze(
                "test-model",
                {"gpu_count": 1, "gpu_memory_gb": 24},
                quantization="int8",
            )

        # Verify that get_quantization_bytes was called
        # (indirectly through the flow)
        assert mock_find.called


def test_integration_with_real_small_config():
    """Test with a realistic small configuration without external dependencies."""
    small_config = MockConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=1024,
        num_experts=0,
        model_type="test",
    )

    cluster = {"gpu_count": 1, "gpu_memory_gb": 24.0}

    with patch("autoparallel.public_api._load_model_config", return_value=small_config):
        # Test analyze
        results = analyze("test-model", cluster)
        assert len(results) >= 1
        assert all(r["total_gpus"] <= cluster["gpu_count"] for r in results)

        # Test best_config
        best = best_config("test-model", cluster, objective="minimize_gpus")
        assert best["total_gpus"] <= cluster["gpu_count"]

        # Test check_memory_requirements
        memory_req = check_memory_requirements("test-model")
        assert memory_req["total_memory_gb"] > 0
        assert "architecture_info" in memory_req
