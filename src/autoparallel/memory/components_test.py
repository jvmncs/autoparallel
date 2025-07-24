"""Tests for memory components."""

import pytest

from .components import MemoryComponents


def test_memory_components_creation():
    """Test basic MemoryComponents creation."""
    components = MemoryComponents(
        weights=1000,
        activations=500,
        kv_cache=200,
        cuda_graphs=100,
        optimizer_states=300,
        fragmentation_overhead=50,
        framework_overhead=25,
        safety_margin=15,
    )

    assert components.weights == 1000
    assert components.activations == 500
    assert components.kv_cache == 200
    assert components.cuda_graphs == 100
    assert components.optimizer_states == 300
    assert components.fragmentation_overhead == 50
    assert components.framework_overhead == 25
    assert components.safety_margin == 15


def test_memory_components_total_memory():
    """Test total memory calculation."""
    components = MemoryComponents(
        weights=1000,
        activations=500,
        kv_cache=200,
        cuda_graphs=100,
        optimizer_states=300,
        fragmentation_overhead=50,
        framework_overhead=25,
        safety_margin=15,
    )

    expected_total = 1000 + 500 + 200 + 100 + 300 + 50 + 25 + 15
    assert components.total_memory == expected_total


def test_memory_components_properties():
    """Test memory component properties."""
    components = MemoryComponents(
        weights=1000,
        activations=500,
        kv_cache=200,
        cuda_graphs=100,
        optimizer_states=300,
        fragmentation_overhead=50,
        framework_overhead=25,
        safety_margin=15,
    )

    assert components.model_memory == 1000 + 500 + 200  # weights + activations + kv_cache
    assert components.overhead_memory == 50 + 25 + 15  # fragmentation + framework + safety
    assert components.framework_memory == 100 + 300  # cuda_graphs + optimizer_states


def test_memory_components_unit_conversion():
    """Test unit conversion methods."""
    components = MemoryComponents(
        weights=1024**3,  # 1 GB in bytes
        activations=512 * 1024**2,  # 512 MB in bytes
    )

    gb_components = components.to_gb()
    assert gb_components.weights == 1
    assert gb_components.activations == 0  # 512 MB rounds down to 0 GB

    mb_components = components.to_mb()
    assert mb_components.weights == 1024  # 1 GB = 1024 MB
    assert mb_components.activations == 512


def test_memory_components_breakdown_percentages():
    """Test percentage breakdown calculation."""
    components = MemoryComponents(
        weights=800,
        activations=150,
        kv_cache=50,
    )  # Total = 1000

    percentages = components.breakdown_percentages()

    assert percentages["weights"] == 80.0
    assert percentages["activations"] == 15.0
    assert percentages["kv_cache"] == 5.0
    assert percentages["cuda_graphs"] == 0.0
    assert percentages["optimizer_states"] == 0.0


def test_memory_components_breakdown_percentages_zero_total():
    """Test percentage breakdown with zero total memory."""
    components = MemoryComponents()  # All zeros

    percentages = components.breakdown_percentages()

    for component_name in [
        "weights",
        "activations",
        "kv_cache",
        "cuda_graphs",
        "optimizer_states",
        "fragmentation_overhead",
        "framework_overhead",
        "safety_margin",
    ]:
        assert percentages[component_name] == 0.0


def test_memory_components_scale_by_parallelism():
    """Test parallelism scaling."""
    components = MemoryComponents(
        weights=1000,
        activations=800,
        kv_cache=400,
        cuda_graphs=100,
        optimizer_states=2000,
        fragmentation_overhead=200,
        framework_overhead=50,
        safety_margin=100,
    )

    scaled = components.scale_by_parallelism(
        tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1
    )

    # Weights, activations, kv_cache should be divided by tensor_parallel_size
    assert scaled.weights == 1000 // 2
    assert scaled.activations == 800 // 2
    assert scaled.kv_cache == 400 // 2

    # CUDA graphs don't scale with TP
    assert scaled.cuda_graphs == 100

    # Optimizer states scale with TP
    assert scaled.optimizer_states == 2000 // 2

    # Overhead scales proportionally
    assert scaled.fragmentation_overhead == 200 // 2
    assert scaled.framework_overhead == 50  # Framework overhead doesn't scale
    assert scaled.safety_margin == 100 // 2


def test_memory_components_validation():
    """Test validation of memory component values."""
    # Negative values should raise ValueError
    with pytest.raises(ValueError, match="Memory components must be non-negative"):
        MemoryComponents(weights=-100)

    with pytest.raises(ValueError, match="Memory components must be non-negative"):
        MemoryComponents(activations=-50)

    # Zero and positive values should be fine
    MemoryComponents(weights=0, activations=100)  # Should not raise


def test_memory_components_string_representation():
    """Test string representation."""
    components = MemoryComponents(
        weights=1024 * 1024,  # 1 MB
        activations=512 * 1024,  # 0.5 MB
        kv_cache=256 * 1024,  # 0.25 MB
    )

    str_repr = str(components)
    assert "MemoryComponents(" in str_repr
    assert "weights: 1 MB" in str_repr
    assert "activations: 0 MB" in str_repr  # Rounds down
    assert "kv_cache: 0 MB" in str_repr  # Rounds down
    assert "total:" in str_repr
