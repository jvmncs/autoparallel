"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent


@pytest.fixture
def sample_model_config():
    """Sample Hugging Face model configuration for testing."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "num_key_value_heads": 32,  # GQA support
        "max_position_embeddings": 4096,
    }


@pytest.fixture
def sample_cluster_config():
    """Sample cluster configuration for testing."""
    return {
        "nodes": 1,
        "gpus_per_node": 8,
        "gpu_memory_gb": 80,
        "gpu_type": "H100",
        "interconnect": "nvlink",
        "bandwidth_gbps": 900,
    }


@pytest.fixture
def sample_memory_config():
    """Sample memory configuration for testing."""
    return {
        "utilization_bound": 0.85,
        "fragmentation_overhead": 0.10,
        "safety_margin": 0.05,
        "quantization_format": "fp16",
    }
