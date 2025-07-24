"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


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


@pytest.fixture
def mock_llama_config():
    """Mock LLaMA model configuration for testing."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
    }


@pytest.fixture
def mock_gqa_config():
    """Mock GQA (Grouped Query Attention) model configuration."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA: fewer KV heads than Q heads
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def mock_moe_config():
    """Mock MoE (Mixture of Experts) model configuration."""
    return {
        "model_type": "mixtral",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 32000,
        "intermediate_size": 14336,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def mock_single_layer_config():
    """Mock single layer model configuration for edge case testing."""
    return {
        "model_type": "gpt2",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "num_hidden_layers": 1,
        "vocab_size": 50257,
        "intermediate_size": 3072,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": True,
    }


@pytest.fixture
def mock_massive_moe_config():
    """Mock massive MoE configuration for edge case testing."""
    return {
        "model_type": "mixtral",
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 80,
        "vocab_size": 100000,
        "intermediate_size": 28672,
        "num_local_experts": 64,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def constraint_params():
    """Default constraint parameters for testing."""
    from dataclasses import dataclass

    @dataclass
    class ParallelismConstraintParameters:
        default_min_layers_per_stage: int = 2
        default_max_tensor_parallel: int = 64
        min_experts_per_device: int = 1
        vocab_large_threshold: int = 100000
        vocab_medium_threshold: int = 50000
        vocab_large_divisibility: int = 8
        vocab_medium_divisibility: int = 4
        vocab_small_divisibility: int = 2

    return ParallelismConstraintParameters()
