"""Tests for vLLM memory estimation."""

import pytest

from autoparallel.frameworks.vllm_memory import (
    WorkloadProfile,
    get_vllm_default_capture_sizes,
    vLLMAutotuningParameters,
    vLLMMemoryEstimator,
)
from autoparallel.memory.config import MemoryConfig


class TestWorkloadProfile:
    """Test WorkloadProfile functionality."""

    def test_create_chatbot_workload(self):
        """Test creating chatbot workload profile."""
        workload = WorkloadProfile.create_synthetic("chatbot")

        assert workload.requests_per_second == 100
        assert workload.target_metric == "latency"
        assert workload.latency_budget_ms == 100
        assert sum(workload.batch_size_distribution.values()) == pytest.approx(1.0)
        assert sum(workload.sequence_length_distribution.values()) == pytest.approx(1.0)

    def test_create_batch_inference_workload(self):
        """Test creating batch inference workload profile."""
        workload = WorkloadProfile.create_synthetic("batch_inference")

        assert workload.requests_per_second == 10
        assert workload.target_metric == "throughput"
        assert workload.latency_budget_ms == 1000
        assert sum(workload.batch_size_distribution.values()) == pytest.approx(1.0)

    def test_create_interactive_workload(self):
        """Test creating interactive workload profile."""
        workload = WorkloadProfile.create_synthetic("interactive")

        assert workload.requests_per_second == 50
        assert workload.target_metric == "latency"
        assert workload.latency_budget_ms == 50
        assert workload.batch_size_distribution[1] == 0.8  # Most requests are single

    def test_custom_parameters(self):
        """Test creating workload with custom parameters."""
        workload = WorkloadProfile.create_synthetic(
            "chatbot", requests_per_second=200, latency_budget_ms=50
        )

        assert workload.requests_per_second == 200
        assert workload.latency_budget_ms == 50

    def test_invalid_workload_type(self):
        """Test error for invalid workload type."""
        with pytest.raises(ValueError, match="Unknown workload type"):
            WorkloadProfile.create_synthetic("invalid_type")

    def test_get_expected_max_batch_size(self):
        """Test getting expected maximum batch size."""
        workload = WorkloadProfile.create_synthetic("batch_inference")

        # Should return size at 95th percentile
        max_batch = workload.get_expected_max_batch_size(percentile=0.95)
        assert max_batch >= 16  # At least the minimum batch size

        # Should return maximum for 100th percentile
        max_batch_100 = workload.get_expected_max_batch_size(percentile=1.0)
        assert max_batch_100 == max(workload.batch_size_distribution.keys())

    def test_empty_batch_distribution_fallback(self):
        """Test fallback when batch distribution is empty."""
        workload = WorkloadProfile(
            requests_per_second=100,
            batch_size_distribution={},
            sequence_length_distribution={512: 1.0},
        )

        max_batch = workload.get_expected_max_batch_size()
        assert max_batch == 32  # Default fallback


class TestvLLMDefaultCaptureSizes:
    """Test vLLM default capture sizes function."""

    def test_default_capture_sizes_small_limit(self):
        """Test with small limit."""
        sizes = get_vllm_default_capture_sizes(10)
        expected = [1, 2, 4, 8]
        assert sizes == expected

    def test_default_capture_sizes_medium_limit(self):
        """Test with medium limit."""
        sizes = get_vllm_default_capture_sizes(32)
        expected = [1, 2, 4, 8, 16, 24, 32]
        assert sizes == expected

    def test_default_capture_sizes_large_limit(self):
        """Test with large limit includes full range."""
        sizes = get_vllm_default_capture_sizes(520)

        # Should include base sizes
        assert 1 in sizes
        assert 2 in sizes
        assert 4 in sizes

        # Should include range(8, 513, 8)
        assert 8 in sizes
        assert 16 in sizes
        assert 512 in sizes

        # Should not exceed limit
        assert all(size <= 520 for size in sizes)

    def test_version_parameter_ignored(self):
        """Test that version parameter doesn't affect output currently."""
        sizes_v0 = get_vllm_default_capture_sizes(32, vllm_version="v0")
        sizes_v1 = get_vllm_default_capture_sizes(32, vllm_version="v1")
        assert sizes_v0 == sizes_v1


class TestvLLMAutotuningParameters:
    """Test vLLM autotuning parameters."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = vLLMAutotuningParameters()

        assert params.graph_memory_overhead_base_ratio == 0.1
        assert params.graph_memory_batch_scaling_factor == 0.02
        assert params.compilation_level == "PIECEWISE"
        assert params.min_gpu_memory_utilization == 0.8
        assert params.max_gpu_memory_utilization == 0.98

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = vLLMAutotuningParameters(
            graph_memory_overhead_base_ratio=0.15, compilation_level="FULL"
        )

        assert params.graph_memory_overhead_base_ratio == 0.15
        assert params.compilation_level == "FULL"


class TestvLLMMemoryEstimator:
    """Test vLLM memory estimator functionality."""

    @pytest.fixture
    def estimator(self):
        """Create vLLM memory estimator for testing."""
        return vLLMMemoryEstimator()

    @pytest.fixture
    def model_config(self):
        """Create sample model configuration."""
        return {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

    def test_initialization(self):
        """Test estimator initialization."""
        # Default initialization
        estimator = vLLMMemoryEstimator()
        assert estimator.config is not None
        assert estimator.autotuning_params is not None

        # Custom initialization
        config = MemoryConfig(utilization_bound=0.9)
        params = vLLMAutotuningParameters(compilation_level="FULL")
        estimator = vLLMMemoryEstimator(config=config, autotuning_params=params)

        assert estimator.config.utilization_bound == 0.9
        assert estimator.autotuning_params.compilation_level == "FULL"

    def test_estimate_memory_basic(self, estimator, model_config):
        """Test basic memory estimation."""
        components = estimator.estimate_memory(
            model_config=model_config,
            sequence_length=2048,
            batch_size=8,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            is_training=False,
        )

        assert components.weights > 0
        assert components.activations > 0
        assert components.kv_cache > 0
        assert components.cuda_graphs > 0
        assert components.optimizer_states == 0  # Not training
        assert components.total_memory > 0

    def test_estimate_memory_with_parallelism(self, estimator, model_config):
        """Test memory estimation with parallelism scaling."""
        components_tp1 = estimator.estimate_memory(
            model_config=model_config,
            sequence_length=2048,
            batch_size=8,
            tensor_parallel_size=1,
        )

        components_tp2 = estimator.estimate_memory(
            model_config=model_config,
            sequence_length=2048,
            batch_size=8,
            tensor_parallel_size=2,
        )

        # Weights should be split across TP devices
        assert components_tp2.weights < components_tp1.weights
        assert components_tp2.kv_cache < components_tp1.kv_cache

    def test_estimate_memory_training(self, estimator, model_config):
        """Test memory estimation for training."""
        components = estimator.estimate_memory(
            model_config=model_config,
            sequence_length=2048,
            batch_size=8,
            is_training=True,
        )

        assert components.optimizer_states > 0

    def test_estimate_vllm_activations_memory(self, estimator, model_config):
        """Test vLLM-specific activation memory estimation."""
        activation_memory = estimator.estimate_vllm_activations_memory(
            model_config, sequence_length=2048, batch_size=8
        )

        assert activation_memory > 0

        # Should scale with batch size and sequence length
        activation_memory_larger = estimator.estimate_vllm_activations_memory(
            model_config, sequence_length=4096, batch_size=16
        )
        assert activation_memory_larger > activation_memory

    def test_estimate_vllm_kv_cache_memory(self, estimator, model_config):
        """Test vLLM KV cache memory estimation."""
        # Test with default dtype (fp16)
        kv_cache_fp16 = estimator.estimate_vllm_kv_cache_memory(
            model_config, sequence_length=2048, batch_size=8, kv_cache_dtype="auto"
        )

        # Test with fp8
        kv_cache_fp8 = estimator.estimate_vllm_kv_cache_memory(
            model_config, sequence_length=2048, batch_size=8, kv_cache_dtype="fp8"
        )

        assert kv_cache_fp16 > 0
        assert kv_cache_fp8 > 0
        assert kv_cache_fp8 < kv_cache_fp16  # fp8 should use less memory

    def test_estimate_vllm_kv_cache_memory_gqa(self, estimator):
        """Test KV cache memory with Grouped Query Attention."""
        model_config_gqa = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,  # GQA: fewer KV heads
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

        kv_cache_gqa = estimator.estimate_vllm_kv_cache_memory(
            model_config_gqa, sequence_length=2048, batch_size=8
        )

        kv_cache_standard = estimator.estimate_vllm_kv_cache_memory(
            {**model_config_gqa, "num_key_value_heads": 32},
            sequence_length=2048,
            batch_size=8,
        )

        assert kv_cache_gqa > 0
        assert kv_cache_gqa < kv_cache_standard  # GQA should use less memory

    def test_estimate_vllm_cuda_graphs_memory(self, estimator, model_config):
        """Test vLLM CUDA graphs memory estimation."""
        # Test with no capture sizes
        cuda_memory_none = estimator.estimate_vllm_cuda_graphs_memory(
            model_config, capture_sizes=[]
        )
        assert cuda_memory_none == 0

        # Test with capture sizes
        capture_sizes = [1, 2, 4, 8]
        cuda_memory = estimator.estimate_vllm_cuda_graphs_memory(
            model_config, capture_sizes=capture_sizes
        )
        assert cuda_memory > 0

        # More capture sizes should use more memory
        larger_capture_sizes = [1, 2, 4, 8, 16, 32]
        cuda_memory_larger = estimator.estimate_vllm_cuda_graphs_memory(
            model_config, capture_sizes=larger_capture_sizes
        )
        assert cuda_memory_larger > cuda_memory

    def test_estimate_vllm_cuda_graphs_memory_compilation_levels(
        self, estimator, model_config
    ):
        """Test CUDA graphs memory with different compilation levels."""
        capture_sizes = [1, 2, 4, 8]

        # Test PIECEWISE compilation
        estimator.autotuning_params.compilation_level = "PIECEWISE"
        cuda_memory_piecewise = estimator.estimate_vllm_cuda_graphs_memory(
            model_config, capture_sizes=capture_sizes
        )

        # Test FULL compilation
        estimator.autotuning_params.compilation_level = "FULL"
        cuda_memory_full = estimator.estimate_vllm_cuda_graphs_memory(
            model_config, capture_sizes=capture_sizes
        )

        assert cuda_memory_full > cuda_memory_piecewise

    def test_calculate_effective_batch_size(self, estimator, model_config):
        """Test effective batch size calculation."""
        gpu_memory_gb = 80  # 80GB GPU
        max_model_len = 2048

        batch_size = estimator.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=max_model_len,
            gpu_memory_capacity_bytes=int(gpu_memory_gb * (1024**3)),
            gpu_memory_utilization=0.9,
        )

        assert batch_size > 0

        # Higher memory utilization should allow larger batch size
        batch_size_higher = estimator.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=max_model_len,
            gpu_memory_capacity_bytes=int(gpu_memory_gb * (1024**3)),
            gpu_memory_utilization=0.95,
        )

        assert batch_size_higher >= batch_size

    def test_calculate_effective_batch_size_fp8(self, estimator, model_config):
        """Test effective batch size with fp8 KV cache."""
        gpu_memory_gb = 80
        max_model_len = 2048

        batch_size_fp16 = estimator.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=max_model_len,
            gpu_memory_capacity_bytes=int(gpu_memory_gb * (1024**3)),
            kv_cache_dtype="auto",
        )

        batch_size_fp8 = estimator.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=max_model_len,
            gpu_memory_capacity_bytes=int(gpu_memory_gb * (1024**3)),
            kv_cache_dtype="fp8",
        )

        assert batch_size_fp8 > batch_size_fp16  # fp8 should allow larger batch size

    def test_calculate_effective_batch_size_insufficient_memory(
        self, estimator, model_config
    ):
        """Test effective batch size with insufficient memory."""
        gpu_memory_gb = 1  # Very small GPU
        max_model_len = 2048

        batch_size = estimator.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=max_model_len,
            gpu_memory_capacity_bytes=int(gpu_memory_gb * (1024**3)),
            gpu_memory_utilization=0.9,
        )

        assert batch_size == 0  # Should return 0 for insufficient memory

    def test_calculate_graph_coverage(self, estimator):
        """Test graph coverage calculation."""
        workload = WorkloadProfile.create_synthetic("batch_inference")
        capture_sizes = [16, 32, 64]

        coverage = estimator.calculate_graph_coverage(workload, capture_sizes)

        assert 0.0 <= coverage <= 1.0

        # Should be > 0 since workload has matching batch sizes
        assert coverage > 0

        # Empty capture sizes should give 0 coverage
        coverage_empty = estimator.calculate_graph_coverage(workload, [])
        assert coverage_empty == 0.0

    def test_calculate_memory_breakdown(self, estimator, model_config):
        """Test detailed memory breakdown calculation."""
        gpu_memory_gb = 80
        capture_sizes = [1, 2, 4, 8]

        breakdown = estimator.calculate_memory_breakdown(
            model_config=model_config,
            gpu_memory_capacity_gb=gpu_memory_gb,
            gpu_memory_utilization=0.9,
            capture_sizes=capture_sizes,
            max_model_len=2048,
        )

        required_keys = [
            "model_memory",
            "activation_memory",
            "cuda_graph_memory",
            "kv_cache_memory",
            "total_used",
            "utilization_ratio",
        ]

        for key in required_keys:
            assert key in breakdown
            assert breakdown[key] >= 0

        assert breakdown["utilization_ratio"] == 0.9
        assert breakdown["total_used"] <= gpu_memory_gb

    def test_evaluate_config_performance_throughput(self, estimator, model_config):
        """Test performance evaluation for throughput workload."""
        workload = WorkloadProfile.create_synthetic("batch_inference")

        score = estimator.evaluate_config_performance(
            model_config=model_config,
            workload=workload,
            gpu_memory_capacity_gb=80,
            gpu_memory_utilization=0.9,
            capture_sizes=[16, 32, 64],
            max_model_len=2048,
        )

        assert score > 0

    def test_evaluate_config_performance_latency(self, estimator, model_config):
        """Test performance evaluation for latency workload."""
        workload = WorkloadProfile.create_synthetic("chatbot")

        score = estimator.evaluate_config_performance(
            model_config=model_config,
            workload=workload,
            gpu_memory_capacity_gb=80,
            gpu_memory_utilization=0.9,
            capture_sizes=[1, 2, 4],
            max_model_len=2048,
        )

        assert score > 0

    def test_evaluate_config_performance_invalid_metric(self, estimator, model_config):
        """Test error for invalid performance metric."""
        workload = WorkloadProfile(
            requests_per_second=100,
            batch_size_distribution={16: 1.0},
            target_metric="invalid_metric",
        )

        with pytest.raises(ValueError, match="Unknown target metric"):
            estimator.evaluate_config_performance(
                model_config=model_config,
                workload=workload,
                gpu_memory_capacity_gb=80,
                gpu_memory_utilization=0.9,
                capture_sizes=[16],
                max_model_len=2048,
            )

    def test_estimate_vllm_framework_overhead(self, estimator):
        """Test vLLM framework overhead estimation."""
        base_memory = 1024**3  # 1GB
        overhead = estimator.estimate_vllm_framework_overhead(base_memory)

        assert overhead > 0
        assert overhead < base_memory  # Should be reasonable overhead

        # Should scale with base memory
        larger_base = 10 * (1024**3)  # 10GB
        larger_overhead = estimator.estimate_vllm_framework_overhead(larger_base)
        assert larger_overhead > overhead


class TestIntegration:
    """Integration tests for vLLM memory estimator."""

    def test_realistic_llama_7b_config(self):
        """Test with realistic Llama 7B configuration."""
        estimator = vLLMMemoryEstimator()

        # Llama 7B configuration
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

        # Test memory estimation
        components = estimator.estimate_memory(
            model_config=model_config,
            sequence_length=2048,
            batch_size=8,
        )

        # Sanity checks for reasonable values
        assert components.total_memory > 0
        assert components.weights > 9 * (1024**3)  # > 9GB for 7B model (around 10GB)

        # Test effective batch size
        batch_size = estimator.calculate_effective_batch_size(
            model_config=model_config,
            max_model_len=2048,
            gpu_memory_capacity_bytes=80 * (1024**3),  # 80GB GPU
        )

        assert batch_size > 0
        assert batch_size < 1000  # Should be reasonable

    def test_workload_optimization_scenario(self):
        """Test complete workload optimization scenario."""
        estimator = vLLMMemoryEstimator()
        workload = WorkloadProfile.create_synthetic("chatbot")

        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

        # Test different configurations
        configs = [
            {"capture_sizes": [], "gpu_memory_utilization": 0.85},
            {"capture_sizes": [1, 2, 4], "gpu_memory_utilization": 0.90},
            {"capture_sizes": [1, 2, 4, 8, 16], "gpu_memory_utilization": 0.95},
        ]

        scores = []
        for config in configs:
            score = estimator.evaluate_config_performance(
                model_config=model_config,
                workload=workload,
                gpu_memory_capacity_gb=80,
                capture_sizes=config["capture_sizes"],
                gpu_memory_utilization=config["gpu_memory_utilization"],
                max_model_len=2048,
            )
            scores.append(score)

        # All scores should be valid
        assert all(score >= 0 for score in scores)

        # Should have variation in scores
        assert len(set(scores)) > 1  # Not all identical
