"""Tests for vLLM configuration system."""

import pytest
import transformers

from autoparallel.frameworks.vllm_config import (
    AutotuningParameters,
    calculate_model_memory_from_config,
    estimate_activation_memory_from_config,
    generate_deployment_recommendations,
    optimize_vllm_config_for_cluster,
    vLLMConfigOptimizer,
    vLLMPerformanceModel,
)
from autoparallel.frameworks.vllm_memory import WorkloadProfile


@pytest.fixture
def mock_llama_config():
    """Create a mock Llama-like model configuration."""
    config = transformers.PretrainedConfig()
    config.hidden_size = 4096
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.vocab_size = 32000
    config.intermediate_size = 11008
    config.torch_dtype = "float16"
    return config


@pytest.fixture
def small_model_config():
    """Create a small model configuration for testing."""
    config = transformers.PretrainedConfig()
    config.hidden_size = 768
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.num_key_value_heads = 12
    config.vocab_size = 30522
    config.intermediate_size = 3072
    config.torch_dtype = "float16"
    return config


@pytest.fixture
def autotuning_params():
    """Create test autotuning parameters."""
    return AutotuningParameters(
        graph_memory_overhead_base_ratio=0.1,
        graph_memory_batch_scaling_factor=0.02,
        min_gpu_memory_utilization=0.8,
        max_gpu_memory_utilization=0.95,
        throughput_batch_weight=0.7,
        throughput_graph_weight=0.3,
    )


@pytest.fixture
def chatbot_workload():
    """Create a chatbot workload profile."""
    return WorkloadProfile.create_synthetic("chatbot", requests_per_second=100)


@pytest.fixture
def batch_workload():
    """Create a batch inference workload profile."""
    return WorkloadProfile.create_synthetic("batch_inference", requests_per_second=10)


class TestCalculateModelMemoryFromConfig:
    """Test model memory calculation from config."""

    def test_calculate_memory_fp16(self, small_model_config):
        """Test memory calculation for fp16 model."""
        memory_gb = calculate_model_memory_from_config(small_model_config)
        assert memory_gb > 0
        assert isinstance(memory_gb, float)
        # Should be reasonable for small model
        assert 0.1 < memory_gb < 2.0

    def test_calculate_memory_large_model(self, mock_llama_config):
        """Test memory calculation for larger model."""
        memory_gb = calculate_model_memory_from_config(mock_llama_config)
        assert memory_gb > 0
        # Should be larger than small model
        assert memory_gb > 5.0
        assert memory_gb < 50.0

    def test_calculate_memory_quantized(self, small_model_config):
        """Test memory calculation with quantization."""
        # Add quantization config
        small_model_config.quantization_config = {"bits": 8}
        memory_gb = calculate_model_memory_from_config(small_model_config)
        assert memory_gb > 0
        # Should be less than fp16 version
        assert memory_gb < 1.0


class TestEstimateActivationMemoryFromConfig:
    """Test activation memory estimation from config."""

    def test_estimate_activation_memory(self, small_model_config):
        """Test activation memory estimation."""
        memory_gb = estimate_activation_memory_from_config(
            small_model_config, batch_size=8, sequence_length=512
        )
        assert memory_gb > 0
        assert isinstance(memory_gb, float)
        # Should be reasonable for small model and batch
        assert 0.01 < memory_gb < 5.0

    def test_activation_memory_scales_with_batch(self, small_model_config):
        """Test that activation memory scales with batch size."""
        memory_small = estimate_activation_memory_from_config(
            small_model_config, batch_size=4, sequence_length=512
        )
        memory_large = estimate_activation_memory_from_config(
            small_model_config, batch_size=16, sequence_length=512
        )
        assert memory_large > memory_small

    def test_activation_memory_scales_with_sequence(self, small_model_config):
        """Test that activation memory scales with sequence length."""
        memory_short = estimate_activation_memory_from_config(
            small_model_config, batch_size=8, sequence_length=256
        )
        memory_long = estimate_activation_memory_from_config(
            small_model_config, batch_size=8, sequence_length=1024
        )
        assert memory_long > memory_short


class TestAutotuningParameters:
    """Test autotuning parameters."""

    def test_default_values(self):
        """Test default parameter values."""
        params = AutotuningParameters()
        assert params.graph_memory_overhead_base_ratio == 0.1
        assert params.graph_memory_batch_scaling_factor == 0.02
        assert params.compilation_level == "PIECEWISE"
        assert params.min_gpu_memory_utilization == 0.8
        assert params.max_gpu_memory_utilization == 0.98

    def test_custom_values(self):
        """Test custom parameter values."""
        params = AutotuningParameters(
            graph_memory_overhead_base_ratio=0.15,
            compilation_level="FULL",
            min_gpu_memory_utilization=0.7,
        )
        assert params.graph_memory_overhead_base_ratio == 0.15
        assert params.compilation_level == "FULL"
        assert params.min_gpu_memory_utilization == 0.7


class TestVLLMPerformanceModel:
    """Test vLLM performance model."""

    def test_from_transformers_config(self, small_model_config, autotuning_params):
        """Test creating performance model from transformers config."""
        model = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        assert model.gpu_memory_capacity_gb == 24.0
        assert model.gpu_memory_utilization == 0.9
        assert model.max_model_len == 2048
        assert model.hidden_size == 768
        assert model.num_layers == 12
        assert model.model_memory_gb > 0
        assert model.activation_memory_gb > 0

    def test_calculate_memory_breakdown(self, small_model_config, autotuning_params):
        """Test memory breakdown calculation."""
        model = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )

        breakdown = model.calculate_memory_breakdown()
        assert "model_memory" in breakdown
        assert "activation_memory" in breakdown
        assert "cuda_graph_memory" in breakdown
        assert "kv_cache_memory" in breakdown
        assert "total_used" in breakdown
        assert "utilization_ratio" in breakdown

        # All values should be positive
        assert breakdown["model_memory"] > 0
        assert breakdown["activation_memory"] > 0
        assert breakdown["cuda_graph_memory"] > 0
        assert breakdown["total_used"] > 0
        assert 0 < breakdown["utilization_ratio"] <= 1

    def test_calculate_effective_batch_size(self, small_model_config, autotuning_params):
        """Test effective batch size calculation."""
        model = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        batch_size = model.calculate_effective_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size >= 0

    def test_calculate_graph_coverage(self, small_model_config, autotuning_params, chatbot_workload):
        """Test graph coverage calculation."""
        model = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )

        coverage = model.calculate_graph_coverage(chatbot_workload)
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_memory_breakdown_no_cuda_graphs(self, small_model_config, autotuning_params):
        """Test memory breakdown with no CUDA graphs."""
        model = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
            cudagraph_capture_sizes=[],
        )

        breakdown = model.calculate_memory_breakdown()
        assert breakdown["cuda_graph_memory"] == 0.0
        # Should have more KV cache memory without CUDA graphs
        assert breakdown["kv_cache_memory"] > 0


class TestVLLMConfigOptimizer:
    """Test vLLM configuration optimizer."""

    def test_init_with_valid_model(self, autotuning_params):
        """Test initializing optimizer with valid model."""
        # Using a standard model that should be available
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )
        assert optimizer.model_name == "microsoft/DialoGPT-small"
        assert optimizer.gpu_memory_capacity_gb == 24.0
        assert optimizer.tuning_params == autotuning_params

    def test_get_default_search_space(self, autotuning_params, chatbot_workload):
        """Test default search space generation."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        search_space = optimizer.get_default_search_space(chatbot_workload)
        assert "gpu_memory_utilization" in search_space
        assert "cudagraph_capture_sizes" in search_space
        assert "max_model_len" in search_space
        assert "kv_cache_dtype" in search_space

        # Check reasonable values
        assert len(search_space["gpu_memory_utilization"]) > 0
        assert len(search_space["cudagraph_capture_sizes"]) > 0
        assert len(search_space["max_model_len"]) > 0

    def test_generate_configs(self, autotuning_params):
        """Test configuration generation."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        search_space = {
            "gpu_memory_utilization": [0.8, 0.9],
            "max_model_len": [1024, 2048],
            "kv_cache_dtype": ["auto"],
        }

        configs = list(optimizer.generate_configs(search_space))
        assert len(configs) == 4  # 2 * 2 * 1 combinations

        # Check all configs have required keys
        for config in configs:
            assert "gpu_memory_utilization" in config
            assert "max_model_len" in config
            assert "kv_cache_dtype" in config

    def test_is_feasible_config(self, small_model_config, autotuning_params):
        """Test configuration feasibility check."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        # Create a feasible config
        feasible_config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.8,
            max_model_len=1024,
            tuning_params=autotuning_params,
        )

        assert optimizer.is_feasible_config(feasible_config)

        # Create an infeasible config (too high memory utilization)
        infeasible_config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=2.0,  # Too small
            gpu_memory_utilization=0.99,
            max_model_len=8192,  # Too large
            tuning_params=autotuning_params,
        )

        assert not optimizer.is_feasible_config(infeasible_config)

    def test_evaluate_config_throughput(self, small_model_config, autotuning_params, batch_workload):
        """Test configuration evaluation for throughput."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        score = optimizer.evaluate_config(config, batch_workload)
        assert isinstance(score, float)
        assert score >= 0

    def test_evaluate_config_latency(self, small_model_config, autotuning_params, chatbot_workload):
        """Test configuration evaluation for latency."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        score = optimizer.evaluate_config(config, chatbot_workload)
        assert isinstance(score, float)
        assert score >= 0

    def test_get_config_predictions(self, small_model_config, autotuning_params, chatbot_workload):
        """Test configuration predictions."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        predictions = optimizer.get_config_predictions(config, chatbot_workload)
        assert "effective_batch_size" in predictions
        assert "graph_coverage" in predictions
        assert "memory_breakdown" in predictions
        assert "recommended_max_num_seqs" in predictions

        assert isinstance(predictions["effective_batch_size"], int)
        assert isinstance(predictions["graph_coverage"], float)
        assert isinstance(predictions["memory_breakdown"], dict)
        assert isinstance(predictions["recommended_max_num_seqs"], int)

    def test_validate_configuration(self, small_model_config, autotuning_params):
        """Test configuration validation."""
        optimizer = vLLMConfigOptimizer(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            tuning_params=autotuning_params,
        )

        config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        validation = optimizer.validate_configuration(config)
        assert "valid" in validation
        assert "warnings" in validation
        assert "recommendations" in validation
        assert "effective_batch_size" in validation
        assert "memory_breakdown" in validation

        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["recommendations"], list)


class TestOptimizeVLLMConfigForCluster:
    """Test cluster-wide vLLM configuration optimization."""

    def test_optimize_for_cluster(self, chatbot_workload, autotuning_params):
        """Test cluster optimization."""
        parallelism_strategy = {"tp": 2, "pp": 1, "dp": 4}

        result = optimize_vllm_config_for_cluster(
            model_name="microsoft/DialoGPT-small",
            gpu_memory_capacity_gb=24.0,
            workload=chatbot_workload,
            parallelism_strategy=parallelism_strategy,
            tuning_params=autotuning_params,
        )

        assert "vllm_config" in result
        assert "parallelism_strategy" in result
        assert "cluster_predictions" in result
        assert "recommendations" in result
        assert "optimization_results" in result

        assert result["parallelism_strategy"] == parallelism_strategy

        # Check cluster predictions
        cluster_pred = result["cluster_predictions"]
        assert "total_throughput" in cluster_pred
        assert "instances_per_cluster" in cluster_pred
        assert "memory_efficiency" in cluster_pred
        assert "graph_coverage" in cluster_pred

        assert cluster_pred["instances_per_cluster"] == parallelism_strategy["dp"]


class TestGenerateDeploymentRecommendations:
    """Test deployment recommendations generation."""

    def test_generate_recommendations_valid_config(self, small_model_config, autotuning_params):
        """Test recommendations for valid configuration."""
        config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=24.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
        )

        optimal_result = {
            "optimal_config": config,
            "memory_breakdown": config.calculate_memory_breakdown(),
            "predictions": {
                "effective_batch_size": 16,
                "graph_coverage": 0.8,
            },
        }

        parallelism_strategy = {"tp": 2, "pp": 1, "dp": 4}

        recommendations = generate_deployment_recommendations(optimal_result, parallelism_strategy)
        assert isinstance(recommendations, list)

    def test_generate_recommendations_no_config(self):
        """Test recommendations when no feasible config found."""
        optimal_result = {
            "optimal_config": None,
            "memory_breakdown": None,
            "predictions": None,
        }

        parallelism_strategy = {"tp": 1, "pp": 1, "dp": 1}

        recommendations = generate_deployment_recommendations(optimal_result, parallelism_strategy)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert "No feasible configuration found" in recommendations[0]

    def test_generate_recommendations_low_kv_cache(self, small_model_config, autotuning_params):
        """Test recommendations for low KV cache ratio."""
        config = vLLMPerformanceModel.from_transformers_config(
            config=small_model_config,
            gpu_memory_capacity_gb=4.0,  # Small memory to trigger low KV cache
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            tuning_params=autotuning_params,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],  # Many captures
        )

        memory_breakdown = config.calculate_memory_breakdown()
        optimal_result = {
            "optimal_config": config,
            "memory_breakdown": memory_breakdown,
            "predictions": {
                "effective_batch_size": 2,
                "graph_coverage": 0.3,
            },
        }

        parallelism_strategy = {"tp": 1, "pp": 1, "dp": 1}

        recommendations = generate_deployment_recommendations(optimal_result, parallelism_strategy)
        assert isinstance(recommendations, list)

        # Should contain recommendations about KV cache and graph coverage
        rec_text = " ".join(recommendations)
        if memory_breakdown["kv_cache_memory"] / config.gpu_memory_capacity_gb < 0.1:
            assert "KV cache" in rec_text or "memory" in rec_text
