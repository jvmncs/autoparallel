"""Tests for vLLM optimizer functionality."""

import pytest
from unittest.mock import Mock, patch

from autoparallel.frameworks.vllm_config import (
    AutotuningParameters,
    vLLMPerformanceModel,
)
from autoparallel.frameworks.vllm_memory import (
    WorkloadProfile,
    vLLMAutotuningParameters,
)
from autoparallel.frameworks.vllm_optimizer import (
    GPUArchitecture,
    GPUArchitectureSpec,
    OptimizationResult,
    VLLMOptimizer,
    optimize_vllm_for_deployment,
)


class TestGPUArchitectureSpec:
    """Tests for GPU architecture specifications."""

    def test_h100_spec(self):
        """Test H100 GPU architecture specification."""
        spec = GPUArchitectureSpec.from_architecture(GPUArchitecture.H100)

        assert spec.memory_bandwidth_gb_s == 3350.0
        assert spec.compute_capability == "9.0"
        assert spec.tensor_core_support is True
        assert spec.fp8_support is True
        assert spec.nvlink_bandwidth_gb_s == 900.0
        assert spec.recommended_memory_utilization == 0.95
        assert spec.cuda_graph_efficiency_multiplier == 1.2
        assert spec.fp8_speedup_factor == 1.6

    def test_a100_spec(self):
        """Test A100 GPU architecture specification."""
        spec = GPUArchitectureSpec.from_architecture(GPUArchitecture.A100)

        assert spec.memory_bandwidth_gb_s == 1935.0
        assert spec.compute_capability == "8.0"
        assert spec.tensor_core_support is True
        assert spec.fp8_support is False
        assert spec.nvlink_bandwidth_gb_s == 600.0
        assert spec.recommended_memory_utilization == 0.9
        assert spec.cuda_graph_efficiency_multiplier == 1.1

    def test_generic_fallback(self):
        """Test generic GPU architecture fallback."""
        spec = GPUArchitectureSpec.from_architecture(GPUArchitecture.GENERIC)

        assert spec.memory_bandwidth_gb_s == 1000.0
        assert spec.compute_capability == "7.0"
        assert spec.recommended_memory_utilization == 0.9


class TestOptimizationResult:
    """Tests for optimization result data structure."""

    def test_successful_result(self):
        """Test successful optimization result."""
        mock_config = Mock(spec=vLLMPerformanceModel)
        mock_config.calculate_effective_batch_size.return_value = 16

        result = OptimizationResult(
            optimal_config=mock_config,
            performance_score=0.8,
            memory_breakdown={"kv_cache_memory": 8.0, "total_used": 20.0},
            predictions={"effective_batch_size": 16},
            deployment_command="python -m vllm.entrypoints.openai.api_server --model test",
        )

        assert result.is_successful is True
        assert result.effective_batch_size == 16
        assert result.memory_efficiency == 0.4  # 8.0 / 20.0

    def test_failed_result(self):
        """Test failed optimization result."""
        result = OptimizationResult(
            optimal_config=None,
            performance_score=0.0,
            memory_breakdown=None,
            predictions=None,
        )

        assert result.is_successful is False
        assert result.effective_batch_size == 0
        assert result.memory_efficiency == 0.0


class TestVLLMOptimizer:
    """Tests for VLLMOptimizer class."""

    @pytest.fixture
    def mock_config(self):
        """Mock transformers config."""
        config = Mock()
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.vocab_size = 50257
        config.intermediate_size = 11008
        config.torch_dtype = "float16"
        return config

    @pytest.fixture
    def optimizer(self, mock_config):
        """Create VLLMOptimizer instance for testing."""
        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            return VLLMOptimizer(
                model_name="test-model",
                gpu_memory_capacity_gb=80.0,
                gpu_architecture=GPUArchitecture.H100,
            )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.model_name == "test-model"
        assert optimizer.gpu_memory_capacity_gb == 80.0
        assert optimizer.gpu_architecture == GPUArchitecture.H100
        assert isinstance(optimizer.tuning_params, AutotuningParameters)
        assert optimizer.config_optimizer is not None
        assert optimizer.memory_estimator is not None

    def test_architecture_tuning_adjustments(self, mock_config):
        """Test GPU architecture-specific tuning parameter adjustments."""
        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            # H100 should increase graph weights due to efficiency multiplier
            h100_optimizer = VLLMOptimizer(
                model_name="test-model",
                gpu_memory_capacity_gb=80.0,
                gpu_architecture=GPUArchitecture.H100,
            )

            # A100 should have lower max memory utilization
            a100_optimizer = VLLMOptimizer(
                model_name="test-model",
                gpu_memory_capacity_gb=80.0,
                gpu_architecture=GPUArchitecture.A100,
            )

            assert h100_optimizer.tuning_params.max_gpu_memory_utilization == 0.95
            assert a100_optimizer.tuning_params.max_gpu_memory_utilization == 0.9

    def test_create_tradeoff_search_space(self, optimizer):
        """Test creation of tradeoff-focused search space."""
        workload = WorkloadProfile.create_synthetic("chatbot")
        search_space = optimizer._create_tradeoff_search_space(workload)

        assert "gpu_memory_utilization" in search_space
        assert "cudagraph_capture_sizes" in search_space
        assert "max_model_len" in search_space
        assert "kv_cache_dtype" in search_space

        # Should have multiple capture size strategies
        assert len(search_space["cudagraph_capture_sizes"]) >= 4
        # Should include empty list for no graphs
        assert [] in search_space["cudagraph_capture_sizes"]

    def test_generate_deployment_command_basic(self, optimizer):
        """Test basic deployment command generation."""
        mock_config = Mock(spec=vLLMPerformanceModel)
        mock_config.gpu_memory_utilization = 0.9
        mock_config.max_model_len = 2048
        mock_config.kv_cache_dtype = "auto"
        mock_config.cudagraph_capture_sizes = [1, 2, 4, 8]
        mock_config.calculate_effective_batch_size.return_value = 16

        workload = WorkloadProfile.create_synthetic("chatbot")
        parallelism_strategy = {"tp": 1, "pp": 1, "dp": 1}

        command = optimizer.generate_deployment_command(
            mock_config, workload, parallelism_strategy
        )

        assert "python -m vllm.entrypoints.openai.api_server" in command
        assert "--model test-model" in command
        assert "--gpu-memory-utilization 0.9" in command
        assert "--max-model-len 2048" in command
        assert "--max-num-seqs" in command

    def test_generate_deployment_command_with_parallelism(self, optimizer):
        """Test deployment command generation with parallelism."""
        mock_config = Mock(spec=vLLMPerformanceModel)
        mock_config.gpu_memory_utilization = 0.9
        mock_config.max_model_len = 2048
        mock_config.kv_cache_dtype = "auto"
        mock_config.cudagraph_capture_sizes = [1, 2, 4]
        mock_config.calculate_effective_batch_size.return_value = 16

        workload = WorkloadProfile.create_synthetic("chatbot")
        parallelism_strategy = {"tp": 4, "pp": 2, "dp": 1}

        command = optimizer.generate_deployment_command(
            mock_config, workload, parallelism_strategy
        )

        assert "--tensor-parallel-size 4" in command
        assert "--pipeline-parallel-size 2" in command

    def test_generate_deployment_command_h100_optimizations(self, optimizer):
        """Test H100-specific deployment command optimizations."""
        mock_config = Mock(spec=vLLMPerformanceModel)
        mock_config.gpu_memory_utilization = 0.95
        mock_config.max_model_len = 4096
        mock_config.kv_cache_dtype = "fp8_e4m3"
        mock_config.cudagraph_capture_sizes = [1, 2, 4, 8, 16]
        mock_config.calculate_effective_batch_size.return_value = 32

        workload = WorkloadProfile.create_synthetic("batch_inference")
        parallelism_strategy = {"tp": 1, "pp": 1, "dp": 1}

        command = optimizer.generate_deployment_command(
            mock_config, workload, parallelism_strategy
        )

        assert "--kv-cache-dtype fp8_e4m3" in command
        assert "--quantization fp8" in command
        assert "--enable-chunked-prefill" in command

    def test_generate_deployment_command_with_custom_args(self, optimizer):
        """Test deployment command generation with custom arguments."""
        mock_config = Mock(spec=vLLMPerformanceModel)
        mock_config.gpu_memory_utilization = 0.9
        mock_config.max_model_len = 2048
        mock_config.kv_cache_dtype = "auto"
        mock_config.cudagraph_capture_sizes = []
        mock_config.calculate_effective_batch_size.return_value = 16

        workload = WorkloadProfile.create_synthetic("chatbot")
        parallelism_strategy = {"tp": 1, "pp": 1, "dp": 1}
        custom_args = {
            "host": "0.0.0.0",
            "port": 8000,
            "disable-log-requests": True,
        }

        command = optimizer.generate_deployment_command(
            mock_config, workload, parallelism_strategy, custom_args
        )

        assert "--host 0.0.0.0" in command
        assert "--port 8000" in command
        assert "--disable-log-requests" in command

    def test_optimize_kv_cache_vs_cuda_graphs_success(self, optimizer):
        """Test successful KV cache vs CUDA graphs optimization."""
        # Mock the optimizer's methods to avoid transformer config issues
        with patch.object(
            optimizer.config_optimizer, "generate_configs"
        ) as mock_generate:
            with patch.object(
                optimizer.config_optimizer, "is_feasible_config", return_value=True
            ):
                with patch.object(
                    optimizer.config_optimizer, "validate_configuration"
                ) as mock_validate:
                    with patch.object(
                        optimizer.config_optimizer, "get_config_predictions"
                    ) as mock_predictions:
                        # Create a properly mocked config
                        mock_config = Mock()
                        mock_config.calculate_memory_breakdown.return_value = {
                            "kv_cache_memory": 20.0,
                            "cuda_graph_memory": 5.0,
                            "model_memory": 40.0,
                            "total_used": 70.0,
                        }
                        mock_config.calculate_effective_batch_size.return_value = 24
                        mock_config.calculate_graph_coverage.return_value = 0.8
                        mock_config.gpu_memory_capacity_gb = 80.0
                        mock_config.cudagraph_capture_sizes = [1, 2, 4, 8]
                        mock_config.gpu_memory_utilization = 0.9
                        mock_config.max_model_len = 2048
                        mock_config.kv_cache_dtype = "auto"

                        # Mock the config generation to return one good config
                        mock_generate.return_value = [{}]  # Single empty config params

                        # Mock validation and predictions
                        mock_validate.return_value = {
                            "valid": True,
                            "warnings": [],
                            "recommendations": [],
                        }
                        mock_predictions.return_value = {
                            "effective_batch_size": 24,
                            "graph_coverage": 0.8,
                        }

                        # Mock the performance model creation
                        with patch(
                            "autoparallel.frameworks.vllm_optimizer.vLLMPerformanceModel.from_transformers_config",
                            return_value=mock_config,
                        ):
                            workload = WorkloadProfile.create_synthetic("chatbot")

                            result = optimizer.optimize_kv_cache_vs_cuda_graphs(
                                workload, memory_priority=0.6, performance_priority=0.4
                            )

                            assert result.is_successful is True
                            assert result.optimal_config is not None
                            assert result.deployment_command != ""
                            assert len(result.recommendations) > 0
                            assert (
                                result.optimization_metadata["optimization_type"]
                                == "kv_cache_vs_cuda_graphs"
                            )

    @patch(
        "autoparallel.frameworks.vllm_optimizer.vLLMConfigOptimizer.search_optimal_config"
    )
    def test_optimize_for_workload_success(self, mock_search, optimizer):
        """Test successful workload-specific optimization."""
        # Mock successful optimization result
        mock_config = Mock()
        mock_config.calculate_effective_batch_size.return_value = 16
        mock_config.calculate_graph_coverage.return_value = 0.7
        mock_config.gpu_memory_utilization = 0.9
        mock_config.max_model_len = 2048
        mock_config.kv_cache_dtype = "auto"
        mock_config.cudagraph_capture_sizes = [1, 2, 4, 8]
        mock_config.gpu_memory_capacity_gb = 80.0

        mock_search.return_value = {
            "optimal_config": mock_config,
            "performance_score": 0.85,
            "memory_breakdown": {"kv_cache_memory": 15.0, "total_used": 60.0},
            "predictions": {"effective_batch_size": 16, "graph_coverage": 0.7},
            "all_evaluated_configs": [(mock_config, 0.85)],
        }

        optimizer.config_optimizer.validate_configuration = Mock(
            return_value={
                "valid": True,
                "warnings": [],
                "recommendations": [],
            }
        )

        workload = WorkloadProfile.create_synthetic("batch_inference")

        result = optimizer.optimize_for_workload(
            workload, optimization_objective="throughput", max_configurations=50
        )

        assert result.is_successful is True
        assert result.performance_score == 0.85
        assert result.optimization_metadata["optimization_objective"] == "throughput"
        assert result.optimization_metadata["workload_type"] == "throughput"

    def test_optimize_for_workload_no_feasible_config(self, optimizer):
        """Test optimization when no feasible configuration is found."""
        # Mock no feasible configuration
        with patch.object(
            optimizer.config_optimizer, "search_optimal_config"
        ) as mock_search:
            mock_search.return_value = {
                "optimal_config": None,
                "performance_score": 0.0,
                "memory_breakdown": None,
                "predictions": None,
                "all_evaluated_configs": [],
            }

            workload = WorkloadProfile.create_synthetic("chatbot")

            result = optimizer.optimize_for_workload(workload)

            assert result.is_successful is False
            assert result.optimal_config is None
            assert "No feasible configuration found" in result.recommendations[0]

    def test_adjust_tuning_for_objective(self, optimizer):
        """Test tuning parameter adjustment for different objectives."""
        # Test latency optimization
        latency_params = optimizer._adjust_tuning_for_objective("latency")
        assert latency_params.latency_graph_weight == 0.9
        assert latency_params.latency_batch_weight == 0.1

        # Test throughput optimization
        throughput_params = optimizer._adjust_tuning_for_objective("throughput")
        assert throughput_params.throughput_batch_weight == 0.8
        assert throughput_params.throughput_graph_weight == 0.2

        # Test memory optimization
        memory_params = optimizer._adjust_tuning_for_objective("memory")
        assert memory_params.min_kv_cache_ratio == 0.1
        assert memory_params.max_gpu_memory_utilization <= 0.93

    def test_reduce_search_space(self, optimizer):
        """Test search space reduction for limited configurations."""
        workload = WorkloadProfile.create_synthetic("chatbot")
        full_search_space = optimizer.config_optimizer.get_default_search_space(
            workload
        )

        # Test very limited search
        limited_space = optimizer._reduce_search_space(full_search_space, 20)
        assert len(limited_space["gpu_memory_utilization"]) <= 3
        assert len(limited_space["cudagraph_capture_sizes"]) <= 2
        assert len(limited_space["kv_cache_dtype"]) <= 1

        # Test moderate search
        moderate_space = optimizer._reduce_search_space(full_search_space, 50)
        assert len(moderate_space["gpu_memory_utilization"]) <= 3
        assert len(moderate_space["cudagraph_capture_sizes"]) <= 3

    def test_generate_tradeoff_recommendations(self, optimizer):
        """Test generation of tradeoff-specific recommendations."""
        mock_config = Mock(spec=vLLMPerformanceModel)
        mock_config.calculate_memory_breakdown.return_value = {
            "kv_cache_memory": 25.0,  # High KV cache ratio (31.25%)
            "total_used": 80.0,
        }
        mock_config.cudagraph_capture_sizes = [1, 2, 4]
        mock_config.kv_cache_dtype = "auto"
        mock_config.gpu_memory_utilization = 0.9

        workload = WorkloadProfile.create_synthetic(
            "batch_inference"
        )  # Use valid workload type
        tradeoff_configs = [(mock_config, 0.8, {})]

        recommendations = optimizer._generate_tradeoff_recommendations(
            mock_config, workload, tradeoff_configs
        )

        assert len(recommendations) > 0
        # Should detect high KV cache prioritization
        assert any("prioritizes KV cache" in rec for rec in recommendations)
        # Should have H100-specific recommendations
        assert any("H100" in rec for rec in recommendations)

    def test_generate_comprehensive_recommendations(self, optimizer):
        """Test generation of comprehensive recommendations."""
        mock_config = Mock()
        mock_config.calculate_effective_batch_size.return_value = 32
        mock_config.calculate_graph_coverage.return_value = 0.2  # Low coverage
        mock_config.gpu_memory_utilization = 0.96  # High utilization
        mock_config.cudagraph_capture_sizes = [1, 2, 4]
        mock_config.gpu_memory_capacity_gb = 80.0  # Add missing attribute

        optimal_result = {
            "optimal_config": mock_config,
            "memory_breakdown": {"kv_cache_memory": 6.0},  # Low KV cache
            "predictions": {"effective_batch_size": 32, "graph_coverage": 0.2},
        }

        workload = WorkloadProfile.create_synthetic(
            "interactive"
        )  # Use valid workload type

        recommendations = optimizer._generate_comprehensive_recommendations(
            optimal_result, workload, "latency"
        )

        assert len(recommendations) > 0
        # Should have architecture-specific recommendations
        assert any("H100" in rec for rec in recommendations)


class TestOptimizeVLLMForDeployment:
    """Tests for high-level optimization function."""

    @patch("transformers.AutoConfig.from_pretrained")
    @patch("autoparallel.frameworks.vllm_optimizer.VLLMOptimizer.optimize_for_workload")
    def test_basic_optimization(self, mock_optimize, mock_config):
        """Test basic deployment optimization."""
        # Mock config
        config = Mock()
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.vocab_size = 50257
        config.torch_dtype = "float16"
        mock_config.return_value = config

        # Mock optimization result
        mock_result = Mock()
        mock_result.optimal_config = Mock()
        mock_result.optimal_config.gpu_memory_utilization = 0.9
        mock_result.optimal_config.max_model_len = 2048
        mock_result.optimal_config.kv_cache_dtype = "auto"
        mock_result.optimal_config.cudagraph_capture_sizes = [1, 2, 4]
        mock_result.optimal_config.calculate_effective_batch_size.return_value = 16
        mock_optimize.return_value = mock_result

        result = optimize_vllm_for_deployment(
            model_name="test-model",
            gpu_memory_capacity_gb=40.0,
            workload_type="chatbot",
            gpu_architecture="A100",
        )

        assert result is not None
        mock_optimize.assert_called_once()

    @patch("transformers.AutoConfig.from_pretrained")
    def test_custom_parameters(self, mock_config):
        """Test optimization with custom parameters."""
        # Mock config
        config = Mock()
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.vocab_size = 50257
        config.torch_dtype = "float16"
        mock_config.return_value = config

        custom_params = {
            "max_gpu_memory_utilization": 0.85,
            "throughput_batch_weight": 0.8,
        }

        with patch(
            "autoparallel.frameworks.vllm_optimizer.VLLMOptimizer"
        ) as mock_optimizer:
            mock_instance = Mock()
            mock_result = Mock()
            mock_result.optimal_config = Mock()
            mock_instance.optimize_for_workload.return_value = mock_result
            mock_instance.generate_deployment_command.return_value = "test command"
            mock_optimizer.return_value = mock_instance

            result = optimize_vllm_for_deployment(
                model_name="test-model",
                gpu_memory_capacity_gb=80.0,
                custom_tuning_params=custom_params,
            )

            # Verify custom parameters were passed
            args, kwargs = mock_optimizer.call_args
            assert kwargs["tuning_params"] is not None

    def test_invalid_gpu_architecture(self):
        """Test handling of invalid GPU architecture."""
        with patch("transformers.AutoConfig.from_pretrained"):
            with patch(
                "autoparallel.frameworks.vllm_optimizer.VLLMOptimizer"
            ) as mock_optimizer:
                mock_instance = Mock()
                mock_result = Mock()
                mock_result.optimal_config = Mock()
                mock_instance.optimize_for_workload.return_value = mock_result
                mock_optimizer.return_value = mock_instance

                # Should fallback to GENERIC for invalid architecture
                result = optimize_vllm_for_deployment(
                    model_name="test-model",
                    gpu_memory_capacity_gb=40.0,
                    gpu_architecture="INVALID_ARCH",
                )

                # Verify GENERIC architecture was used
                args, kwargs = mock_optimizer.call_args
                assert kwargs["gpu_architecture"] == GPUArchitecture.GENERIC


if __name__ == "__main__":
    pytest.main([__file__])
