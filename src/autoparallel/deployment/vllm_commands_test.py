"""Tests for vLLM deployment command generation."""

from unittest.mock import Mock, patch

import pytest

from autoparallel.config.optimizer import (
    HardwareProfile,
    ParallelismConfiguration,
    WorkloadProfile,
    WorkloadType,
)
from autoparallel.deployment.vllm_commands import (
    VLLMCommandGenerator,
    VLLMDeploymentOptions,
    VLLMDeploymentResult,
    generate_vllm_deployment_command,
    generate_vllm_offline_command,
)
from autoparallel.frameworks.vllm_optimizer import GPUArchitecture


class TestVLLMCommandGenerator:
    """Test vLLM command generator functionality."""

    @pytest.fixture
    def hardware_profile(self):
        """Test hardware profile."""
        return HardwareProfile(
            gpu_model="A100",
            gpu_memory_gb=80.0,
            gpus_per_node=4,
            num_nodes=1,
            intra_node_bandwidth_gbps=900.0,
            inter_node_bandwidth_gbps=200.0,
        )

    @pytest.fixture
    def workload_profile(self):
        """Test workload profile."""
        return WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=32,
            sequence_length=512,
        )

    @pytest.fixture
    def parallelism_config(self):
        """Test parallelism configuration."""
        return ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            data_parallel_size=1,
        )

    @pytest.fixture
    def mock_optimizer(self):
        """Mock VLLMOptimizer for testing."""
        with patch("autoparallel.deployment.vllm_commands.VLLMOptimizer") as mock:
            # Mock optimization result
            mock_result = Mock()
            mock_result.is_successful = True
            mock_result.optimal_config = Mock()
            mock_result.optimal_config.gpu_memory_utilization = 0.9
            mock_result.optimal_config.max_model_len = 4096
            mock_result.optimal_config.calculate_effective_batch_size.return_value = 64
            mock_result.optimal_config.kv_cache_dtype = "auto"
            mock_result.optimal_config.cudagraph_capture_sizes = [8, 16, 32]
            mock_result.recommendations = ["Use chunked prefill for better latency"]
            mock_result.predictions = {"throughput": 1000.0, "latency": 50.0}
            mock_result.memory_efficiency = 0.85
            mock_result.performance_score = 95.0

            mock.return_value.optimize_for_workload.return_value = mock_result
            yield mock

    def test_generator_initialization(self, hardware_profile):
        """Test VLLMCommandGenerator initialization."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        assert generator.model_name == "meta-llama/Llama-2-7b-hf"
        assert generator.hardware_profile == hardware_profile
        assert generator.gpu_architecture == GPUArchitecture.A100

    def test_generate_serving_command_basic(
        self, mock_optimizer, hardware_profile, workload_profile, parallelism_config
    ):
        """Test basic serving command generation."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        result = generator.generate_serving_command(
            workload=workload_profile,
            parallelism_config=parallelism_config,
        )

        assert isinstance(result, VLLMDeploymentResult)
        assert "python -m vllm.entrypoints.openai.api_server" in result.command
        assert "meta-llama/Llama-2-7b-hf" in result.command
        assert "--tensor-parallel-size 2" in result.command
        assert "--pipeline-parallel-size 2" in result.command
        assert "--host 0.0.0.0" in result.command
        assert "--port 8000" in result.command

    def test_generate_serving_command_with_options(
        self, mock_optimizer, hardware_profile, workload_profile, parallelism_config
    ):
        """Test serving command generation with custom options."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.H100,
        )

        options = VLLMDeploymentOptions(
            host="127.0.0.1",
            port=9000,
            enable_chunked_prefill=True,
            disable_cuda_graphs=True,
            custom_args={"max-num-batched-tokens": 16384},
        )

        result = generator.generate_serving_command(
            workload=workload_profile,
            parallelism_config=parallelism_config,
            options=options,
        )

        assert "--host 127.0.0.1" in result.command
        assert "--port 9000" in result.command
        assert "--enforce-eager" in result.command
        assert "--max-num-batched-tokens 16384" in result.command

    def test_generate_offline_command(
        self, mock_optimizer, hardware_profile, parallelism_config
    ):
        """Test offline inference command generation."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        result = generator.generate_offline_command(
            input_file="/path/to/input.jsonl",
            output_file="/path/to/output.jsonl",
            parallelism_config=parallelism_config,
            batch_size=500,
        )

        assert isinstance(result, VLLMDeploymentResult)
        assert "python -m vllm.entrypoints.offline_inference" in result.command
        assert "--input /path/to/input.jsonl" in result.command
        assert "--output /path/to/output.jsonl" in result.command
        assert "--batch-size 500" in result.command

    def test_architecture_specific_optimizations(
        self, mock_optimizer, hardware_profile, workload_profile, parallelism_config
    ):
        """Test GPU architecture-specific optimizations."""
        # Test H100 optimizations
        generator_h100 = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.H100,
        )

        # Mock config with fp8
        mock_result = mock_optimizer.return_value.optimize_for_workload.return_value
        mock_result.optimal_config.kv_cache_dtype = "fp8_e4m3"

        options = VLLMDeploymentOptions(enable_chunked_prefill=True)
        result = generator_h100.generate_serving_command(
            workload=workload_profile,
            parallelism_config=parallelism_config,
            options=options,
        )

        assert "--quantization fp8" in result.command
        assert "--enable-chunked-prefill" in result.command

    def test_workload_specific_optimizations(
        self, mock_optimizer, hardware_profile, parallelism_config
    ):
        """Test workload-specific optimizations."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        # Test latency workload
        latency_workload = WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=8,
            sequence_length=256,
            latency_budget_ms=50.0,  # Low latency requirement
        )

        result = generator.generate_serving_command(
            workload=latency_workload,
            parallelism_config=parallelism_config,
        )

        assert "--max-num-batched-tokens 8192" in result.command

    def test_environment_variables(
        self, mock_optimizer, hardware_profile, workload_profile, parallelism_config
    ):
        """Test environment variable generation."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        options = VLLMDeploymentOptions(environment_vars={"NCCL_DEBUG": "INFO"})

        result = generator.generate_serving_command(
            workload=workload_profile,
            parallelism_config=parallelism_config,
            options=options,
        )

        assert "CUDA_VISIBLE_DEVICES" in result.environment_vars
        assert result.environment_vars["NCCL_DEBUG"] == "INFO"

    def test_configuration_summary(
        self, mock_optimizer, hardware_profile, workload_profile, parallelism_config
    ):
        """Test configuration summary generation."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        result = generator.generate_serving_command(
            workload=workload_profile,
            parallelism_config=parallelism_config,
        )

        config_summary = result.configuration_summary
        assert config_summary["model_name"] == "meta-llama/Llama-2-7b-hf"
        assert config_summary["gpu_architecture"] == "A100"
        assert config_summary["parallelism"]["tensor_parallel"] == 2
        assert config_summary["parallelism"]["pipeline_parallel"] == 2
        assert config_summary["deployment_mode"] == "serving"

    def test_performance_estimates(
        self, mock_optimizer, hardware_profile, workload_profile, parallelism_config
    ):
        """Test performance estimate extraction."""
        generator = VLLMCommandGenerator(
            model_name="meta-llama/Llama-2-7b-hf",
            hardware_profile=hardware_profile,
            gpu_architecture=GPUArchitecture.A100,
        )

        result = generator.generate_serving_command(
            workload=workload_profile,
            parallelism_config=parallelism_config,
        )

        performance = result.estimated_performance
        assert performance is not None
        assert performance["throughput_tokens_per_second"] == 1000.0
        assert performance["latency_ms"] == 50.0
        assert performance["memory_efficiency"] == 0.85
        assert performance["performance_score"] == 95.0

    def test_optimization_failure_fallback(
        self, hardware_profile, workload_profile, parallelism_config
    ):
        """Test fallback behavior when optimization fails."""
        with patch("autoparallel.deployment.vllm_commands.VLLMOptimizer") as mock:
            # Mock optimization failure
            mock_result = Mock()
            mock_result.is_successful = False
            mock.return_value.optimize_for_workload.return_value = mock_result

            generator = VLLMCommandGenerator(
                model_name="meta-llama/Llama-2-7b-hf",
                hardware_profile=hardware_profile,
                gpu_architecture=GPUArchitecture.A100,
            )

            with pytest.raises(
                ValueError, match="Failed to find optimal configuration"
            ):
                generator.generate_serving_command(
                    workload=workload_profile,
                    parallelism_config=parallelism_config,
                )


class TestDeploymentFunctions:
    """Test top-level deployment functions."""

    @pytest.fixture
    def hardware_profile(self):
        """Test hardware profile."""
        return HardwareProfile(
            gpu_model="V100",
            gpu_memory_gb=40.0,
            gpus_per_node=2,
            num_nodes=1,
            intra_node_bandwidth_gbps=300.0,
            inter_node_bandwidth_gbps=100.0,
        )

    @pytest.fixture
    def workload_profile(self):
        """Test workload profile."""
        return WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=16,
            sequence_length=1024,
        )

    @pytest.fixture
    def parallelism_config(self):
        """Test parallelism configuration."""
        return ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=1,
        )

    def test_generate_vllm_deployment_command(
        self, hardware_profile, workload_profile, parallelism_config
    ):
        """Test top-level deployment command generation function."""
        with patch(
            "autoparallel.deployment.vllm_commands.VLLMCommandGenerator"
        ) as mock_generator:
            mock_result = VLLMDeploymentResult(
                command="test command",
                environment_vars={},
                configuration_summary={},
                optimization_insights=[],
            )
            mock_generator.return_value.generate_serving_command.return_value = (
                mock_result
            )

            result = generate_vllm_deployment_command(
                model_name="test-model",
                hardware_profile=hardware_profile,
                workload=workload_profile,
                parallelism_config=parallelism_config,
                gpu_architecture=GPUArchitecture.V100,
            )

            assert result == mock_result
            mock_generator.assert_called_once_with(
                model_name="test-model",
                hardware_profile=hardware_profile,
                gpu_architecture=GPUArchitecture.V100,
            )

    def test_generate_vllm_offline_command(self, hardware_profile, parallelism_config):
        """Test top-level offline command generation function."""
        with patch(
            "autoparallel.deployment.vllm_commands.VLLMCommandGenerator"
        ) as mock_generator:
            mock_result = VLLMDeploymentResult(
                command="test offline command",
                environment_vars={},
                configuration_summary={},
                optimization_insights=[],
            )
            mock_generator.return_value.generate_offline_command.return_value = (
                mock_result
            )

            result = generate_vllm_offline_command(
                model_name="test-model",
                hardware_profile=hardware_profile,
                parallelism_config=parallelism_config,
                input_file="input.jsonl",
                output_file="output.jsonl",
                batch_size=100,
                gpu_architecture=GPUArchitecture.RTX_4090,
            )

            assert result == mock_result
            mock_generator.assert_called_once_with(
                model_name="test-model",
                hardware_profile=hardware_profile,
                gpu_architecture=GPUArchitecture.RTX_4090,
            )


class TestVLLMDeploymentOptions:
    """Test VLLMDeploymentOptions dataclass."""

    def test_default_options(self):
        """Test default deployment options."""
        options = VLLMDeploymentOptions()

        assert options.host == "0.0.0.0"
        assert options.port == 8000
        assert options.enable_offline_mode is False
        assert options.serve_api is True
        assert options.enable_chunked_prefill is False
        assert options.enable_prefix_caching is True
        assert options.disable_cuda_graphs is False
        assert options.custom_args is None
        assert options.environment_vars is None

    def test_custom_options(self):
        """Test custom deployment options."""
        custom_args = {"max-num-seqs": 256}
        env_vars = {"CUDA_LAUNCH_BLOCKING": "1"}

        options = VLLMDeploymentOptions(
            host="localhost",
            port=9000,
            enable_offline_mode=True,
            serve_api=False,
            enable_chunked_prefill=True,
            disable_cuda_graphs=True,
            custom_args=custom_args,
            environment_vars=env_vars,
        )

        assert options.host == "localhost"
        assert options.port == 9000
        assert options.enable_offline_mode is True
        assert options.serve_api is False
        assert options.enable_chunked_prefill is True
        assert options.disable_cuda_graphs is True
        assert options.custom_args == custom_args
        assert options.environment_vars == env_vars


class TestVLLMDeploymentResult:
    """Test VLLMDeploymentResult dataclass."""

    def test_deployment_result_creation(self):
        """Test VLLMDeploymentResult creation."""
        result = VLLMDeploymentResult(
            command="test command",
            environment_vars={"GPU": "A100"},
            configuration_summary={"model": "test"},
            optimization_insights=["Use FP16"],
            estimated_performance={"throughput": 100.0},
        )

        assert result.command == "test command"
        assert result.environment_vars == {"GPU": "A100"}
        assert result.configuration_summary == {"model": "test"}
        assert result.optimization_insights == ["Use FP16"]
        assert result.estimated_performance == {"throughput": 100.0}

    def test_deployment_result_defaults(self):
        """Test VLLMDeploymentResult with default values."""
        result = VLLMDeploymentResult(
            command="test command",
            environment_vars={},
            configuration_summary={},
            optimization_insights=[],
        )

        assert result.estimated_performance is None
