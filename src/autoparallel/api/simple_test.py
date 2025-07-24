"""Tests for the simple public API."""

import pytest
from unittest.mock import Mock, patch

from autoparallel.api.simple import (
    AnalysisResult,
    OptimizedConfig,
    analyze,
    optimize,
    _convert_cluster_dict_to_hardware_profile,
    _convert_workload_string_to_profile,
)
from autoparallel.config.optimizer import HardwareProfile, WorkloadProfile, WorkloadType
from autoparallel.config.generator import ConfigurationGenerationResult, ScoredConfiguration
from autoparallel.config.optimizer import ParallelismConfiguration, PerformanceMetrics
from autoparallel.memory.components import MemoryComponents
from autoparallel.constraints.analyzer import ModelConstraints


class TestClusterConversion:
    """Test cluster dictionary to HardwareProfile conversion."""

    def test_convert_cluster_dict_basic(self):
        """Test basic cluster dictionary conversion."""
        cluster = {
            "gpu_memory_gb": 80,
            "gpus_per_node": 8,
            "num_nodes": 2,
        }

        profile = _convert_cluster_dict_to_hardware_profile(cluster)

        assert isinstance(profile, HardwareProfile)
        assert profile.gpu_memory_gb == 80
        assert profile.gpus_per_node == 8
        assert profile.num_nodes == 2
        assert profile.gpu_model == "H100"  # Default

    def test_convert_cluster_dict_with_optional_fields(self):
        """Test cluster conversion with optional fields."""
        cluster = {
            "gpu_memory_gb": 40,
            "gpus_per_node": 4,
            "num_nodes": 1,
            "gpu_model": "A100",
            "intra_node_bandwidth_gbps": 600.0,
            "inter_node_bandwidth_gbps": 100.0,
            "network_topology": "mesh",
        }

        profile = _convert_cluster_dict_to_hardware_profile(cluster)

        assert profile.gpu_memory_gb == 40
        assert profile.gpus_per_node == 4
        assert profile.num_nodes == 1
        assert profile.gpu_model == "A100"
        assert profile.intra_node_bandwidth_gbps == 600.0
        assert profile.inter_node_bandwidth_gbps == 100.0
        assert profile.network_topology == "mesh"

    def test_convert_cluster_dict_defaults(self):
        """Test conversion with missing fields uses defaults."""
        cluster = {"gpu_memory_gb": 24}

        profile = _convert_cluster_dict_to_hardware_profile(cluster)

        assert profile.gpu_memory_gb == 24
        assert profile.gpus_per_node == 8  # Default
        assert profile.num_nodes == 1  # Default


class TestWorkloadConversion:
    """Test workload string to WorkloadProfile conversion."""

    def test_convert_workload_inference(self):
        """Test inference workload conversion."""
        profile = _convert_workload_string_to_profile("inference")

        assert profile.workload_type == WorkloadType.INFERENCE
        assert profile.batch_size == 32
        assert profile.sequence_length == 2048
        assert not profile.is_training

    def test_convert_workload_training(self):
        """Test training workload conversion."""
        profile = _convert_workload_string_to_profile("training")

        assert profile.workload_type == WorkloadType.TRAINING
        assert profile.batch_size == 64
        assert profile.sequence_length == 4096
        assert profile.is_training

    def test_convert_workload_chatbot(self):
        """Test chatbot workload conversion."""
        profile = _convert_workload_string_to_profile("chatbot")

        assert profile.workload_type == WorkloadType.CHATBOT
        assert profile.batch_size == 16
        assert profile.sequence_length == 1024
        assert not profile.is_training

    def test_convert_workload_case_insensitive(self):
        """Test workload conversion is case insensitive."""
        profile = _convert_workload_string_to_profile("INFERENCE")
        assert profile.workload_type == WorkloadType.INFERENCE

        profile = _convert_workload_string_to_profile("Training")
        assert profile.workload_type == WorkloadType.TRAINING

    def test_convert_workload_unknown_defaults_to_inference(self):
        """Test unknown workload defaults to inference."""
        profile = _convert_workload_string_to_profile("unknown_workload")
        assert profile.workload_type == WorkloadType.INFERENCE


class TestAnalysisResult:
    """Test AnalysisResult data class."""

    def test_analysis_result_creation(self):
        """Test AnalysisResult creation and validation."""
        configurations = [{"tensor_parallel_size": 1}, {"tensor_parallel_size": 2}]
        
        result = AnalysisResult(
            model_name="test-model",
            cluster_info={"gpu_memory_gb": 80},
            total_configurations=2,
            configurations=configurations,
            recommendations={"best": "config1"},
        )

        assert result.model_name == "test-model"
        assert result.total_configurations == 2
        assert len(result.configurations) == 2

    def test_analysis_result_validation_error(self):
        """Test AnalysisResult validation catches mismatches."""
        configurations = [{"tensor_parallel_size": 1}]
        
        with pytest.raises(ValueError, match="Mismatch between total_configurations"):
            AnalysisResult(
                model_name="test-model",
                cluster_info={"gpu_memory_gb": 80},
                total_configurations=2,  # Mismatch: should be 1
                configurations=configurations,
                recommendations={},
            )


class TestOptimizedConfig:
    """Test OptimizedConfig data class."""

    def test_optimized_config_creation(self):
        """Test OptimizedConfig creation."""
        best_config = {"tensor_parallel_size": 4}
        
        result = OptimizedConfig(
            model_name="test-model",
            workload_type="inference",
            best_configuration=best_config,
            performance_estimate={"throughput_tokens_per_second": 1000},
            deployment_command="python -m vllm",
            alternative_configurations=[],
        )

        assert result.model_name == "test-model"
        assert result.workload_type == "inference"
        assert result.best_configuration == best_config


class TestAnalyzeFunction:
    """Test the analyze() function."""

    def test_analyze_missing_transformers(self):
        """Test analyze function when transformers is not available."""
        with patch("autoparallel.api.simple.transformers", None):
            with pytest.raises(ImportError, match="transformers library is required"):
                analyze("test-model", {"gpu_memory_gb": 80, "gpus_per_node": 8})

    @patch("autoparallel.api.simple.ConfigurationGenerator")
    @patch("autoparallel.api.simple.analyze_model_constraints")
    @patch("autoparallel.api.simple._load_model_config")
    def test_analyze_basic_flow(self, mock_load_config, mock_analyze_constraints, mock_generator_class):
        """Test basic analyze function flow."""
        # Mock model config
        mock_config = Mock()
        mock_config.to_dict.return_value = {"hidden_size": 768}
        mock_load_config.return_value = mock_config

        # Mock constraints
        mock_constraints = Mock(spec=ModelConstraints)
        mock_constraints.max_tensor_parallel = 8
        mock_constraints.max_pipeline_parallel = 4
        mock_constraints.max_expert_parallel = 1
        mock_constraints.supports_grouped_query_attention = False
        mock_analyze_constraints.return_value = mock_constraints

        # Mock configuration generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock generation result
        scored_config = Mock(spec=ScoredConfiguration)
        scored_config.configuration = Mock(spec=ParallelismConfiguration)
        scored_config.configuration.tensor_parallel_size = 2
        scored_config.configuration.pipeline_parallel_size = 1
        scored_config.configuration.expert_parallel_size = 1
        scored_config.configuration.data_parallel_size = 1
        scored_config.configuration.total_gpus_needed = 2
        scored_config.performance_metrics = Mock(spec=PerformanceMetrics)
        scored_config.performance_metrics.throughput_score = 0.8
        scored_config.performance_metrics.latency_score = 0.7
        scored_config.performance_metrics.memory_utilization_gb_per_gpu = 60.0
        scored_config.performance_metrics.cost_score = 0.6
        scored_config.performance_metrics.communication_efficiency = 0.9
        scored_config.performance_metrics.is_feasible = True
        scored_config.memory_components = Mock(spec=MemoryComponents)
        scored_config.memory_components.total_memory = 100 * 1024**3  # 100GB in bytes
        scored_config.memory_components.weights = 50 * 1024**3
        scored_config.memory_components.activations = 30 * 1024**3
        scored_config.memory_components.kv_cache = 15 * 1024**3
        scored_config.memory_components.cuda_graphs = 3 * 1024**3
        scored_config.memory_components.optimizer_states = 0
        scored_config.memory_components.fragmentation_overhead = 2 * 1024**3
        scored_config.is_valid = True

        mock_result = Mock(spec=ConfigurationGenerationResult)
        mock_result.configurations = [scored_config]
        mock_generator.generate_valid_configs.return_value = mock_result

        # Call analyze
        cluster = {"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 1}
        result = analyze("test-model", cluster)

        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.model_name == "test-model"
        assert result.cluster_info == cluster
        assert result.total_configurations == 1
        assert len(result.configurations) == 1

        # Verify configuration structure
        config = result.configurations[0]
        assert config["tensor_parallel_size"] == 2
        assert config["total_gpus"] == 2
        assert "performance" in config
        assert "memory_breakdown" in config
        assert config["is_valid"] is True

        # Verify calls
        mock_load_config.assert_called_once_with("test-model")
        mock_analyze_constraints.assert_called_once()
        mock_generator.generate_valid_configs.assert_called_once()


class TestOptimizeFunction:
    """Test the optimize() function."""

    @patch("autoparallel.api.simple.ConfigurationGenerator")
    @patch("autoparallel.api.simple.analyze_model_constraints")
    @patch("autoparallel.api.simple._load_model_config")
    def test_optimize_basic_flow(self, mock_load_config, mock_analyze_constraints, mock_generator_class):
        """Test basic optimize function flow."""
        # Mock model config
        mock_config = Mock()
        mock_config.to_dict.return_value = {"hidden_size": 768}
        mock_load_config.return_value = mock_config

        # Mock constraints
        mock_constraints = Mock(spec=ModelConstraints)
        mock_analyze_constraints.return_value = mock_constraints

        # Mock configuration generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock generation result with valid configuration
        scored_config = Mock(spec=ScoredConfiguration)
        scored_config.configuration = Mock(spec=ParallelismConfiguration)
        scored_config.configuration.tensor_parallel_size = 4
        scored_config.configuration.pipeline_parallel_size = 1
        scored_config.configuration.expert_parallel_size = 1
        scored_config.configuration.data_parallel_size = 1
        scored_config.configuration.total_gpus_needed = 4
        scored_config.performance_metrics = Mock(spec=PerformanceMetrics)
        scored_config.performance_metrics.throughput_score = 0.9
        scored_config.performance_metrics.latency_score = 0.8
        scored_config.performance_metrics.memory_utilization_gb_per_gpu = 70.0
        scored_config.performance_metrics.cost_score = 0.7
        scored_config.performance_metrics.communication_efficiency = 0.95
        scored_config.performance_metrics.is_feasible = True
        scored_config.memory_components = Mock(spec=MemoryComponents)
        scored_config.memory_components.total_memory = 200 * 1024**3  # 200GB in bytes
        scored_config.memory_components.weights = 100 * 1024**3
        scored_config.memory_components.activations = 60 * 1024**3
        scored_config.memory_components.kv_cache = 30 * 1024**3
        scored_config.memory_components.cuda_graphs = 6 * 1024**3
        scored_config.memory_components.optimizer_states = 0
        scored_config.memory_components.fragmentation_overhead = 4 * 1024**3
        scored_config.is_valid = True

        mock_result = Mock(spec=ConfigurationGenerationResult)
        mock_result.configurations = [scored_config]
        mock_generator.generate_valid_configs.return_value = mock_result

        # Call optimize
        cluster = {"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 1}
        result = optimize("test-model", cluster, "inference")

        # Verify result structure
        assert isinstance(result, OptimizedConfig)
        assert result.model_name == "test-model"
        assert result.workload_type == "inference"
        assert "best_configuration" in result.__dict__
        assert "performance_estimate" in result.__dict__
        assert "deployment_command" in result.__dict__

        # Verify calls
        mock_load_config.assert_called_once_with("test-model")
        mock_analyze_constraints.assert_called_once()
        mock_generator.generate_valid_configs.assert_called_once()

    @patch("autoparallel.api.simple.ConfigurationGenerator")
    @patch("autoparallel.api.simple.analyze_model_constraints")
    @patch("autoparallel.api.simple._load_model_config")
    def test_optimize_no_valid_configurations(self, mock_load_config, mock_analyze_constraints, mock_generator_class):
        """Test optimize function when no valid configurations are found."""
        # Mock model config
        mock_config = Mock()
        mock_config.to_dict.return_value = {"hidden_size": 768}
        mock_load_config.return_value = mock_config

        # Mock constraints
        mock_constraints = Mock(spec=ModelConstraints)
        mock_analyze_constraints.return_value = mock_constraints

        # Mock configuration generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock empty result
        mock_result = Mock(spec=ConfigurationGenerationResult)
        mock_result.configurations = []  # No valid configurations
        mock_generator.generate_valid_configs.return_value = mock_result

        # Call optimize should raise error
        cluster = {"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 1}
        with pytest.raises(RuntimeError, match="No valid configurations found"):
            optimize("test-model", cluster, "inference")
