"""Tests for the advanced public API."""

import pytest
from unittest.mock import Mock, patch

from autoparallel.api.advanced import (
    AutoParallel,
    Cluster,
    Workload,
    Preferences,
    OptimizationResult,
    DetailedConfiguration,
    AnalysisInsights,
)
from autoparallel.config.optimizer import HardwareProfile, WorkloadProfile, WorkloadType, OptimizationObjective
from autoparallel.memory.config import MemoryConfig


class TestCluster:
    """Test Cluster configuration class."""

    def test_cluster_creation(self):
        """Test basic Cluster creation."""
        cluster = Cluster(
            gpu_model="A100",
            gpu_memory_gb=40,
            gpus_per_node=4,
            num_nodes=2,
        )

        assert cluster.gpu_model == "A100"
        assert cluster.gpu_memory_gb == 40
        assert cluster.gpus_per_node == 4
        assert cluster.num_nodes == 2

    def test_cluster_defaults(self):
        """Test Cluster with default values."""
        cluster = Cluster()

        assert cluster.gpu_model == "H100"
        assert cluster.gpu_memory_gb == 80.0
        assert cluster.gpus_per_node == 8
        assert cluster.num_nodes == 1

    def test_cluster_from_dict(self):
        """Test Cluster.from_dict() method."""
        cluster_dict = {
            "gpu_model": "A100",
            "gpu_memory_gb": 40,
            "gpus_per_node": 4,
            "num_nodes": 2,
            "intra_node_bandwidth_gbps": 600.0,
        }

        cluster = Cluster.from_dict(cluster_dict)

        assert cluster.gpu_model == "A100"
        assert cluster.gpu_memory_gb == 40
        assert cluster.intra_node_bandwidth_gbps == 600.0

    def test_cluster_from_dict_partial(self):
        """Test Cluster.from_dict() with partial data."""
        cluster_dict = {"gpu_memory_gb": 24}

        cluster = Cluster.from_dict(cluster_dict)

        assert cluster.gpu_memory_gb == 24
        assert cluster.gpu_model == "H100"  # Default

    @patch.dict('os.environ', {'SLURM_GPUS_PER_NODE': '4', 'SLURM_NNODES': '2'})
    def test_cluster_from_slurm(self):
        """Test Cluster.from_slurm() method."""
        cluster = Cluster.from_slurm()

        assert cluster.gpus_per_node == 4
        assert cluster.num_nodes == 2
        assert cluster.gpu_model == "H100"  # Default

    @patch.dict('os.environ', {}, clear=True)
    def test_cluster_from_slurm_defaults(self):
        """Test Cluster.from_slurm() with no environment variables."""
        cluster = Cluster.from_slurm()

        assert cluster.gpus_per_node == 8  # Default
        assert cluster.num_nodes == 1  # Default

    def test_to_hardware_profile(self):
        """Test conversion to HardwareProfile."""
        cluster = Cluster(
            gpu_model="A100",
            gpu_memory_gb=40,
            gpus_per_node=4,
            num_nodes=2,
        )

        profile = cluster.to_hardware_profile()

        assert isinstance(profile, HardwareProfile)
        assert profile.gpu_model == "A100"
        assert profile.gpu_memory_gb == 40
        assert profile.gpus_per_node == 4
        assert profile.num_nodes == 2


class TestWorkload:
    """Test Workload configuration class."""

    def test_workload_creation(self):
        """Test basic Workload creation."""
        workload = Workload(
            workload_type="chatbot",
            batch_size=16,
            sequence_length=1024,
            requests_per_second=50.0,
        )

        assert workload.workload_type == "chatbot"
        assert workload.batch_size == 16
        assert workload.sequence_length == 1024
        assert workload.requests_per_second == 50.0

    def test_workload_defaults(self):
        """Test Workload with default values."""
        workload = Workload()

        assert workload.workload_type == "inference"
        assert workload.batch_size == 32
        assert workload.sequence_length == 2048
        assert workload.is_training is False

    def test_to_workload_profile(self):
        """Test conversion to WorkloadProfile."""
        workload = Workload(
            workload_type="training",
            batch_size=64,
            is_training=True,
        )

        profile = workload.to_workload_profile()

        assert isinstance(profile, WorkloadProfile)
        assert profile.workload_type == WorkloadType.TRAINING
        assert profile.batch_size == 64
        assert profile.is_training is True


class TestPreferences:
    """Test Preferences configuration class."""

    def test_preferences_creation(self):
        """Test basic Preferences creation."""
        prefs = Preferences(
            memory_conservatism="conservative",
            precision="fp16",
            framework="vllm",
        )

        assert prefs.memory_conservatism == "conservative"
        assert prefs.precision == "fp16"
        assert prefs.framework == "vllm"

    def test_preferences_defaults(self):
        """Test Preferences with default values."""
        prefs = Preferences()

        assert prefs.memory_conservatism == "moderate"
        assert prefs.precision == "auto"
        assert prefs.framework == "auto"
        assert prefs.max_configurations == 100

    def test_to_optimization_objective(self):
        """Test conversion to OptimizationObjective."""
        prefs = Preferences(optimization_objective="minimize_latency")

        objective = prefs.to_optimization_objective()

        assert objective == OptimizationObjective.MINIMIZE_LATENCY

    def test_to_optimization_objective_default(self):
        """Test conversion with invalid objective defaults to balance."""
        prefs = Preferences(optimization_objective="invalid_objective")

        objective = prefs.to_optimization_objective()

        assert objective == OptimizationObjective.BALANCE_EFFICIENCY

    def test_to_memory_config(self):
        """Test conversion to MemoryConfig."""
        prefs = Preferences(memory_conservatism="aggressive")

        config = prefs.to_memory_config()

        assert isinstance(config, MemoryConfig)
        assert config.utilization_bound == 0.95  # Aggressive


class TestDetailedConfiguration:
    """Test DetailedConfiguration class."""

    def test_detailed_configuration_creation(self):
        """Test DetailedConfiguration creation."""
        config = DetailedConfiguration(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            data_parallel_size=1,
            total_gpus=8,
            performance_metrics={"throughput_tokens_per_second": 1000},
            memory_breakdown={"total_memory_gb": 100},
            deployment_commands={"vllm": "python -m vllm"},
            configuration_rationale="Optimized for throughput",
            is_recommended=True,
        )

        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 2
        assert config.total_gpus == 8
        assert config.is_recommended is True


class TestAnalysisInsights:
    """Test AnalysisInsights class."""

    def test_analysis_insights_creation(self):
        """Test AnalysisInsights creation."""
        insights = AnalysisInsights(
            model_architecture="Dense Transformer",
            parameter_count="7B",
            memory_requirements={"minimum_memory_gb": 50},
            parallelism_constraints={"max_tensor_parallel": 8},
            performance_bottlenecks=["Memory bandwidth"],
            optimization_recommendations=["Use tensor parallelism"],
        )

        assert insights.model_architecture == "Dense Transformer"
        assert insights.parameter_count == "7B"
        assert len(insights.performance_bottlenecks) == 1
        assert len(insights.optimization_recommendations) == 1


class TestOptimizationResult:
    """Test OptimizationResult class."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        cluster = Cluster()
        workload = Workload()
        preferences = Preferences()
        config = DetailedConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=1,
            total_gpus=2,
            performance_metrics={},
            memory_breakdown={},
            deployment_commands={"vllm": "test command"},
            configuration_rationale="Test",
            is_recommended=True,
        )
        insights = AnalysisInsights(
            model_architecture="Dense",
            parameter_count="7B",
            memory_requirements={},
            parallelism_constraints={},
            performance_bottlenecks=[],
            optimization_recommendations=[],
        )

        result = OptimizationResult(
            model_name="test-model",
            cluster=cluster,
            workload=workload,
            preferences=preferences,
            recommended_configuration=config,
            all_configurations=[config],
            insights=insights,
            generation_time_seconds=1.5,
        )

        assert result.model_name == "test-model"
        assert result.generation_time_seconds == 1.5
        assert result.recommended_configuration == config

    def test_deploy_vllm(self):
        """Test deploy_vllm method."""
        config = DetailedConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=1,
            total_gpus=2,
            performance_metrics={},
            memory_breakdown={},
            deployment_commands={"vllm": "python -m vllm --model test"},
            configuration_rationale="Test",
            is_recommended=True,
        )

        result = OptimizationResult(
            model_name="test-model",
            cluster=Cluster(),
            workload=Workload(),
            preferences=Preferences(),
            recommended_configuration=config,
            all_configurations=[config],
            insights=Mock(),
            generation_time_seconds=1.0,
        )

        command = result.deploy_vllm()
        assert command == "python -m vllm --model test"

    def test_deploy_vllm_no_config(self):
        """Test deploy_vllm with no recommended configuration."""
        result = OptimizationResult(
            model_name="test-model",
            cluster=Cluster(),
            workload=Workload(),
            preferences=Preferences(),
            recommended_configuration=None,
            all_configurations=[],
            insights=Mock(),
            generation_time_seconds=1.0,
        )

        with pytest.raises(ValueError, match="No recommended configuration available"):
            result.deploy_vllm()

    def test_explain_with_config(self):
        """Test explain method with configuration."""
        config = DetailedConfiguration(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            expert_parallel_size=1,
            data_parallel_size=1,
            total_gpus=8,
            performance_metrics={
                "throughput_tokens_per_second": 1500.0,
                "latency_ms": 50.0,
                "memory_utilization": 0.8,
                "cost_per_hour": 25.0,
            },
            memory_breakdown={
                "weights_gb": 30.0,
                "activations_gb": 20.0,
                "kv_cache_gb": 10.0,
                "total_memory_gb": 80.0,
            },
            deployment_commands={},
            configuration_rationale="Test",
            is_recommended=True,
        )

        insights = AnalysisInsights(
            model_architecture="Dense",
            parameter_count="7B",
            memory_requirements={},
            parallelism_constraints={},
            performance_bottlenecks=[],
            optimization_recommendations=["Use tensor parallelism", "Consider pipeline parallelism"],
        )

        cluster = Cluster(num_nodes=2, gpus_per_node=4)
        
        result = OptimizationResult(
            model_name="test-model",
            cluster=cluster,
            workload=Workload(),
            preferences=Preferences(),
            recommended_configuration=config,
            all_configurations=[config],
            insights=insights,
            generation_time_seconds=1.0,
        )

        explanation = result.explain()
        
        assert "test-model" in explanation
        assert "2 nodes" in explanation
        assert "4 GPUs" in explanation
        assert "Tensor Parallelism: 4x" in explanation
        assert "Pipeline Parallelism: 2x" in explanation
        assert "1500.0 tokens/sec" in explanation
        assert "Use tensor parallelism" in explanation

    def test_explain_no_config(self):
        """Test explain method with no configuration."""
        result = OptimizationResult(
            model_name="test-model",
            cluster=Cluster(),
            workload=Workload(),
            preferences=Preferences(),
            recommended_configuration=None,
            all_configurations=[],
            insights=Mock(),
            generation_time_seconds=1.0,
        )

        explanation = result.explain()
        assert "No valid configurations found" in explanation


class TestAutoParallel:
    """Test AutoParallel main class."""

    def test_autoparallel_creation(self):
        """Test AutoParallel creation with defaults."""
        ap = AutoParallel()

        assert isinstance(ap.cluster, Cluster)
        assert isinstance(ap.preferences, Preferences)
        assert ap.cluster.gpu_model == "H100"  # Default

    def test_autoparallel_creation_with_params(self):
        """Test AutoParallel creation with custom parameters."""
        cluster = Cluster(gpu_model="A100")
        preferences = Preferences(framework="vllm")

        ap = AutoParallel(cluster=cluster, preferences=preferences)

        assert ap.cluster.gpu_model == "A100"
        assert ap.preferences.framework == "vllm"

    @patch("autoparallel.api.advanced.ConfigurationGenerator")
    @patch("autoparallel.api.advanced.analyze_model_constraints")
    @patch.object(AutoParallel, "_load_model_config")
    def test_analyze_model_basic(self, mock_load_config, mock_analyze_constraints, mock_generator_class):
        """Test analyze_model method basic flow."""
        # Setup mocks
        mock_config = Mock()
        mock_config.to_dict.return_value = {"hidden_size": 768}
        mock_load_config.return_value = mock_config

        mock_constraints = Mock()
        mock_constraints.max_tensor_parallel = 8
        mock_constraints.max_pipeline_parallel = 4
        mock_constraints.max_expert_parallel = 1
        mock_constraints.supports_grouped_query_attention = False
        mock_analyze_constraints.return_value = mock_constraints

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock result with empty configurations (edge case)
        mock_result = Mock()
        mock_result.configurations = []
        mock_generator.generate_valid_configs.return_value = mock_result

        # Create AutoParallel instance
        ap = AutoParallel()

        # Call analyze_model
        result = ap.analyze_model("test-model")

        # Verify result
        assert isinstance(result, OptimizationResult)
        assert result.model_name == "test-model"
        assert result.recommended_configuration is None  # No configurations
        assert len(result.all_configurations) == 0

        # Verify method calls
        mock_load_config.assert_called_once_with("test-model")
        mock_analyze_constraints.assert_called_once()
        mock_generator_class.assert_called_once()
        mock_generator.generate_valid_configs.assert_called_once()

    def test_optimize_delegates_to_analyze_model(self):
        """Test optimize method delegates to analyze_model."""
        ap = AutoParallel()
        
        # Mock analyze_model
        mock_result = Mock(spec=OptimizationResult)
        with patch.object(ap, 'analyze_model', return_value=mock_result) as mock_analyze:
            workload = Workload(workload_type="training")
            result = ap.optimize("test-model", workload)
            
            assert result == mock_result
            mock_analyze.assert_called_once_with("test-model", workload)
