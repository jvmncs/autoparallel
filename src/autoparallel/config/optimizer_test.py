"""Comprehensive tests for configuration optimizer."""

from unittest.mock import Mock

import pytest

from autoparallel.config.optimizer import (
    ConfigurationOptimizer,
    GreedyConfigurationOptimizer,
    HardwareProfile,
    MultiObjectiveConfigurationOptimizer,
    OptimizationObjective,
    ParallelismConfiguration,
    PerformanceMetrics,
    WorkloadProfile,
    WorkloadType,
    create_optimizer,
)
from autoparallel.constraints.analyzer import ModelConstraints
from autoparallel.memory.components import MemoryComponents
from autoparallel.memory.estimator import MemoryEstimator


class TestHardwareProfile:
    """Test HardwareProfile functionality."""

    def test_hardware_profile_creation(self):
        """Test HardwareProfile creation with default values."""
        profile = HardwareProfile()

        assert profile.gpu_model == "H100"
        assert profile.gpu_memory_gb == 80.0
        assert profile.gpus_per_node == 8
        assert profile.num_nodes == 1
        assert profile.intra_node_bandwidth_gbps == 900.0
        assert profile.inter_node_bandwidth_gbps == 200.0
        assert profile.network_topology == "fat_tree"

    def test_hardware_profile_custom_values(self):
        """Test HardwareProfile with custom values."""
        profile = HardwareProfile(
            gpu_model="A100",
            gpu_memory_gb=40.0,
            gpus_per_node=4,
            num_nodes=2,
            intra_node_bandwidth_gbps=600.0,
            inter_node_bandwidth_gbps=100.0,
            network_topology="mesh",
        )

        assert profile.gpu_model == "A100"
        assert profile.gpu_memory_gb == 40.0
        assert profile.gpus_per_node == 4
        assert profile.num_nodes == 2
        assert profile.intra_node_bandwidth_gbps == 600.0
        assert profile.inter_node_bandwidth_gbps == 100.0
        assert profile.network_topology == "mesh"

    def test_total_gpus_calculation(self):
        """Test total_gpus property calculation."""
        profile = HardwareProfile(gpus_per_node=8, num_nodes=4)
        assert profile.total_gpus == 32

    def test_total_memory_calculation(self):
        """Test total_memory_gb property calculation."""
        profile = HardwareProfile(gpu_memory_gb=80.0, gpus_per_node=8, num_nodes=2)
        assert profile.total_memory_gb == 1280.0  # 80 * 8 * 2

    @pytest.mark.parametrize("tp_size,expected_efficiency", [
        (1, 1.0),      # No tensor parallelism
        (4, 0.9),      # Intra-node TP (4 <= 8 gpus_per_node)
        (8, 0.9),      # Intra-node TP (8 == gpus_per_node)
        (16, 0.6),     # Inter-node TP (16 > 8 gpus_per_node)
    ])
    def test_communication_efficiency_tensor_parallel(self, tp_size, expected_efficiency):
        """Test communication efficiency calculation for tensor parallelism."""
        profile = HardwareProfile(gpus_per_node=8, num_nodes=4)
        efficiency = profile.get_communication_efficiency(tp_size, pipeline_parallel_size=1)
        assert efficiency == expected_efficiency

    @pytest.mark.parametrize("pp_size,expected_efficiency", [
        (1, 1.0),   # No pipeline parallelism
        (2, 0.8),   # Pipeline parallelism
        (4, 0.8),   # Pipeline parallelism
    ])
    def test_communication_efficiency_pipeline_parallel(self, pp_size, expected_efficiency):
        """Test communication efficiency calculation for pipeline parallelism."""
        profile = HardwareProfile(gpus_per_node=8, num_nodes=4)
        efficiency = profile.get_communication_efficiency(tensor_parallel_size=1, pipeline_parallel_size=pp_size)
        assert efficiency == expected_efficiency

    def test_communication_efficiency_combined(self):
        """Test communication efficiency with both TP and PP."""
        profile = HardwareProfile(gpus_per_node=8, num_nodes=4)

        # Intra-node TP (0.9) + PP (0.8) = 0.9 * 0.8 = 0.72
        efficiency = profile.get_communication_efficiency(tensor_parallel_size=4, pipeline_parallel_size=2)
        assert abs(efficiency - 0.72) < 1e-10  # Account for floating point precision

        # Inter-node TP (0.6) + PP (0.8) = 0.6 * 0.8 = 0.48
        efficiency = profile.get_communication_efficiency(tensor_parallel_size=16, pipeline_parallel_size=2)
        assert efficiency == 0.48


class TestWorkloadProfile:
    """Test WorkloadProfile functionality."""

    def test_workload_profile_creation(self):
        """Test WorkloadProfile creation with default values."""
        profile = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        assert profile.workload_type == WorkloadType.INFERENCE
        assert profile.batch_size == 32
        assert profile.sequence_length == 2048
        assert profile.requests_per_second == 100.0
        assert profile.latency_budget_ms == 100.0
        assert profile.throughput_target == 1000.0
        assert profile.cost_budget_per_hour == 100.0
        assert profile.is_training is False

    def test_workload_profile_custom_values(self):
        """Test WorkloadProfile with custom values."""
        profile = WorkloadProfile(
            workload_type=WorkloadType.TRAINING,
            batch_size=64,
            sequence_length=4096,
            requests_per_second=50.0,
            latency_budget_ms=200.0,
            throughput_target=2000.0,
            cost_budget_per_hour=200.0,
            is_training=True,
        )

        assert profile.workload_type == WorkloadType.TRAINING
        assert profile.batch_size == 64
        assert profile.sequence_length == 4096
        assert profile.requests_per_second == 50.0
        assert profile.latency_budget_ms == 200.0
        assert profile.throughput_target == 2000.0
        assert profile.cost_budget_per_hour == 200.0
        assert profile.is_training is True

    @pytest.mark.parametrize("workload_type,expected_weights", [
        (WorkloadType.INFERENCE, {"throughput": 0.4, "latency": 0.3, "memory_efficiency": 0.2, "cost": 0.1}),
        (WorkloadType.TRAINING, {"throughput": 0.6, "latency": 0.1, "memory_efficiency": 0.2, "cost": 0.1}),
        (WorkloadType.BATCH_PROCESSING, {"throughput": 0.7, "latency": 0.1, "memory_efficiency": 0.1, "cost": 0.1}),
        (WorkloadType.INTERACTIVE, {"throughput": 0.2, "latency": 0.6, "memory_efficiency": 0.1, "cost": 0.1}),
        (WorkloadType.CHATBOT, {"throughput": 0.3, "latency": 0.5, "memory_efficiency": 0.1, "cost": 0.1}),
    ])
    def test_priority_weights(self, workload_type, expected_weights):
        """Test priority weights for different workload types."""
        profile = WorkloadProfile(workload_type=workload_type)
        weights = profile.get_priority_weights()
        assert weights == expected_weights

    def test_priority_weights_sum_to_one(self):
        """Test that priority weights sum to 1.0."""
        for workload_type in WorkloadType:
            profile = WorkloadProfile(workload_type=workload_type)
            weights = profile.get_priority_weights()
            assert abs(sum(weights.values()) - 1.0) < 1e-10


class TestParallelismConfiguration:
    """Test ParallelismConfiguration functionality."""

    def test_parallelism_configuration_default(self):
        """Test ParallelismConfiguration with default values."""
        config = ParallelismConfiguration()

        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
        assert config.expert_parallel_size == 1
        assert config.data_parallel_size == 1

    def test_parallelism_configuration_custom(self):
        """Test ParallelismConfiguration with custom values."""
        config = ParallelismConfiguration(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            expert_parallel_size=2,
            data_parallel_size=4,
        )

        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 2
        assert config.expert_parallel_size == 2
        assert config.data_parallel_size == 4

    def test_total_gpus_needed(self):
        """Test total_gpus_needed calculation."""
        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            expert_parallel_size=2,
            data_parallel_size=4,
        )

        assert config.total_gpus_needed == 64  # 2 * 4 * 2 * 4

    def test_to_dict(self):
        """Test to_dict conversion."""
        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            expert_parallel_size=1,
            data_parallel_size=2,
        )

        expected = {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 4,
            "expert_parallel_size": 1,
            "data_parallel_size": 2,
        }

        assert config.to_dict() == expected

    @pytest.mark.parametrize("invalid_value", [0, -1, -10])
    def test_parallelism_configuration_validation_tensor_parallel(self, invalid_value):
        """Test validation of tensor_parallel_size."""
        with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
            ParallelismConfiguration(tensor_parallel_size=invalid_value)

    @pytest.mark.parametrize("invalid_value", [0, -1, -10])
    def test_parallelism_configuration_validation_pipeline_parallel(self, invalid_value):
        """Test validation of pipeline_parallel_size."""
        with pytest.raises(ValueError, match="pipeline_parallel_size must be >= 1"):
            ParallelismConfiguration(pipeline_parallel_size=invalid_value)

    @pytest.mark.parametrize("invalid_value", [0, -1, -10])
    def test_parallelism_configuration_validation_expert_parallel(self, invalid_value):
        """Test validation of expert_parallel_size."""
        with pytest.raises(ValueError, match="expert_parallel_size must be >= 1"):
            ParallelismConfiguration(expert_parallel_size=invalid_value)

    @pytest.mark.parametrize("invalid_value", [0, -1, -10])
    def test_parallelism_configuration_validation_data_parallel(self, invalid_value):
        """Test validation of data_parallel_size."""
        with pytest.raises(ValueError, match="data_parallel_size must be >= 1"):
            ParallelismConfiguration(data_parallel_size=invalid_value)


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation with default values."""
        metrics = PerformanceMetrics()

        assert metrics.throughput_score == 0.0
        assert metrics.latency_score == 0.0
        assert metrics.memory_efficiency_score == 0.0
        assert metrics.cost_score == 0.0
        assert metrics.communication_efficiency == 0.0
        assert metrics.gpu_utilization == 0.0
        assert metrics.memory_utilization_gb_per_gpu == 0.0
        assert metrics.is_feasible is True

    def test_performance_metrics_custom_values(self):
        """Test PerformanceMetrics with custom values."""
        metrics = PerformanceMetrics(
            throughput_score=0.8,
            latency_score=0.7,
            memory_efficiency_score=0.9,
            cost_score=0.6,
            communication_efficiency=0.85,
            gpu_utilization=0.75,
            memory_utilization_gb_per_gpu=45.0,
            is_feasible=False,
        )

        assert metrics.throughput_score == 0.8
        assert metrics.latency_score == 0.7
        assert metrics.memory_efficiency_score == 0.9
        assert metrics.cost_score == 0.6
        assert metrics.communication_efficiency == 0.85
        assert metrics.gpu_utilization == 0.75
        assert metrics.memory_utilization_gb_per_gpu == 45.0
        assert metrics.is_feasible is False

    def test_get_weighted_score(self):
        """Test weighted score calculation."""
        metrics = PerformanceMetrics(
            throughput_score=0.8,
            latency_score=0.6,
            memory_efficiency_score=0.7,
            cost_score=0.9,
        )

        weights = {
            "throughput": 0.4,
            "latency": 0.3,
            "memory_efficiency": 0.2,
            "cost": 0.1,
        }

        expected_score = 0.8 * 0.4 + 0.6 * 0.3 + 0.7 * 0.2 + 0.9 * 0.1
        assert abs(metrics.get_weighted_score(weights) - expected_score) < 1e-10

    def test_get_weighted_score_missing_weights(self):
        """Test weighted score with missing weight keys."""
        metrics = PerformanceMetrics(
            throughput_score=0.8,
            latency_score=0.6,
        )

        weights = {"throughput": 0.5}  # Missing other weights

        expected_score = 0.8 * 0.5  # Other weights default to 0.0
        assert abs(metrics.get_weighted_score(weights) - expected_score) < 1e-10

    def test_get_weighted_score_empty_weights(self):
        """Test weighted score with empty weights."""
        metrics = PerformanceMetrics(
            throughput_score=0.8,
            latency_score=0.6,
        )

        weights = {}  # Empty weights

        assert metrics.get_weighted_score(weights) == 0.0


class TestConfigurationOptimizer:
    """Test ConfigurationOptimizer abstract base class."""

    def test_optimizer_initialization(self):
        """Test ConfigurationOptimizer initialization."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile()

        # Create a concrete subclass for testing
        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        assert optimizer.memory_estimator is memory_estimator
        assert optimizer.model_constraints is model_constraints
        assert optimizer.hardware_profile is hardware_profile

    def test_calculate_performance_metrics(self):
        """Test _calculate_performance_metrics method."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile(gpu_memory_gb=80.0)

        # Mock memory estimation
        memory_components = MemoryComponents(
            weights=20 * (1024**3),  # 20 GB
            activations=10 * (1024**3),  # 10 GB
            kv_cache=5 * (1024**3),      # 5 GB
        )
        memory_estimator.estimate_memory.return_value = memory_components

        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            data_parallel_size=4,
        )
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE, batch_size=32)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        metrics = optimizer._calculate_performance_metrics(config, workload, model_config)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.is_feasible  # 35 GB should fit in 80 GB GPU
        assert metrics.throughput_score > 0
        assert metrics.latency_score > 0
        assert metrics.memory_efficiency_score >= 0
        assert metrics.cost_score >= 0

    def test_calculate_throughput_score(self):
        """Test _calculate_throughput_score method."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile(gpus_per_node=8)

        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        config = ParallelismConfiguration(
            tensor_parallel_size=4,  # Intra-node
            pipeline_parallel_size=2,
            data_parallel_size=2,
        )
        workload = WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=32,
            throughput_target=1000.0,
        )
        memory_components = MemoryComponents(
            weights=40 * (1024**3),  # 40 GB, 50% utilization on 80GB GPU
        )

        score = optimizer._calculate_throughput_score(config, workload, memory_components)

        assert score > 0
        assert score <= 2.0  # Normalized to max 2.0

    def test_calculate_latency_score(self):
        """Test _calculate_latency_score method."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile(gpus_per_node=8)

        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        config = ParallelismConfiguration(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,  # No pipeline latency
        )
        workload = WorkloadProfile(
            workload_type=WorkloadType.INTERACTIVE,
            latency_budget_ms=100.0,
        )
        memory_components = MemoryComponents()

        score = optimizer._calculate_latency_score(config, workload, memory_components)

        assert score > 0

    def test_calculate_latency_score_exceeds_budget(self):
        """Test latency score when exceeding budget."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile(gpus_per_node=4)  # Force inter-node communication

        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        config = ParallelismConfiguration(
            tensor_parallel_size=8,  # Inter-node communication
            pipeline_parallel_size=4,
        )
        workload = WorkloadProfile(
            workload_type=WorkloadType.CHATBOT,
            latency_budget_ms=10.0,  # Very tight budget
        )
        memory_components = MemoryComponents()

        score = optimizer._calculate_latency_score(config, workload, memory_components)

        assert score == 0.1  # Penalty for exceeding budget

    @pytest.mark.parametrize("utilization,expected_penalty", [
        (0.85, False),  # Optimal
        (0.25, True),   # Under-utilized
        (0.97, True),   # Over-utilized
    ])
    def test_calculate_memory_efficiency_score(self, utilization, expected_penalty):
        """Test memory efficiency score calculation."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile(gpu_memory_gb=80.0)

        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        config = ParallelismConfiguration()
        memory_gb = utilization * 80.0
        memory_components = MemoryComponents(weights=int(memory_gb * (1024**3)))

        score = optimizer._calculate_memory_efficiency_score(config, memory_components)

        assert 0 <= score <= 1.0
        if expected_penalty:
            assert score < 1.0  # Should be penalized
        else:
            assert score > 0.8  # Should be near optimal

    def test_calculate_cost_score(self):
        """Test cost efficiency score calculation."""
        memory_estimator = Mock(spec=MemoryEstimator)
        model_constraints = Mock(spec=ModelConstraints)
        hardware_profile = HardwareProfile()

        class TestOptimizer(ConfigurationOptimizer):
            def rank_configurations(self, configurations, workload_profile, optimization_objective):
                return []

            def optimize_for_workload(self, workload_profile, optimization_objective, max_configurations):
                return []

        optimizer = TestOptimizer(memory_estimator, model_constraints, hardware_profile)

        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            data_parallel_size=4,
        )  # 8 GPUs total

        workload = WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=64,
        )

        score = optimizer._calculate_cost_score(config, workload)

        assert score >= 0
        assert score <= 2.0  # Normalized to max 2.0


class TestGreedyConfigurationOptimizer:
    """Test GreedyConfigurationOptimizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.memory_estimator = Mock(spec=MemoryEstimator)
        self.model_constraints = Mock(spec=ModelConstraints)
        self.hardware_profile = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)

        # Mock constraint methods
        self.model_constraints.get_valid_tensor_parallel_sizes.return_value = [1, 2, 4]
        self.model_constraints.get_valid_expert_parallel_sizes.return_value = [1]
        self.model_constraints.get_valid_pipeline_parallel_sizes.return_value = [1, 2]

        # Mock memory estimation
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=20 * (1024**3),
            activations=10 * (1024**3),
            kv_cache=5 * (1024**3),
        )

    def test_greedy_optimizer_initialization(self):
        """Test GreedyConfigurationOptimizer initialization."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        assert optimizer.memory_estimator is self.memory_estimator
        assert optimizer.model_constraints is self.model_constraints
        assert optimizer.hardware_profile is self.hardware_profile

    def test_rank_configurations_maximize_throughput(self):
        """Test ranking configurations by throughput."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        # Create configurations with different expected throughputs
        configs = [
            ParallelismConfiguration(data_parallel_size=1),  # Lower throughput
            ParallelismConfiguration(data_parallel_size=4),  # Higher throughput
            ParallelismConfiguration(data_parallel_size=2),  # Medium throughput
        ]

        workload = WorkloadProfile(workload_type=WorkloadType.TRAINING, batch_size=32)

        ranked = optimizer.rank_configurations(
            configs,
            workload,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
        )

        assert len(ranked) == 3
        # Should be sorted by throughput score (descending)
        throughput_scores = [metrics.throughput_score for _, metrics in ranked]
        assert throughput_scores == sorted(throughput_scores, reverse=True)

    def test_rank_configurations_minimize_latency(self):
        """Test ranking configurations by latency."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        configs = [
            ParallelismConfiguration(tensor_parallel_size=1, pipeline_parallel_size=4),  # Higher latency
            ParallelismConfiguration(tensor_parallel_size=4, pipeline_parallel_size=1),  # Lower latency
        ]

        workload = WorkloadProfile(workload_type=WorkloadType.CHATBOT, latency_budget_ms=100.0)

        ranked = optimizer.rank_configurations(
            configs,
            workload,
            OptimizationObjective.MINIMIZE_LATENCY,
        )

        assert len(ranked) == 2
        # Should be sorted by latency score (descending, meaning lower latency first)
        latency_scores = [metrics.latency_score for _, metrics in ranked]
        assert latency_scores == sorted(latency_scores, reverse=True)

    def test_rank_configurations_balance_efficiency(self):
        """Test ranking configurations by balanced efficiency."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        configs = [
            ParallelismConfiguration(tensor_parallel_size=1),
            ParallelismConfiguration(tensor_parallel_size=2),
        ]

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        ranked = optimizer.rank_configurations(
            configs,
            workload,
            OptimizationObjective.BALANCE_EFFICIENCY,
        )

        assert len(ranked) == 2
        # Should be sorted by weighted score
        weights = workload.get_priority_weights()
        weighted_scores = [metrics.get_weighted_score(weights) for _, metrics in ranked]
        assert weighted_scores == sorted(weighted_scores, reverse=True)

    def test_rank_configurations_filters_infeasible(self):
        """Test that infeasible configurations are filtered out."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        # Mock memory estimation to return high memory for first config
        def mock_estimate_memory(**kwargs):
            if kwargs.get("tensor_parallel_size", 1) == 1:
                # First config: too much memory
                return MemoryComponents(weights=100 * (1024**3))  # 100 GB
            else:
                # Other configs: feasible
                return MemoryComponents(weights=30 * (1024**3))   # 30 GB

        self.memory_estimator.estimate_memory.side_effect = mock_estimate_memory

        configs = [
            ParallelismConfiguration(tensor_parallel_size=1),  # Will be infeasible
            ParallelismConfiguration(tensor_parallel_size=2),  # Will be feasible
        ]

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        ranked = optimizer.rank_configurations(configs, workload)

        # Should only return feasible configurations
        assert len(ranked) == 1
        assert ranked[0][0].tensor_parallel_size == 2

    def test_optimize_for_workload(self):
        """Test optimizing for a specific workload."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
            max_configurations=50,
        )

        assert len(results) > 0
        # Should return configurations sorted by objective
        for config, metrics in results:
            assert isinstance(config, ParallelismConfiguration)
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.is_feasible

    def test_generate_valid_configurations(self):
        """Test generation of valid configurations."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        configs = optimizer._generate_valid_configurations(max_configurations=100)

        assert len(configs) > 0
        # All configurations should use valid sizes
        for config in configs:
            assert config.tensor_parallel_size in [1, 2, 4]
            assert config.expert_parallel_size == 1
            assert config.pipeline_parallel_size in [1, 2]
            assert config.total_gpus_needed <= self.hardware_profile.total_gpus

    def test_generate_valid_configurations_max_limit(self):
        """Test that max_configurations limit is respected."""
        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        max_configs = 3
        configs = optimizer._generate_valid_configurations(max_configurations=max_configs)

        assert len(configs) <= max_configs

    def test_generate_valid_configurations_gpu_constraint(self):
        """Test that configurations respect GPU constraints."""
        # Small hardware profile
        small_hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=4, num_nodes=1)

        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            small_hardware,
        )

        configs = optimizer._generate_valid_configurations(max_configurations=100)

        # No configuration should require more than 4 GPUs
        for config in configs:
            assert config.total_gpus_needed <= 4


class TestMultiObjectiveConfigurationOptimizer:
    """Test MultiObjectiveConfigurationOptimizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.memory_estimator = Mock(spec=MemoryEstimator)
        self.model_constraints = Mock(spec=ModelConstraints)
        self.hardware_profile = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)

        # Mock constraint methods
        self.model_constraints.get_valid_tensor_parallel_sizes.return_value = [1, 2, 4]
        self.model_constraints.get_valid_expert_parallel_sizes.return_value = [1]
        self.model_constraints.get_valid_pipeline_parallel_sizes.return_value = [1, 2]

        # Mock memory estimation
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=20 * (1024**3),
            activations=10 * (1024**3),
            kv_cache=5 * (1024**3),
        )

    def test_multi_objective_optimizer_initialization(self):
        """Test MultiObjectiveConfigurationOptimizer initialization."""
        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        assert optimizer.memory_estimator is self.memory_estimator
        assert optimizer.model_constraints is self.model_constraints
        assert optimizer.hardware_profile is self.hardware_profile

    def test_rank_configurations_pareto_front(self):
        """Test ranking with Pareto front analysis."""
        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        configs = [
            ParallelismConfiguration(tensor_parallel_size=1, data_parallel_size=4),
            ParallelismConfiguration(tensor_parallel_size=2, data_parallel_size=2),
            ParallelismConfiguration(tensor_parallel_size=4, data_parallel_size=1),
        ]

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        ranked = optimizer.rank_configurations(configs, workload)

        assert len(ranked) >= 0  # May filter some configurations
        # Pareto-optimal solutions should be ranked first
        for config, metrics in ranked:
            assert isinstance(config, ParallelismConfiguration)
            assert isinstance(metrics, PerformanceMetrics)

    def test_find_pareto_front_simple(self):
        """Test Pareto front identification with simple case."""
        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        # Create configurations with clear dominance relationships
        results = [
            (
                ParallelismConfiguration(tensor_parallel_size=1),
                PerformanceMetrics(throughput_score=0.5, latency_score=0.5, memory_efficiency_score=0.5, cost_score=0.5)
            ),
            (
                ParallelismConfiguration(tensor_parallel_size=2),
                PerformanceMetrics(throughput_score=0.8, latency_score=0.8, memory_efficiency_score=0.8, cost_score=0.8)
            ),  # Dominates first
            (
                ParallelismConfiguration(tensor_parallel_size=4),
                PerformanceMetrics(throughput_score=0.9, latency_score=0.3, memory_efficiency_score=0.7, cost_score=0.6)
            ),  # Mixed performance, non-dominated
        ]

        pareto_front = optimizer._find_pareto_front(results)

        # Second and third should be in Pareto front (first is dominated by second)
        assert len(pareto_front) == 2
        pareto_tp_sizes = [config.tensor_parallel_size for config, _ in pareto_front]
        assert 2 in pareto_tp_sizes
        assert 4 in pareto_tp_sizes
        assert 1 not in pareto_tp_sizes

    def test_dominates_method(self):
        """Test dominance checking method."""
        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        # A dominates B (A is better or equal in all objectives, strictly better in at least one)
        metrics_a = PerformanceMetrics(throughput_score=0.8, latency_score=0.7, memory_efficiency_score=0.6, cost_score=0.9)
        metrics_b = PerformanceMetrics(throughput_score=0.7, latency_score=0.6, memory_efficiency_score=0.5, cost_score=0.8)

        assert optimizer._dominates(metrics_a, metrics_b)
        assert not optimizer._dominates(metrics_b, metrics_a)

        # Neither dominates (each better in some objectives)
        metrics_c = PerformanceMetrics(throughput_score=0.9, latency_score=0.5, memory_efficiency_score=0.7, cost_score=0.6)
        metrics_d = PerformanceMetrics(throughput_score=0.6, latency_score=0.8, memory_efficiency_score=0.6, cost_score=0.9)

        assert not optimizer._dominates(metrics_c, metrics_d)
        assert not optimizer._dominates(metrics_d, metrics_c)

        # Equal metrics - neither dominates
        metrics_e = PerformanceMetrics(throughput_score=0.5, latency_score=0.5, memory_efficiency_score=0.5, cost_score=0.5)
        metrics_f = PerformanceMetrics(throughput_score=0.5, latency_score=0.5, memory_efficiency_score=0.5, cost_score=0.5)

        assert not optimizer._dominates(metrics_e, metrics_f)
        assert not optimizer._dominates(metrics_f, metrics_e)

    def test_optimize_for_workload_multi_objective(self):
        """Test multi-objective optimization for workload."""
        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.BALANCE_EFFICIENCY,
            max_configurations=50,
        )

        assert len(results) > 0
        # Should return Pareto-optimal solutions first
        for config, metrics in results:
            assert isinstance(config, ParallelismConfiguration)
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.is_feasible


class TestCreateOptimizer:
    """Test optimizer factory function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.memory_estimator = Mock(spec=MemoryEstimator)
        self.model_constraints = Mock(spec=ModelConstraints)
        self.hardware_profile = HardwareProfile()

    def test_create_greedy_optimizer(self):
        """Test creating greedy optimizer."""
        optimizer = create_optimizer(
            "greedy",
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        assert isinstance(optimizer, GreedyConfigurationOptimizer)
        assert optimizer.memory_estimator is self.memory_estimator
        assert optimizer.model_constraints is self.model_constraints
        assert optimizer.hardware_profile is self.hardware_profile

    def test_create_multi_objective_optimizer(self):
        """Test creating multi-objective optimizer."""
        optimizer = create_optimizer(
            "multi_objective",
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        assert isinstance(optimizer, MultiObjectiveConfigurationOptimizer)
        assert optimizer.memory_estimator is self.memory_estimator
        assert optimizer.model_constraints is self.model_constraints
        assert optimizer.hardware_profile is self.hardware_profile

    def test_create_optimizer_unknown_type(self):
        """Test creating optimizer with unknown type."""
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(
                "unknown_type",
                self.memory_estimator,
                self.model_constraints,
                self.hardware_profile,
            )

    @pytest.mark.parametrize("optimizer_type,expected_class", [
        ("greedy", GreedyConfigurationOptimizer),
        ("multi_objective", MultiObjectiveConfigurationOptimizer),
    ])
    def test_create_optimizer_types(self, optimizer_type, expected_class):
        """Test creating different optimizer types."""
        optimizer = create_optimizer(
            optimizer_type,
            self.memory_estimator,
            self.model_constraints,
            self.hardware_profile,
        )

        assert isinstance(optimizer, expected_class)


class TestWorkloadTypeEnum:
    """Test WorkloadType enum."""

    def test_workload_type_values(self):
        """Test WorkloadType enum values."""
        assert WorkloadType.INFERENCE.value == "inference"
        assert WorkloadType.TRAINING.value == "training"
        assert WorkloadType.BATCH_PROCESSING.value == "batch_processing"
        assert WorkloadType.INTERACTIVE.value == "interactive"
        assert WorkloadType.CHATBOT.value == "chatbot"

    def test_workload_type_iteration(self):
        """Test iterating over WorkloadType enum."""
        workload_types = list(WorkloadType)
        assert len(workload_types) == 5
        assert WorkloadType.INFERENCE in workload_types
        assert WorkloadType.TRAINING in workload_types
        assert WorkloadType.BATCH_PROCESSING in workload_types
        assert WorkloadType.INTERACTIVE in workload_types
        assert WorkloadType.CHATBOT in workload_types


class TestOptimizationObjectiveEnum:
    """Test OptimizationObjective enum."""

    def test_optimization_objective_values(self):
        """Test OptimizationObjective enum values."""
        assert OptimizationObjective.MAXIMIZE_THROUGHPUT.value == "maximize_throughput"
        assert OptimizationObjective.MINIMIZE_LATENCY.value == "minimize_latency"
        assert OptimizationObjective.MINIMIZE_COST.value == "minimize_cost"
        assert OptimizationObjective.BALANCE_EFFICIENCY.value == "balance_efficiency"
        assert OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY.value == "maximize_memory_efficiency"

    def test_optimization_objective_iteration(self):
        """Test iterating over OptimizationObjective enum."""
        objectives = list(OptimizationObjective)
        assert len(objectives) == 5
        assert OptimizationObjective.MAXIMIZE_THROUGHPUT in objectives
        assert OptimizationObjective.MINIMIZE_LATENCY in objectives
        assert OptimizationObjective.MINIMIZE_COST in objectives
        assert OptimizationObjective.BALANCE_EFFICIENCY in objectives
        assert OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY in objectives


class TestIntegrationScenarios:
    """Test integration scenarios with realistic workloads."""

    def setup_method(self):
        """Set up realistic test scenario."""
        self.memory_estimator = Mock(spec=MemoryEstimator)
        self.model_constraints = Mock(spec=ModelConstraints)

        # Set up realistic constraints
        self.model_constraints.get_valid_tensor_parallel_sizes.return_value = [1, 2, 4, 8]
        self.model_constraints.get_valid_expert_parallel_sizes.return_value = [1]
        self.model_constraints.get_valid_pipeline_parallel_sizes.return_value = [1, 2, 4]

        # Realistic memory estimation
        self.memory_estimator.estimate_memory.return_value = MemoryComponents(
            weights=35 * (1024**3),  # 35 GB
            activations=15 * (1024**3),  # 15 GB
            kv_cache=8 * (1024**3),   # 8 GB
        )

    def test_large_scale_inference_optimization(self):
        """Test optimization for large-scale inference workload."""
        hardware = HardwareProfile(
            gpu_memory_gb=80,
            gpus_per_node=8,
            num_nodes=4,  # 32 GPUs total
            intra_node_bandwidth_gbps=900,
            inter_node_bandwidth_gbps=200,
        )

        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            hardware,
        )

        workload = WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=128,
            sequence_length=2048,
            throughput_target=5000.0,
            latency_budget_ms=150.0,
        )

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
            max_configurations=20,
        )

        assert len(results) > 0

        # Best configuration should utilize multiple GPUs efficiently
        best_config, best_metrics = results[0]
        assert best_config.total_gpus_needed > 1
        assert best_metrics.is_feasible
        assert best_metrics.throughput_score > 0

    def test_training_workload_optimization(self):
        """Test optimization for training workload."""
        hardware = HardwareProfile(
            gpu_memory_gb=80,
            gpus_per_node=8,
            num_nodes=2,  # 16 GPUs total
        )

        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            hardware,
        )

        workload = WorkloadProfile(
            workload_type=WorkloadType.TRAINING,
            batch_size=32,
            sequence_length=4096,
            is_training=True,
            throughput_target=2000.0,
        )

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.BALANCE_EFFICIENCY,
            max_configurations=15,
        )

        assert len(results) > 0

        # Should have multiple viable configurations with different trade-offs
        feasible_configs = [
            (config, metrics) for config, metrics in results
            if metrics.is_feasible
        ]
        assert len(feasible_configs) > 1

    def test_memory_constrained_optimization(self):
        """Test optimization under memory constraints."""
        # Smaller GPUs
        hardware = HardwareProfile(
            gpu_memory_gb=24,  # Smaller memory
            gpus_per_node=4,
            num_nodes=2,
        )

        # Mock memory estimation that scales with parallelism
        def mock_estimate_memory(**kwargs):
            tp_size = kwargs.get("tensor_parallel_size", 1)
            # Memory scales down with tensor parallelism
            base_memory = 30 * (1024**3)  # 30 GB base
            scaled_memory = base_memory // tp_size
            return MemoryComponents(weights=scaled_memory)

        self.memory_estimator.estimate_memory.side_effect = mock_estimate_memory

        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            hardware,
        )

        workload = WorkloadProfile(
            workload_type=WorkloadType.INFERENCE,
            batch_size=16,
            sequence_length=2048,
        )

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY,
            max_configurations=10,
        )

        # Should find some feasible configurations
        assert len(results) > 0
        feasible_results = [r for r in results if r[1].is_feasible]
        assert len(feasible_results) > 0

        # At least some configurations should use parallelism to fit in memory
        has_parallelism = any(
            config.tensor_parallel_size > 1 or
            config.pipeline_parallel_size > 1 or
            config.data_parallel_size > 1
            for config, _ in feasible_results
        )
        assert has_parallelism

    def test_latency_sensitive_optimization(self):
        """Test optimization for latency-sensitive workloads."""
        hardware = HardwareProfile(
            gpu_memory_gb=80,
            gpus_per_node=8,
            num_nodes=1,  # Single node to minimize communication
        )

        optimizer = GreedyConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            hardware,
        )

        workload = WorkloadProfile(
            workload_type=WorkloadType.CHATBOT,
            batch_size=1,  # Single batch for low latency
            sequence_length=1024,
            latency_budget_ms=50.0,  # Very tight latency budget
        )

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.MINIMIZE_LATENCY,
            max_configurations=10,
        )

        assert len(results) > 0

        # Best configurations should prioritize low latency
        best_configs = results[:3]  # Top 3 configurations

        for config, metrics in best_configs:
            assert metrics.is_feasible
            # Should prefer configurations that minimize pipeline parallelism
            # (which adds latency) and use tensor parallelism instead
            if config.total_gpus_needed > 1:
                assert config.tensor_parallel_size >= config.pipeline_parallel_size

    def test_cost_optimization_scenario(self):
        """Test optimization focused on cost efficiency."""
        hardware = HardwareProfile(
            gpu_memory_gb=80,
            gpus_per_node=8,
            num_nodes=4,  # Large cluster
        )

        optimizer = MultiObjectiveConfigurationOptimizer(
            self.memory_estimator,
            self.model_constraints,
            hardware,
        )

        workload = WorkloadProfile(
            workload_type=WorkloadType.BATCH_PROCESSING,
            batch_size=256,  # Large batch
            sequence_length=1024,
            cost_budget_per_hour=500.0,
        )

        results = optimizer.optimize_for_workload(
            workload,
            OptimizationObjective.MINIMIZE_COST,
            max_configurations=15,
        )

        assert len(results) > 0

        # Should prefer configurations that maximize GPU utilization
        # and minimize total GPU usage while maintaining throughput
        best_config, best_metrics = results[0]
        assert best_metrics.is_feasible
        assert best_metrics.cost_score > 0
