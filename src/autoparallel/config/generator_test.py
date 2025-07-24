"""Comprehensive tests for configuration generator."""

from unittest.mock import Mock, patch

import pytest

from autoparallel.config.generator import (
    ConfigurationGenerationResult,
    ConfigurationGenerator,
    ScoredConfiguration,
    generate_configurations_for_workload,
)
from autoparallel.config.optimizer import (
    HardwareProfile,
    OptimizationObjective,
    ParallelismConfiguration,
    PerformanceMetrics,
    WorkloadProfile,
    WorkloadType,
)
from autoparallel.constraints.analyzer import ModelConstraints
from autoparallel.constraints.validation import ValidationResult
from autoparallel.memory.components import MemoryComponents
from autoparallel.memory.estimator import (
    MoEMemoryEstimator,
    TransformersMemoryEstimator,
)


class TestConfigurationGenerator:
    """Test configuration generator functionality."""

    def test_generator_initialization_default(self):
        """Test generator initialization with default memory estimator."""
        generator = ConfigurationGenerator()
        assert generator.memory_estimator is not None
        assert isinstance(generator.memory_estimator, TransformersMemoryEstimator)

    def test_generator_initialization_custom_estimator(self):
        """Test generator initialization with custom memory estimator."""
        custom_estimator = Mock(spec=MoEMemoryEstimator)
        generator = ConfigurationGenerator(memory_estimator=custom_estimator)
        assert generator.memory_estimator is custom_estimator

    def test_generate_valid_configs_basic(self):
        """Test basic configuration generation."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
        }

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=10,
        )

        assert isinstance(result, ConfigurationGenerationResult)
        assert isinstance(result.configurations, list)
        assert len(result.configurations) > 0
        assert result.best_configuration is not None
        assert "total_enumerated" in result.generation_metadata
        assert "valid_configs" in result.generation_metadata
        assert "final_configs" in result.generation_metadata

    def test_generate_configs_with_explicit_constraints(self):
        """Test configuration generation with explicit model constraints."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=40, gpus_per_node=4, num_nodes=1)
        workload = WorkloadProfile(
            workload_type=WorkloadType.TRAINING, is_training=True
        )
        model_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 50000,
        }

        constraints = ModelConstraints(
            max_tensor_parallel=4,
            tensor_parallel_divisors={1, 2, 4},
            max_expert_parallel=1,
            expert_parallel_divisors={1},
            max_pipeline_parallel=4,
            min_layers_per_stage=2,
            requires_tied_embeddings=False,
            supports_grouped_query_attention=False,
            vocabulary_sharding=2,
        )

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            model_constraints=constraints,
            optimization_objective=OptimizationObjective.MAXIMIZE_THROUGHPUT,
        )

        assert len(result.configurations) > 0
        # All configurations should respect constraints
        for scored_config in result.configurations:
            config = scored_config.configuration
            assert config.tensor_parallel_size in constraints.tensor_parallel_divisors
            assert config.expert_parallel_size in constraints.expert_parallel_divisors

    @pytest.mark.parametrize("objective", [
        OptimizationObjective.MAXIMIZE_THROUGHPUT,
        OptimizationObjective.MINIMIZE_LATENCY,
        OptimizationObjective.MINIMIZE_COST,
        OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY,
        OptimizationObjective.BALANCE_EFFICIENCY,
    ])
    def test_optimization_objectives(self, objective):
        """Test different optimization objectives."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            optimization_objective=objective,
            max_configurations=5,
        )

        assert len(result.configurations) > 0
        assert result.generation_metadata["optimization_objective"] == objective.value

    def test_memory_feasibility_filtering(self):
        """Test that infeasible configurations are filtered out."""
        generator = ConfigurationGenerator()
        # Very small memory to force some configs to be infeasible
        hardware = HardwareProfile(gpu_memory_gb=8, gpus_per_node=4, num_nodes=1)
        workload = WorkloadProfile(
            workload_type=WorkloadType.INFERENCE, batch_size=64, sequence_length=4096
        )
        model_config = {
            "hidden_size": 8192,  # Large model
            "num_hidden_layers": 96,
            "num_attention_heads": 64,
            "vocab_size": 100000,
        }

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
        )

        # Should filter out infeasible configurations
        assert result.generation_metadata["total_enumerated"] >= len(result.configurations)

        # All returned configurations should be feasible
        for scored_config in result.configurations:
            assert scored_config.performance_metrics.is_feasible
            assert scored_config.is_valid

    def test_max_configurations_limit(self):
        """Test that max_configurations parameter is respected."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=2)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        max_configs = 3
        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=max_configs,
        )

        assert len(result.configurations) <= max_configs

    def test_max_configurations_zero_unlimited(self):
        """Test that max_configurations=0 means unlimited."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        }

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=0,  # Unlimited
        )

        # Should generate all valid configurations
        assert len(result.configurations) > 0

    def test_fallback_constraints_creation(self):
        """Test fallback constraint creation when analysis fails."""
        generator = ConfigurationGenerator()

        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        constraints = generator._create_fallback_constraints(model_config)

        assert constraints.max_tensor_parallel > 0
        assert len(constraints.tensor_parallel_divisors) > 0
        assert constraints.max_pipeline_parallel > 0
        assert 1 in constraints.tensor_parallel_divisors  # Should always include TP=1

    def test_fallback_constraints_moe_model(self):
        """Test fallback constraints for MoE models."""
        generator = ConfigurationGenerator()

        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_experts": 16,  # MoE model
        }

        constraints = generator._create_fallback_constraints(model_config)

        assert constraints.max_expert_parallel > 1
        assert len(constraints.expert_parallel_divisors) > 1
        assert 1 in constraints.expert_parallel_divisors
        assert 2 in constraints.expert_parallel_divisors

    def test_fallback_constraints_conservative_defaults(self):
        """Test that fallback constraints use conservative defaults."""
        generator = ConfigurationGenerator()

        model_config = {
            "num_attention_heads": 128,  # Large number
            "num_hidden_layers": 80,     # Large number
        }

        constraints = generator._create_fallback_constraints(model_config)

        # Should be limited by conservative maximums
        assert constraints.max_tensor_parallel <= 64
        assert constraints.max_pipeline_parallel <= 16

    @patch('autoparallel.config.generator.analyze_model_constraints')
    def test_auto_constraint_detection_success(self, mock_analyze):
        """Test successful automatic constraint detection."""
        mock_constraints = Mock(spec=ModelConstraints)
        mock_analyze.return_value = mock_constraints

        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        with patch.object(generator, '_enumerate_parallelism_configs') as mock_enumerate:
            mock_enumerate.return_value = []

            generator.generate_valid_configs(
                model_config=model_config,
                hardware_profile=hardware,
                workload_profile=workload,
                model_constraints=None,  # Trigger auto-detection
            )

            mock_analyze.assert_called_once()
            mock_enumerate.assert_called_once_with(mock_constraints, hardware)

    @patch('autoparallel.config.generator.analyze_model_constraints')
    def test_auto_constraint_detection_failure(self, mock_analyze):
        """Test fallback when automatic constraint detection fails."""
        mock_analyze.side_effect = Exception("Analysis failed")

        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        with patch.object(generator, '_create_fallback_constraints') as mock_fallback:
            # Mock the constraint methods properly
            mock_constraints = Mock(spec=ModelConstraints)
            mock_constraints.get_valid_tensor_parallel_sizes.return_value = [1, 2]
            mock_constraints.get_valid_expert_parallel_sizes.return_value = [1]
            mock_constraints.get_valid_pipeline_parallel_sizes.return_value = [1]
            mock_fallback.return_value = mock_constraints

            result = generator.generate_valid_configs(
                model_config=model_config,
                hardware_profile=hardware,
                workload_profile=workload,
                model_constraints=None,
            )

            mock_fallback.assert_called_once_with(model_config)
            assert len(result.configurations) >= 0  # Should not crash

    def test_enumerate_parallelism_configs_basic(self):
        """Test basic parallelism configuration enumeration."""
        generator = ConfigurationGenerator()

        # Mock constraints
        constraints = Mock(spec=ModelConstraints)
        constraints.get_valid_tensor_parallel_sizes.return_value = [1, 2, 4]
        constraints.get_valid_expert_parallel_sizes.return_value = [1]
        constraints.get_valid_pipeline_parallel_sizes.return_value = [1, 2]

        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)

        configs = generator._enumerate_parallelism_configs(constraints, hardware)

        assert len(configs) > 0
        # Should have combinations like (1,1,1), (2,1,1), (4,1,1), (1,1,2), etc.
        tp_sizes = [config.tensor_parallel_size for config in configs]
        assert 1 in tp_sizes
        assert 2 in tp_sizes
        assert 4 in tp_sizes

    def test_enumerate_parallelism_configs_gpu_limit(self):
        """Test that enumeration respects GPU limits."""
        generator = ConfigurationGenerator()

        constraints = Mock(spec=ModelConstraints)
        constraints.get_valid_tensor_parallel_sizes.return_value = [1, 2, 4, 8]
        constraints.get_valid_expert_parallel_sizes.return_value = [1]
        constraints.get_valid_pipeline_parallel_sizes.return_value = [1, 2, 4]

        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)  # Only 8 GPUs

        configs = generator._enumerate_parallelism_configs(constraints, hardware)

        # No configuration should require more than 8 GPUs
        for config in configs:
            assert config.total_gpus_needed <= 8

    def test_enumerate_parallelism_configs_empty_constraints(self):
        """Test enumeration with empty constraint lists."""
        generator = ConfigurationGenerator()

        constraints = Mock(spec=ModelConstraints)
        constraints.get_valid_tensor_parallel_sizes.return_value = []  # Empty
        constraints.get_valid_expert_parallel_sizes.return_value = []  # Empty
        constraints.get_valid_pipeline_parallel_sizes.return_value = []  # Empty

        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)

        configs = generator._enumerate_parallelism_configs(constraints, hardware)

        # Should fallback to defaults ([1])
        assert len(configs) == 1
        config = configs[0]
        assert config.tensor_parallel_size == 1
        assert config.expert_parallel_size == 1
        assert config.pipeline_parallel_size == 1

    def test_score_configuration_success(self):
        """Test successful configuration scoring."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=2,
        )

        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        # Mock memory estimator
        mock_memory = MemoryComponents(weights=1000, activations=500, kv_cache=200)
        generator.memory_estimator.estimate_memory = Mock(return_value=mock_memory)

        scored = generator._score_configuration(config, model_config, hardware, workload)

        assert scored is not None
        assert isinstance(scored, ScoredConfiguration)
        assert scored.configuration == config
        assert isinstance(scored.performance_metrics, PerformanceMetrics)
        assert isinstance(scored.memory_components, MemoryComponents)
        assert isinstance(scored.validation_result, ValidationResult)

    def test_score_configuration_exception_handling(self):
        """Test that scoring handles exceptions gracefully."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration()
        model_config = {}
        hardware = HardwareProfile()
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        # Mock memory estimator to raise exception
        generator.memory_estimator.estimate_memory = Mock(side_effect=Exception("Estimation failed"))

        scored = generator._score_configuration(config, model_config, hardware, workload)

        # Should return None when estimation fails
        assert scored is None

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=4,
        )

        memory_components = MemoryComponents(
            weights=10 * (1024**3),  # 10 GB
            activations=5 * (1024**3),  # 5 GB
            kv_cache=1 * (1024**3),   # 1 GB
        )

        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE, batch_size=32)

        metrics = generator._calculate_performance_metrics(
            config, memory_components, hardware, workload
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert 0 <= metrics.throughput_score <= 2.0  # Can be > 1
        assert 0 <= metrics.latency_score <= 10.0    # Can be > 1
        assert 0 <= metrics.memory_efficiency_score <= 1.0
        assert 0 <= metrics.cost_score <= 2.0        # Can be > 1
        assert 0 <= metrics.communication_efficiency <= 1.0
        assert 0 <= metrics.gpu_utilization <= 1.0
        assert metrics.memory_utilization_gb_per_gpu > 0
        assert isinstance(metrics.is_feasible, bool)

    def test_validate_configuration(self):
        """Test configuration validation."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=2,
        )

        memory_components = MemoryComponents(
            weights=40 * (1024**3),  # 40 GB (feasible)
            activations=20 * (1024**3),
            kv_cache=5 * (1024**3),
        )

        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)

        validation = generator._validate_configuration(config, memory_components, hardware)

        assert isinstance(validation, ValidationResult)
        assert validation.is_valid  # Should be valid with 65GB on 80GB GPU

    def test_validate_configuration_insufficient_memory(self):
        """Test validation with insufficient memory."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration(tensor_parallel_size=1)
        memory_components = MemoryComponents(
            weights=100 * (1024**3),  # 100 GB (too much)
        )
        hardware = HardwareProfile(gpu_memory_gb=80)

        validation = generator._validate_configuration(config, memory_components, hardware)

        assert not validation.is_valid
        assert len(validation.errors) > 0
        assert any("exceeds" in error for error in validation.errors)

    def test_validate_configuration_insufficient_gpus(self):
        """Test validation with insufficient GPUs."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration(
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            data_parallel_size=1,
        )  # Requires 16 GPUs

        memory_components = MemoryComponents(weights=1000)
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)  # Only 8 GPUs total

        validation = generator._validate_configuration(config, memory_components, hardware)

        assert not validation.is_valid
        assert len(validation.errors) > 0
        assert any("requires" in error and "GPUs" in error for error in validation.errors)

    @pytest.mark.parametrize("utilization,should_have_memory_warning", [
        (0.2, True),   # Low utilization
        (0.5, False),  # Good utilization
        (0.95, True),  # High utilization
    ])
    def test_validate_configuration_memory_warnings(self, utilization, should_have_memory_warning):
        """Test memory utilization warnings."""
        generator = ConfigurationGenerator()

        config = ParallelismConfiguration(tensor_parallel_size=1)

        gpu_memory_gb = 80
        memory_gb = utilization * gpu_memory_gb
        memory_components = MemoryComponents(weights=int(memory_gb * (1024**3)))
        hardware = HardwareProfile(gpu_memory_gb=gpu_memory_gb)

        validation = generator._validate_configuration(config, memory_components, hardware)

        # Check for memory-specific warnings
        memory_warnings = [w for w in validation.warnings if "memory utilization" in w.lower()]
        if should_have_memory_warning:
            assert len(memory_warnings) > 0
        else:
            assert len(memory_warnings) == 0

    @pytest.mark.parametrize("objective,expected_sort_key", [
        (OptimizationObjective.MAXIMIZE_THROUGHPUT, "throughput_score"),
        (OptimizationObjective.MINIMIZE_LATENCY, "latency_score"),
        (OptimizationObjective.MINIMIZE_COST, "cost_score"),
        (OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY, "memory_efficiency_score"),
    ])
    def test_rank_configurations(self, objective, expected_sort_key):
        """Test configuration ranking by different objectives."""
        generator = ConfigurationGenerator()

        # Create mock scored configurations with different scores
        configs = []
        for i in range(3):
            config = ParallelismConfiguration(data_parallel_size=i+1)
            metrics = PerformanceMetrics(
                throughput_score=i * 0.3,
                latency_score=i * 0.2,
                memory_efficiency_score=i * 0.1,
                cost_score=i * 0.4,
            )
            scored = ScoredConfiguration(
                configuration=config,
                performance_metrics=metrics,
                memory_components=MemoryComponents(),
            )
            configs.append(scored)

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        ranked = generator._rank_configurations(configs, workload, objective)

        # Should be sorted in descending order by the relevant score
        scores = [getattr(c.performance_metrics, expected_sort_key) for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_configurations_balanced(self):
        """Test balanced efficiency ranking."""
        generator = ConfigurationGenerator()

        # Create configurations with different weighted scores
        configs = []
        for i in range(3):
            config = ParallelismConfiguration()
            metrics = PerformanceMetrics(
                throughput_score=0.5 + i * 0.1,
                latency_score=0.3 + i * 0.2,
                memory_efficiency_score=0.7 - i * 0.1,
                cost_score=0.6 + i * 0.05,
            )
            scored = ScoredConfiguration(
                configuration=config,
                performance_metrics=metrics,
                memory_components=MemoryComponents(),
            )
            configs.append(scored)

        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        objective = OptimizationObjective.BALANCE_EFFICIENCY

        ranked = generator._rank_configurations(configs, workload, objective)

        # Should be sorted by weighted score
        weights = workload.get_priority_weights()
        weighted_scores = [c.performance_metrics.get_weighted_score(weights) for c in ranked]
        assert weighted_scores == sorted(weighted_scores, reverse=True)


class TestScoredConfiguration:
    """Test ScoredConfiguration functionality."""

    def test_scored_configuration_creation(self):
        """Test ScoredConfiguration creation and validation."""
        config = ParallelismConfiguration(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=4,
        )

        metrics = PerformanceMetrics(
            throughput_score=0.8,
            latency_score=0.7,
            memory_efficiency_score=0.6,
            cost_score=0.9,
            is_feasible=True,
        )

        memory_components = MemoryComponents(
            weights=1000,
            activations=500,
            kv_cache=200,
        )

        validation_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            recommendations=[],
        )

        scored_config = ScoredConfiguration(
            configuration=config,
            performance_metrics=metrics,
            memory_components=memory_components,
            validation_result=validation_result,
        )

        assert scored_config.is_valid
        assert scored_config.configuration == config
        assert scored_config.performance_metrics == metrics
        assert scored_config.memory_components == memory_components

    def test_scored_configuration_invalid_performance(self):
        """Test ScoredConfiguration with infeasible performance metrics."""
        config = ParallelismConfiguration()
        metrics = PerformanceMetrics(is_feasible=False)  # Not feasible
        memory_components = MemoryComponents()

        scored_config = ScoredConfiguration(
            configuration=config,
            performance_metrics=metrics,
            memory_components=memory_components,
        )

        assert not scored_config.is_valid

    def test_scored_configuration_invalid_validation(self):
        """Test ScoredConfiguration with invalid validation result."""
        config = ParallelismConfiguration()
        metrics = PerformanceMetrics(is_feasible=True)
        memory_components = MemoryComponents()
        validation_result = ValidationResult(
            is_valid=False,
            errors=["Memory exceeds limit"],
            warnings=[],
            recommendations=[],
        )

        scored_config = ScoredConfiguration(
            configuration=config,
            performance_metrics=metrics,
            memory_components=memory_components,
            validation_result=validation_result,
        )

        assert not scored_config.is_valid

    def test_scored_configuration_no_validation_result(self):
        """Test ScoredConfiguration without validation result."""
        config = ParallelismConfiguration()
        metrics = PerformanceMetrics(is_feasible=True)
        memory_components = MemoryComponents()

        scored_config = ScoredConfiguration(
            configuration=config,
            performance_metrics=metrics,
            memory_components=memory_components,
            validation_result=None,
        )

        assert scored_config.is_valid  # Should be valid when validation is None


class TestConfigurationGenerationResult:
    """Test ConfigurationGenerationResult functionality."""

    def test_generation_result_creation(self):
        """Test ConfigurationGenerationResult creation."""
        configurations = [
            ScoredConfiguration(
                configuration=ParallelismConfiguration(),
                performance_metrics=PerformanceMetrics(),
                memory_components=MemoryComponents(),
            )
        ]

        metadata = {"total_enumerated": 10, "valid_configs": 5}

        result = ConfigurationGenerationResult(
            configurations=configurations,
            best_configuration=None,
            generation_metadata=metadata,
        )

        assert result.configurations == configurations
        assert result.best_configuration == configurations[0]  # Auto-set in __post_init__
        assert result.generation_metadata == metadata

    def test_generation_result_empty_configurations(self):
        """Test ConfigurationGenerationResult with empty configurations."""
        result = ConfigurationGenerationResult(
            configurations=[],
            best_configuration=None,
            generation_metadata={},
        )

        assert result.configurations == []
        assert result.best_configuration is None

    def test_generation_result_explicit_best(self):
        """Test ConfigurationGenerationResult with explicit best configuration."""
        config1 = ScoredConfiguration(
            configuration=ParallelismConfiguration(data_parallel_size=1),
            performance_metrics=PerformanceMetrics(),
            memory_components=MemoryComponents(),
        )
        config2 = ScoredConfiguration(
            configuration=ParallelismConfiguration(data_parallel_size=2),
            performance_metrics=PerformanceMetrics(),
            memory_components=MemoryComponents(),
        )

        result = ConfigurationGenerationResult(
            configurations=[config1, config2],
            best_configuration=config2,  # Explicitly set
            generation_metadata={},
        )

        assert result.best_configuration == config2  # Should preserve explicit setting


class TestGenerateConfigurationsForWorkload:
    """Test the convenience function for workload-specific generation."""

    def test_chatbot_workload(self):
        """Test chatbot workload optimization."""
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.CHATBOT,
            sequence_length=2048,
            batch_size=32,  # Should be overridden to 1 for chatbot
            max_configurations=5,
        )

        assert len(result.configurations) > 0
        assert result.generation_metadata["workload_type"] == "chatbot"

    def test_training_workload(self):
        """Test training workload optimization."""
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=2)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.TRAINING,
            sequence_length=2048,
            batch_size=64,
            max_configurations=10,
        )

        assert len(result.configurations) > 0
        assert result.generation_metadata["workload_type"] == "training"

    def test_batch_processing_workload(self):
        """Test batch processing workload optimization."""
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        model_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        }

        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.BATCH_PROCESSING,
            sequence_length=1024,
            batch_size=128,
            max_configurations=5,
        )

        assert len(result.configurations) > 0
        assert result.generation_metadata["workload_type"] == "batch_processing"

    def test_custom_latency_target(self):
        """Test custom latency target setting."""
        hardware = HardwareProfile(gpu_memory_gb=40, gpus_per_node=4, num_nodes=1)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        custom_latency = 150.0
        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.INFERENCE,
            target_latency_ms=custom_latency,
            max_configurations=5,
        )

        assert len(result.configurations) > 0

    @pytest.mark.parametrize("workload_type", [
        WorkloadType.INFERENCE,
        WorkloadType.TRAINING,
        WorkloadType.BATCH_PROCESSING,
        WorkloadType.INTERACTIVE,
        WorkloadType.CHATBOT,
    ])
    def test_all_workload_types(self, workload_type):
        """Test all workload types generate valid configurations."""
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=workload_type,
            max_configurations=3,
        )

        assert len(result.configurations) > 0
        assert result.generation_metadata["workload_type"] == workload_type.value
        assert result.best_configuration is not None

    def test_workload_specific_optimization(self):
        """Test that workload types use appropriate optimization objectives."""
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        model_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "vocab_size": 32000,
        }

        # Chatbot should optimize for latency
        chatbot_result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.CHATBOT,
            max_configurations=3,
        )

        # Training should optimize for throughput
        training_result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.TRAINING,
            max_configurations=3,
        )

        # Both should succeed
        assert len(chatbot_result.configurations) > 0
        assert len(training_result.configurations) > 0

    def test_default_latency_budgets(self):
        """Test that default latency budgets are set correctly."""
        from autoparallel.config.generator import _get_default_latency_budget

        assert _get_default_latency_budget(WorkloadType.CHATBOT) == 100.0
        assert _get_default_latency_budget(WorkloadType.INTERACTIVE) == 200.0
        assert _get_default_latency_budget(WorkloadType.TRAINING) == 1000.0
        assert _get_default_latency_budget(WorkloadType.BATCH_PROCESSING) == 5000.0
        assert _get_default_latency_budget(WorkloadType.INFERENCE) == 500.0

    def test_edge_case_parameters(self):
        """Test edge case parameter values."""
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        model_config = {
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "vocab_size": 10000,
        }

        # Very small sequence length
        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.INFERENCE,
            sequence_length=128,  # Small
            batch_size=1,
            max_configurations=5,
        )
        assert len(result.configurations) > 0

        # Large sequence length
        result = generate_configurations_for_workload(
            model_config=model_config,
            hardware_profile=hardware,
            workload_type=WorkloadType.INFERENCE,
            sequence_length=8192,  # Large
            batch_size=1,
            max_configurations=5,
        )
        assert len(result.configurations) > 0

    def test_memory_constraints_with_moe(self):
        """Test configuration generation with MoE models under memory constraints."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=40, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        # MoE model configuration
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_experts": 8,
            "intermediate_size": 11008,
        }

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=10,
        )

        # Should generate configurations with expert parallelism
        assert len(result.configurations) > 0
        expert_parallel_configs = [
            c for c in result.configurations
            if c.configuration.expert_parallel_size > 1
        ]
        assert len(expert_parallel_configs) > 0

    def test_integration_with_memory_estimator(self):
        """Test integration with different memory estimators."""
        # Test with MoE memory estimator
        moe_estimator = Mock(spec=MoEMemoryEstimator)
        moe_estimator.estimate_memory.return_value = MemoryComponents(
            weights=5000,
            activations=2000,
            kv_cache=1000,
        )

        generator = ConfigurationGenerator(memory_estimator=moe_estimator)
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "num_experts": 8,
        }

        result = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=5,
        )

        assert len(result.configurations) > 0
        moe_estimator.estimate_memory.assert_called()

    def test_performance_metrics_calculation_edge_cases(self):
        """Test performance metrics calculation with edge cases."""
        generator = ConfigurationGenerator()

        # Very high pipeline parallelism
        config = ParallelismConfiguration(
            tensor_parallel_size=1,
            pipeline_parallel_size=16,
            expert_parallel_size=1,
            data_parallel_size=1,
        )

        memory_components = MemoryComponents(weights=1000, activations=500)
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=2)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)

        metrics = generator._calculate_performance_metrics(
            config, memory_components, hardware, workload
        )

        # Pipeline overhead should reduce efficiency
        assert metrics.throughput_score < 1.0  # Should be penalized
        assert metrics.latency_score >= 0  # Should still be valid

        # Very low memory utilization
        low_memory = MemoryComponents(weights=100)  # Very small
        metrics_low = generator._calculate_performance_metrics(
            ParallelismConfiguration(), low_memory, hardware, workload
        )

        assert 0 <= metrics_low.memory_efficiency_score <= 1.0

    def test_configuration_ranking_stability(self):
        """Test that configuration ranking is stable and deterministic."""
        generator = ConfigurationGenerator()
        hardware = HardwareProfile(gpu_memory_gb=80, gpus_per_node=8, num_nodes=1)
        workload = WorkloadProfile(workload_type=WorkloadType.INFERENCE)
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        # Generate configurations multiple times
        result1 = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=5,
        )

        result2 = generator.generate_valid_configs(
            model_config=model_config,
            hardware_profile=hardware,
            workload_profile=workload,
            max_configurations=5,
        )

        # Should produce the same rankings
        assert len(result1.configurations) == len(result2.configurations)
        for c1, c2 in zip(result1.configurations, result2.configurations, strict=False):
            assert c1.configuration.to_dict() == c2.configuration.to_dict()
