"""Configuration ranking and optimization algorithms for autoparallel."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from autoparallel.constraints.analyzer import ModelConstraints
from autoparallel.memory.components import MemoryComponents
from autoparallel.memory.estimator import MemoryEstimator


class WorkloadType(Enum):
    """Different workload types that require different optimization strategies."""

    INFERENCE = "inference"
    TRAINING = "training"
    BATCH_PROCESSING = "batch_processing"
    INTERACTIVE = "interactive"
    CHATBOT = "chatbot"


class OptimizationObjective(Enum):
    """Different optimization objectives for configuration ranking."""

    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_COST = "minimize_cost"
    BALANCE_EFFICIENCY = "balance_efficiency"
    MAXIMIZE_MEMORY_EFFICIENCY = "maximize_memory_efficiency"


@dataclass
class HardwareProfile:
    """Hardware characteristics that affect optimization decisions."""

    gpu_model: str = "H100"
    """GPU model (affects performance characteristics)"""

    gpu_memory_gb: float = 80.0
    """GPU memory capacity in GB"""

    gpus_per_node: int = 8
    """Number of GPUs per node"""

    num_nodes: int = 1
    """Number of nodes in cluster"""

    intra_node_bandwidth_gbps: float = 900.0
    """Intra-node bandwidth (NVSwitch) in GB/s"""

    inter_node_bandwidth_gbps: float = 200.0
    """Inter-node bandwidth (RDMA) in GB/s"""

    network_topology: str = "fat_tree"
    """Network topology type"""

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs in cluster."""
        return self.gpus_per_node * self.num_nodes

    @property
    def total_memory_gb(self) -> float:
        """Total memory across all GPUs."""
        return self.gpu_memory_gb * self.total_gpus

    def get_communication_efficiency(
        self, tensor_parallel_size: int, pipeline_parallel_size: int
    ) -> float:
        """Estimate communication efficiency for given parallelism configuration."""
        # Tensor parallelism communication efficiency
        tp_efficiency = 1.0
        if tensor_parallel_size > self.gpus_per_node:
            # Inter-node TP is less efficient
            tp_efficiency = 0.6
        elif tensor_parallel_size > 1:
            # Intra-node TP is highly efficient
            tp_efficiency = 0.9

        # Pipeline parallelism communication efficiency
        pp_efficiency = 1.0
        if pipeline_parallel_size > 1:
            # PP has pipeline bubbles and sequential dependency
            pp_efficiency = 0.8

        return tp_efficiency * pp_efficiency


@dataclass
class WorkloadProfile:
    """Workload characteristics that influence optimization decisions."""

    workload_type: WorkloadType
    """Type of workload"""

    batch_size: int = 32
    """Expected batch size"""

    sequence_length: int = 2048
    """Expected sequence length"""

    requests_per_second: float = 100.0
    """Expected request rate"""

    latency_budget_ms: float = 100.0
    """P99 latency budget in milliseconds"""

    throughput_target: float = 1000.0
    """Target throughput (tokens/second)"""

    cost_budget_per_hour: float = 100.0
    """Cost budget per hour in dollars"""

    is_training: bool = False
    """Whether this is a training workload"""

    def get_priority_weights(self) -> dict[str, float]:
        """Get optimization priority weights based on workload type."""
        if self.workload_type == WorkloadType.INFERENCE:
            return {
                "throughput": 0.4,
                "latency": 0.3,
                "memory_efficiency": 0.2,
                "cost": 0.1,
            }
        elif self.workload_type == WorkloadType.TRAINING:
            return {
                "throughput": 0.6,
                "latency": 0.1,
                "memory_efficiency": 0.2,
                "cost": 0.1,
            }
        elif self.workload_type == WorkloadType.BATCH_PROCESSING:
            return {
                "throughput": 0.7,
                "latency": 0.1,
                "memory_efficiency": 0.1,
                "cost": 0.1,
            }
        elif self.workload_type == WorkloadType.INTERACTIVE:
            return {
                "throughput": 0.2,
                "latency": 0.6,
                "memory_efficiency": 0.1,
                "cost": 0.1,
            }
        elif self.workload_type == WorkloadType.CHATBOT:
            return {
                "throughput": 0.3,
                "latency": 0.5,
                "memory_efficiency": 0.1,
                "cost": 0.1,
            }
        else:
            # Default balanced weights
            return {
                "throughput": 0.25,
                "latency": 0.25,
                "memory_efficiency": 0.25,
                "cost": 0.25,
            }


@dataclass
class ParallelismConfiguration:
    """Complete parallelism configuration with all parameters."""

    tensor_parallel_size: int = 1
    """Tensor parallelism size"""

    pipeline_parallel_size: int = 1
    """Pipeline parallelism size"""

    expert_parallel_size: int = 1
    """Expert parallelism size (for MoE models)"""

    data_parallel_size: int = 1
    """Data parallelism size"""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                f"pipeline_parallel_size must be >= 1, "
                f"got {self.pipeline_parallel_size}"
            )
        if self.expert_parallel_size < 1:
            raise ValueError(
                f"expert_parallel_size must be >= 1, got {self.expert_parallel_size}"
            )
        if self.data_parallel_size < 1:
            raise ValueError(
                f"data_parallel_size must be >= 1, got {self.data_parallel_size}"
            )

    @property
    def total_gpus_needed(self) -> int:
        """Total number of GPUs needed for this configuration."""
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.data_parallel_size
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format."""
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a configuration."""

    throughput_score: float = 0.0
    """Throughput score (higher is better)"""

    latency_score: float = 0.0
    """Latency score (higher is better, lower latency)"""

    memory_efficiency_score: float = 0.0
    """Memory efficiency score (higher is better)"""

    cost_score: float = 0.0
    """Cost efficiency score (higher is better, lower cost)"""

    communication_efficiency: float = 0.0
    """Communication efficiency score"""

    gpu_utilization: float = 0.0
    """GPU utilization percentage"""

    memory_utilization_gb_per_gpu: float = 0.0
    """Memory utilization per GPU in GB"""

    is_feasible: bool = True
    """Whether this configuration is feasible"""

    def get_weighted_score(self, weights: dict[str, float]) -> float:
        """Calculate weighted composite score."""
        return (
            self.throughput_score * weights.get("throughput", 0.0)
            + self.latency_score * weights.get("latency", 0.0)
            + self.memory_efficiency_score * weights.get("memory_efficiency", 0.0)
            + self.cost_score * weights.get("cost", 0.0)
        )


class ConfigurationOptimizer(ABC):
    """Abstract base class for configuration optimization algorithms."""

    def __init__(
        self,
        memory_estimator: MemoryEstimator,
        model_constraints: ModelConstraints,
        hardware_profile: HardwareProfile,
    ):
        """Initialize optimizer.

        Args:
            memory_estimator: Memory estimator for the model
            model_constraints: Model architecture constraints
            hardware_profile: Hardware characteristics
        """
        self.memory_estimator = memory_estimator
        self.model_constraints = model_constraints
        self.hardware_profile = hardware_profile

    @abstractmethod
    def rank_configurations(
        self,
        configurations: list[ParallelismConfiguration],
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_EFFICIENCY
        ),
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Rank configurations by performance for given workload.

        Args:
            configurations: List of configurations to rank
            workload_profile: Workload characteristics
            optimization_objective: Primary optimization objective

        Returns:
            List of (configuration, metrics) tuples, ranked best to worst
        """
        pass

    @abstractmethod
    def optimize_for_workload(
        self,
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_EFFICIENCY
        ),
        max_configurations: int = 100,
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Find optimal configurations for a specific workload.

        Args:
            workload_profile: Workload characteristics
            optimization_objective: Primary optimization objective
            max_configurations: Maximum number of configurations to evaluate

        Returns:
            List of optimal (configuration, metrics) tuples, ranked best to worst
        """
        pass

    def _calculate_performance_metrics(
        self,
        config: ParallelismConfiguration,
        workload_profile: WorkloadProfile,
        model_config: dict[str, Any],
    ) -> PerformanceMetrics:
        """Calculate performance metrics for a configuration."""
        # Estimate memory usage
        memory_components = self.memory_estimator.estimate_memory(
            model_config=model_config,
            sequence_length=workload_profile.sequence_length,
            batch_size=workload_profile.batch_size,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            expert_parallel_size=config.expert_parallel_size,
            is_training=workload_profile.is_training,
        )

        # Check if configuration fits in memory
        memory_per_gpu = memory_components.total_memory / (1024**3)  # Convert to GB
        is_feasible = memory_per_gpu <= self.hardware_profile.gpu_memory_gb

        # Calculate throughput score
        throughput_score = self._calculate_throughput_score(
            config, workload_profile, memory_components
        )

        # Calculate latency score
        latency_score = self._calculate_latency_score(
            config, workload_profile, memory_components
        )

        # Calculate memory efficiency score
        memory_efficiency_score = self._calculate_memory_efficiency_score(
            config, memory_components
        )

        # Calculate cost score
        cost_score = self._calculate_cost_score(config, workload_profile)

        # Calculate communication efficiency
        communication_efficiency = self.hardware_profile.get_communication_efficiency(
            config.tensor_parallel_size, config.pipeline_parallel_size
        )

        # Calculate GPU utilization
        gpu_utilization = min(memory_per_gpu / self.hardware_profile.gpu_memory_gb, 1.0)

        return PerformanceMetrics(
            throughput_score=throughput_score,
            latency_score=latency_score,
            memory_efficiency_score=memory_efficiency_score,
            cost_score=cost_score,
            communication_efficiency=communication_efficiency,
            gpu_utilization=gpu_utilization,
            memory_utilization_gb_per_gpu=memory_per_gpu,
            is_feasible=is_feasible,
        )

    def _calculate_throughput_score(
        self,
        config: ParallelismConfiguration,
        workload_profile: WorkloadProfile,
        memory_components: MemoryComponents,
    ) -> float:
        """Calculate throughput score for configuration."""
        # Base throughput from data parallelism
        base_throughput = config.data_parallel_size * workload_profile.batch_size

        # Communication efficiency affects throughput
        comm_efficiency = self.hardware_profile.get_communication_efficiency(
            config.tensor_parallel_size, config.pipeline_parallel_size
        )

        # Pipeline parallelism has efficiency penalty due to bubbles
        pipeline_efficiency = 1.0 - (0.1 * (config.pipeline_parallel_size - 1))
        pipeline_efficiency = max(pipeline_efficiency, 0.5)  # Minimum 50% efficiency

        # Memory utilization affects throughput (too low or too high is bad)
        memory_per_gpu = memory_components.total_memory / (1024**3)
        memory_util = memory_per_gpu / self.hardware_profile.gpu_memory_gb
        memory_efficiency = 1.0 - abs(memory_util - 0.85)  # Optimal at 85%
        memory_efficiency = max(memory_efficiency, 0.1)

        throughput_score = (
            base_throughput * comm_efficiency * pipeline_efficiency * memory_efficiency
        )

        # Normalize by workload target
        return min(throughput_score / workload_profile.throughput_target, 2.0)

    def _calculate_latency_score(
        self,
        config: ParallelismConfiguration,
        workload_profile: WorkloadProfile,
        memory_components: MemoryComponents,
    ) -> float:
        """Calculate latency score for configuration."""
        # Base latency factors
        base_latency = 10.0  # Base latency in ms

        # Tensor parallelism can reduce latency through parallel computation
        tp_latency_reduction = 1.0 / (config.tensor_parallel_size**0.5)

        # Pipeline parallelism increases latency due to sequential stages
        pp_latency_penalty = 1.0 + (0.2 * (config.pipeline_parallel_size - 1))

        # Communication overhead
        comm_latency = 0.0
        if config.tensor_parallel_size > 1:
            if config.tensor_parallel_size > self.hardware_profile.gpus_per_node:
                comm_latency += 5.0  # Inter-node communication penalty
            else:
                comm_latency += 1.0  # Intra-node communication

        estimated_latency = (
            base_latency * tp_latency_reduction * pp_latency_penalty + comm_latency
        )

        # Convert to score (lower latency = higher score)
        if estimated_latency > workload_profile.latency_budget_ms:
            return 0.1  # Penalty for exceeding budget
        else:
            return workload_profile.latency_budget_ms / estimated_latency

    def _calculate_memory_efficiency_score(
        self,
        config: ParallelismConfiguration,
        memory_components: MemoryComponents,
    ) -> float:
        """Calculate memory efficiency score for configuration."""
        memory_per_gpu = memory_components.total_memory / (1024**3)
        memory_utilization = memory_per_gpu / self.hardware_profile.gpu_memory_gb

        # Optimal memory utilization is around 85%
        optimal_utilization = 0.85
        efficiency = 1.0 - abs(memory_utilization - optimal_utilization)

        # Penalty for over-utilization (risk of OOM)
        if memory_utilization > 0.95:
            efficiency *= 0.1

        # Penalty for severe under-utilization
        if memory_utilization < 0.3:
            efficiency *= 0.5

        return max(efficiency, 0.0)

    def _calculate_cost_score(
        self,
        config: ParallelismConfiguration,
        workload_profile: WorkloadProfile,
    ) -> float:
        """Calculate cost efficiency score for configuration."""
        # Simple cost model: cost per GPU hour
        cost_per_gpu_hour = 3.0  # Example: $3/GPU/hour

        total_cost_per_hour = config.total_gpus_needed * cost_per_gpu_hour

        # Cost efficiency based on throughput per dollar
        if total_cost_per_hour == 0:
            return 0.0

        throughput_estimate = config.data_parallel_size * workload_profile.batch_size
        cost_efficiency = throughput_estimate / total_cost_per_hour

        # Normalize and cap the score
        return min(cost_efficiency / 10.0, 2.0)


class GreedyConfigurationOptimizer(ConfigurationOptimizer):
    """Greedy optimization algorithm that prioritizes single objectives."""

    def rank_configurations(
        self,
        configurations: list[ParallelismConfiguration],
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_EFFICIENCY
        ),
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Rank configurations using greedy approach."""
        # Dummy model config for now - in real implementation
        # this would come from generator
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        results = []
        for config in configurations:
            metrics = self._calculate_performance_metrics(
                config, workload_profile, model_config
            )

            # Filter out infeasible configurations
            if not metrics.is_feasible:
                continue

            results.append((config, metrics))

        # Sort by optimization objective
        if optimization_objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            results.sort(key=lambda x: x[1].throughput_score, reverse=True)
        elif optimization_objective == OptimizationObjective.MINIMIZE_LATENCY:
            results.sort(key=lambda x: x[1].latency_score, reverse=True)
        elif optimization_objective == OptimizationObjective.MINIMIZE_COST:
            results.sort(key=lambda x: x[1].cost_score, reverse=True)
        elif optimization_objective == OptimizationObjective.MAXIMIZE_MEMORY_EFFICIENCY:
            results.sort(key=lambda x: x[1].memory_efficiency_score, reverse=True)
        else:  # BALANCE_EFFICIENCY
            weights = workload_profile.get_priority_weights()
            results.sort(key=lambda x: x[1].get_weighted_score(weights), reverse=True)

        return results

    def optimize_for_workload(
        self,
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_EFFICIENCY
        ),
        max_configurations: int = 100,
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Find optimal configurations using greedy search."""
        # Generate all valid configurations within constraints
        configurations = self._generate_valid_configurations(max_configurations)

        # Rank them
        return self.rank_configurations(
            configurations, workload_profile, optimization_objective
        )

    def _generate_valid_configurations(
        self, max_configurations: int
    ) -> list[ParallelismConfiguration]:
        """Generate valid parallelism configurations."""
        configurations = []
        total_gpus = self.hardware_profile.total_gpus

        # Get valid sizes from model constraints
        valid_tp = self.model_constraints.get_valid_tensor_parallel_sizes(total_gpus)
        valid_ep = self.model_constraints.get_valid_expert_parallel_sizes(total_gpus)
        valid_pp = self.model_constraints.get_valid_pipeline_parallel_sizes(
            self.hardware_profile.num_nodes
        )

        # Generate combinations
        for tp_size in valid_tp:
            for ep_size in valid_ep:
                for pp_size in valid_pp:
                    # Calculate required GPUs for parallelism
                    parallel_gpus = tp_size * ep_size * pp_size

                    if parallel_gpus > total_gpus:
                        continue

                    # Calculate data parallelism
                    dp_size = total_gpus // parallel_gpus

                    config = ParallelismConfiguration(
                        tensor_parallel_size=tp_size,
                        pipeline_parallel_size=pp_size,
                        expert_parallel_size=ep_size,
                        data_parallel_size=dp_size,
                    )

                    configurations.append(config)

                    if len(configurations) >= max_configurations:
                        return configurations

        return configurations


class MultiObjectiveConfigurationOptimizer(ConfigurationOptimizer):
    """Multi-objective optimization using Pareto efficiency."""

    def rank_configurations(
        self,
        configurations: list[ParallelismConfiguration],
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_EFFICIENCY
        ),
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Rank configurations using multi-objective optimization."""
        # Similar to greedy but with Pareto front analysis
        model_config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
        }

        results = []
        for config in configurations:
            metrics = self._calculate_performance_metrics(
                config, workload_profile, model_config
            )

            if not metrics.is_feasible:
                continue

            results.append((config, metrics))

        # Find Pareto-optimal solutions
        pareto_front = self._find_pareto_front(results)

        # Rank Pareto-optimal solutions by workload preference
        weights = workload_profile.get_priority_weights()
        pareto_front.sort(key=lambda x: x[1].get_weighted_score(weights), reverse=True)

        # Add non-Pareto solutions at the end
        non_pareto = [r for r in results if r not in pareto_front]
        non_pareto.sort(key=lambda x: x[1].get_weighted_score(weights), reverse=True)

        return pareto_front + non_pareto

    def optimize_for_workload(
        self,
        workload_profile: WorkloadProfile,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_EFFICIENCY
        ),
        max_configurations: int = 100,
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Find optimal configurations using multi-objective optimization."""
        configurations = self._generate_valid_configurations(max_configurations)
        return self.rank_configurations(
            configurations, workload_profile, optimization_objective
        )

    def _find_pareto_front(
        self,
        results: list[tuple[ParallelismConfiguration, PerformanceMetrics]],
    ) -> list[tuple[ParallelismConfiguration, PerformanceMetrics]]:
        """Find Pareto-optimal configurations."""
        pareto_front = []

        for i, (config_i, metrics_i) in enumerate(results):
            is_dominated = False

            for j, (_config_j, metrics_j) in enumerate(results):
                if i == j:
                    continue

                # Check if metrics_j dominates metrics_i
                if self._dominates(metrics_j, metrics_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append((config_i, metrics_i))

        return pareto_front

    def _dominates(
        self, metrics_a: PerformanceMetrics, metrics_b: PerformanceMetrics
    ) -> bool:
        """Check if metrics_a dominates metrics_b.

        All objectives >= and at least one >.
        """
        objectives_a = [
            metrics_a.throughput_score,
            metrics_a.latency_score,
            metrics_a.memory_efficiency_score,
            metrics_a.cost_score,
        ]

        objectives_b = [
            metrics_b.throughput_score,
            metrics_b.latency_score,
            metrics_b.memory_efficiency_score,
            metrics_b.cost_score,
        ]

        # All objectives must be >= (not worse)
        all_ge = all(a >= b for a, b in zip(objectives_a, objectives_b, strict=False))

        # At least one objective must be > (strictly better)
        any_gt = any(a > b for a, b in zip(objectives_a, objectives_b, strict=False))

        return all_ge and any_gt

    def _generate_valid_configurations(
        self, max_configurations: int
    ) -> list[ParallelismConfiguration]:
        """Generate valid parallelism configurations."""
        # Same implementation as greedy optimizer
        configurations = []
        total_gpus = self.hardware_profile.total_gpus

        valid_tp = self.model_constraints.get_valid_tensor_parallel_sizes(total_gpus)
        valid_ep = self.model_constraints.get_valid_expert_parallel_sizes(total_gpus)
        valid_pp = self.model_constraints.get_valid_pipeline_parallel_sizes(
            self.hardware_profile.num_nodes
        )

        for tp_size in valid_tp:
            for ep_size in valid_ep:
                for pp_size in valid_pp:
                    parallel_gpus = tp_size * ep_size * pp_size

                    if parallel_gpus > total_gpus:
                        continue

                    dp_size = total_gpus // parallel_gpus

                    config = ParallelismConfiguration(
                        tensor_parallel_size=tp_size,
                        pipeline_parallel_size=pp_size,
                        expert_parallel_size=ep_size,
                        data_parallel_size=dp_size,
                    )

                    configurations.append(config)

                    if len(configurations) >= max_configurations:
                        return configurations

        return configurations


def create_optimizer(
    optimizer_type: str,
    memory_estimator: MemoryEstimator,
    model_constraints: ModelConstraints,
    hardware_profile: HardwareProfile,
) -> ConfigurationOptimizer:
    """Factory function to create optimizers.

    Args:
        optimizer_type: Type of optimizer ("greedy" or "multi_objective")
        memory_estimator: Memory estimator for the model
        model_constraints: Model architecture constraints
        hardware_profile: Hardware characteristics

    Returns:
        ConfigurationOptimizer instance

    Raises:
        ValueError: If optimizer_type is not recognized
    """
    if optimizer_type == "greedy":
        return GreedyConfigurationOptimizer(
            memory_estimator, model_constraints, hardware_profile
        )
    elif optimizer_type == "multi_objective":
        return MultiObjectiveConfigurationOptimizer(
            memory_estimator, model_constraints, hardware_profile
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
