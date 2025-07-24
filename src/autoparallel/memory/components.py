"""Memory components for detailed memory breakdown."""

from dataclasses import dataclass


@dataclass
class MemoryComponents:
    """Detailed breakdown of memory usage components.

    This class provides a comprehensive breakdown of all memory components
    required for model execution, including model weights, activations,
    KV cache, CUDA graphs, optimizer states, and overhead.
    """

    # Core model components (bytes)
    weights: int = 0
    """Model weights memory usage in bytes"""

    activations: int = 0
    """Activation memory usage in bytes"""

    kv_cache: int = 0
    """Key-Value cache memory usage in bytes"""

    # Framework-specific components (bytes)
    cuda_graphs: int = 0
    """CUDA graph memory usage in bytes"""

    optimizer_states: int = 0
    """Optimizer state memory usage in bytes (for training)"""

    # Overhead components (bytes)
    fragmentation_overhead: int = 0
    """Memory fragmentation overhead in bytes"""

    framework_overhead: int = 0
    """Framework-specific overhead in bytes"""

    safety_margin: int = 0
    """Safety margin memory in bytes"""

    def __post_init__(self) -> None:
        """Validate memory component values."""
        components = [
            self.weights,
            self.activations,
            self.kv_cache,
            self.cuda_graphs,
            self.optimizer_states,
            self.fragmentation_overhead,
            self.framework_overhead,
            self.safety_margin,
        ]

        for component in components:
            if component < 0:
                raise ValueError(
                    f"Memory components must be non-negative, got {component}"
                )

    @property
    def total_memory(self) -> int:
        """Total memory usage across all components."""
        return (
            self.weights
            + self.activations
            + self.kv_cache
            + self.cuda_graphs
            + self.optimizer_states
            + self.fragmentation_overhead
            + self.framework_overhead
            + self.safety_margin
        )

    @property
    def model_memory(self) -> int:
        """Core model memory (weights + activations + KV cache)."""
        return self.weights + self.activations + self.kv_cache

    @property
    def overhead_memory(self) -> int:
        """Total overhead memory."""
        return (
            self.fragmentation_overhead + self.framework_overhead + self.safety_margin
        )

    @property
    def framework_memory(self) -> int:
        """Framework-specific memory (CUDA graphs + optimizer states)."""
        return self.cuda_graphs + self.optimizer_states

    def to_gb(self) -> "MemoryComponents":
        """Convert all memory values to GB."""
        gb_factor = 1024**3
        return MemoryComponents(
            weights=int(self.weights / gb_factor),
            activations=int(self.activations / gb_factor),
            kv_cache=int(self.kv_cache / gb_factor),
            cuda_graphs=int(self.cuda_graphs / gb_factor),
            optimizer_states=int(self.optimizer_states / gb_factor),
            fragmentation_overhead=int(self.fragmentation_overhead / gb_factor),
            framework_overhead=int(self.framework_overhead / gb_factor),
            safety_margin=int(self.safety_margin / gb_factor),
        )

    def to_mb(self) -> "MemoryComponents":
        """Convert all memory values to MB."""
        mb_factor = 1024**2
        return MemoryComponents(
            weights=int(self.weights / mb_factor),
            activations=int(self.activations / mb_factor),
            kv_cache=int(self.kv_cache / mb_factor),
            cuda_graphs=int(self.cuda_graphs / mb_factor),
            optimizer_states=int(self.optimizer_states / mb_factor),
            fragmentation_overhead=int(self.fragmentation_overhead / mb_factor),
            framework_overhead=int(self.framework_overhead / mb_factor),
            safety_margin=int(self.safety_margin / mb_factor),
        )

    def breakdown_percentages(self) -> dict[str, float]:
        """Get memory breakdown as percentages of total memory."""
        if self.total_memory == 0:
            return dict.fromkeys(self._component_names(), 0.0)

        return {
            "weights": (self.weights / self.total_memory) * 100,
            "activations": (self.activations / self.total_memory) * 100,
            "kv_cache": (self.kv_cache / self.total_memory) * 100,
            "cuda_graphs": (self.cuda_graphs / self.total_memory) * 100,
            "optimizer_states": (self.optimizer_states / self.total_memory) * 100,
            "fragmentation_overhead": (self.fragmentation_overhead / self.total_memory)
            * 100,
            "framework_overhead": (self.framework_overhead / self.total_memory) * 100,
            "safety_margin": (self.safety_margin / self.total_memory) * 100,
        }

    def _component_names(self) -> list[str]:
        """Get list of component names."""
        return [
            "weights",
            "activations",
            "kv_cache",
            "cuda_graphs",
            "optimizer_states",
            "fragmentation_overhead",
            "framework_overhead",
            "safety_margin",
        ]

    def scale_by_parallelism(
        self,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
    ) -> "MemoryComponents":
        """Scale memory components based on parallelism configuration.

        Args:
            tensor_parallel_size: Tensor parallelism size
            pipeline_parallel_size: Pipeline parallelism size
            data_parallel_size: Data parallelism size

        Returns:
            New MemoryComponents with scaled values
        """
        # Weights are split across tensor parallel devices
        scaled_weights = self.weights // tensor_parallel_size

        # Activations are split across tensor parallel devices for most layers
        scaled_activations = self.activations // tensor_parallel_size

        # KV cache is split across tensor parallel devices
        scaled_kv_cache = self.kv_cache // tensor_parallel_size

        # CUDA graphs and optimizer states typically don't scale with TP
        scaled_cuda_graphs = self.cuda_graphs
        scaled_optimizer_states = self.optimizer_states // tensor_parallel_size

        # Overhead scales proportionally
        scaled_fragmentation = self.fragmentation_overhead // tensor_parallel_size
        scaled_framework = self.framework_overhead
        scaled_safety = self.safety_margin // tensor_parallel_size

        return MemoryComponents(
            weights=scaled_weights,
            activations=scaled_activations,
            kv_cache=scaled_kv_cache,
            cuda_graphs=scaled_cuda_graphs,
            optimizer_states=scaled_optimizer_states,
            fragmentation_overhead=scaled_fragmentation,
            framework_overhead=scaled_framework,
            safety_margin=scaled_safety,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        mb_components = self.to_mb()
        return (
            f"MemoryComponents(\n"
            f"  weights: {mb_components.weights} MB\n"
            f"  activations: {mb_components.activations} MB\n"
            f"  kv_cache: {mb_components.kv_cache} MB\n"
            f"  cuda_graphs: {mb_components.cuda_graphs} MB\n"
            f"  optimizer_states: {mb_components.optimizer_states} MB\n"
            f"  fragmentation_overhead: {mb_components.fragmentation_overhead} MB\n"
            f"  framework_overhead: {mb_components.framework_overhead} MB\n"
            f"  safety_margin: {mb_components.safety_margin} MB\n"
            f"  total: {mb_components.total_memory} MB\n"
            f")"
        )
