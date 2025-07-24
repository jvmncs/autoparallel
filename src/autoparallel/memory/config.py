"""Memory configuration for autoparallel estimation."""

from dataclasses import dataclass
from typing import Literal

QuantizationFormat = Literal[
    "fp16", "bf16", "fp8", "int8", "int4", "gptq", "awq", "bitsandbytes"
]


@dataclass
class MemoryConfig:
    """Configuration for memory estimation.

    This class defines the parameters used to control memory estimation
    behavior across the autoparallel library.
    """

    # Memory utilization bounds
    utilization_bound: float = 0.85
    """Maximum GPU memory utilization (0.0-1.0). Default: 0.85 (85%)"""

    # Memory overhead factors
    fragmentation_overhead: float = 0.10
    """Memory fragmentation overhead factor. Default: 0.10 (10%)"""

    safety_margin: float = 0.05
    """Safety margin for memory calculations. Default: 0.05 (5%)"""

    # Model configuration
    quantization_format: QuantizationFormat = "fp16"
    """Model quantization format. Default: fp16"""

    # KV cache configuration
    min_kv_cache_fraction: float = 0.05
    """Minimum fraction of GPU memory for KV cache. Default: 0.05 (5%)"""

    # CUDA graphs configuration
    cuda_graph_overhead_mb: float = 512.0
    """CUDA graph memory overhead in MB. Default: 512MB"""

    # Optimizer state configuration
    optimizer_memory_fraction: float = 2.0
    """Optimizer memory as fraction of model weights. Default: 2.0 (200%)"""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.utilization_bound <= 1.0:
            raise ValueError(
                f"utilization_bound must be in (0.0, 1.0], got {self.utilization_bound}"
            )

        if not 0.0 <= self.fragmentation_overhead < 1.0:
            raise ValueError(
                f"fragmentation_overhead must be in [0.0, 1.0), "
                f"got {self.fragmentation_overhead}"
            )

        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError(
                f"safety_margin must be in [0.0, 1.0), got {self.safety_margin}"
            )

        if not 0.0 <= self.min_kv_cache_fraction <= 1.0:
            raise ValueError(
                f"min_kv_cache_fraction must be in [0.0, 1.0], "
                f"got {self.min_kv_cache_fraction}"
            )

        if self.cuda_graph_overhead_mb < 0:
            raise ValueError(
                f"cuda_graph_overhead_mb must be >= 0, "
                f"got {self.cuda_graph_overhead_mb}"
            )

        if self.optimizer_memory_fraction < 0:
            raise ValueError(
                f"optimizer_memory_fraction must be >= 0, "
                f"got {self.optimizer_memory_fraction}"
            )

    @property
    def effective_utilization_bound(self) -> float:
        """Get effective utilization bound after accounting for safety margin."""
        return self.utilization_bound * (1.0 - self.safety_margin)

    @property
    def quantization_dtype_bytes(self) -> float:
        """Get bytes per parameter for the quantization format."""
        dtype_bytes = {
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
            "int4": 0.5,
            "gptq": 0.5,  # Assume 4-bit GPTQ
            "awq": 0.5,  # Assume 4-bit AWQ
            "bitsandbytes": 1.0,  # Assume 8-bit
        }
        return dtype_bytes[self.quantization_format]
