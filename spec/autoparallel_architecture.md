# AutoParallel Simplified Architecture

## Core Mission

Given a Hugging Face model ID and GPU cluster specification, determine optimal parallelism configurations (TP/PP/EP/DP) for LLM inference.

## Domain Model

### Core Objects
```python
@dataclass
class HardwareSpec:
    """Single hardware specification."""
    gpu_memory_gb: float
    gpus_per_node: int  
    num_nodes: int
    gpu_architecture: str  # "A100", "H100", etc.

@dataclass
class ModelProfile:
    """Model characteristics needed for parallelism analysis."""
    config: PretrainedConfig  # Hugging Face config
    is_moe: bool
    memory_estimator: MemoryEstimator

@dataclass  
class ParallelismConfig:
    """A valid parallelism configuration."""
    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int  # 1 for non-MoE
    data_parallel: int    # computed: total_gpus / (tp * pp * ep)

@dataclass
class MemoryBreakdown:
    """Memory usage breakdown in GB."""
    weights: float
    activations: float
    kv_cache: float
    cuda_graphs: float
    framework_overhead: float
    total: float
    
    def fits_in_gpu(self, gpu_memory_gb: float, utilization: float = 0.9) -> bool:
        return self.total <= gpu_memory_gb * utilization
```

## Architecture Components

### 1. Constraint Analysis (`constraints.py`)
```python
def valid_tensor_parallel_sizes(model_config: PretrainedConfig, max_size: int) -> List[int]:
    """Return valid TP sizes based on attention heads divisibility."""
    
def valid_pipeline_parallel_sizes(model_config: PretrainedConfig, max_size: int) -> List[int]:
    """Return valid PP sizes based on layer count."""
    
def valid_expert_parallel_sizes(model_config: PretrainedConfig, max_size: int) -> List[int]:
    """Return valid EP sizes for MoE models (1 for dense models)."""
```

### 2. Memory Estimation (`memory.py`)
```python
def estimate_memory(
    model_config: PretrainedConfig,
    sequence_length: int = 2048,
    batch_size: int = 32,
    tensor_parallel: int = 1,
    quantization_bytes: int = 2,
    enable_kv_cache: bool = True,
    cuda_graph_overhead_mb: int = 512,
    num_experts: int = 0,  # For MoE models
) -> MemoryBreakdown:
    """Estimate memory usage for given configuration."""
```

### 3. Configuration Search (`grid_search.py`)
```python
def find_valid_configs(
    model_config: PretrainedConfig,
    hardware: HardwareSpec,
    workload: dict,  # {"sequence_length": 2048, "batch_size": 32}
    max_configs: int = 10
) -> List[Tuple[ParallelismConfig, MemoryBreakdown, float]]:
    """Generate and rank valid parallelism configurations."""
```

### 4. Public API (`public_api.py`)
```python
def analyze(
    model: str,  # Hugging Face model ID
    cluster: dict,  # {"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 2}
    sequence_length: int = 2048,
    batch_size: int = 32
) -> List[dict]:
    """Analyze model and return ranked parallelism configurations."""

def best_config(
    model: str,
    cluster: dict,
    objective: str = "minimize_gpus"  # or "maximize_batch"
) -> dict:
    """Return the single best configuration for given objective."""
```

## Simplification Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Composition Over Inheritance**: Use functions and dataclasses instead of complex class hierarchies
3. **Explicit Dependencies**: Clear input/output types, minimal state
4. **Progressive Disclosure**: Simple API with optional complexity
5. **Focus on Core Use Case**: LLM inference optimization only

## Data Flow

```
model_id → ModelProfile → valid_constraints → memory_estimation → ranked_configs → best_config
```

1. Load model config from Hugging Face
2. Determine architectural constraints (TP/PP/EP limits)
3. Generate all valid parallelism combinations
4. Estimate memory for each configuration
5. Filter configurations that fit in GPU memory
6. Rank by simple heuristics (prefer fewer GPUs, then larger batch)
7. Return top configurations

## Implementation Size Target

- **constraints.py**: ~300 LOC
- **memory.py**: ~400 LOC  
- **grid_search.py**: ~250 LOC
- **public_api.py**: ~200 LOC
- **Total**: ~1200 LOC (vs. current ~4000+ LOC)

This simplified architecture removes:
- Complex inheritance hierarchies
- Duplicate memory calculations
- Multiple API layers
- Unvalidated performance modeling
- Framework-specific optimization (until proven necessary)
- Excessive parameterization
