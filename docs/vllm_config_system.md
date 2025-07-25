# vLLM Configuration System

The vLLM configuration system provides automated optimization of vLLM memory configurations without requiring full engine instantiation. It addresses the critical tradeoff between KV cache allocation and CUDA graph capture sizes to maximize throughput while respecting memory constraints.

## Overview

The system implements the key insight from vLLM research: **KV cache gets the leftover memory after all other components are allocated**. This priority-based allocation follows:

1. **Model weights** - Fixed requirement from transformers config
2. **Activation memory** - Measured via profiling with meta tensors  
3. **CUDA graph memory** - Calculated from capture sizes and model architecture
4. **KV cache memory** - Remaining memory allocated to block pool

## Core Components

### 1. AutotuningParameters

Configurable parameters for fine-tuning the optimization behavior:

```python
from autoparallel.frameworks.vllm_config import AutotuningParameters

params = AutotuningParameters(
    # CUDA graph memory estimation
    graph_memory_overhead_base_ratio=0.1,
    graph_memory_batch_scaling_factor=0.02,
    
    # Memory utilization bounds
    min_gpu_memory_utilization=0.8,
    max_gpu_memory_utilization=0.98,
    
    # Performance scoring weights
    throughput_batch_weight=0.7,
    throughput_graph_weight=0.3,
    latency_graph_weight=0.8,
    latency_batch_weight=0.2,
)
```

### 2. vLLMPerformanceModel

Models vLLM performance without engine instantiation:

```python
from autoparallel.frameworks.vllm_config import vLLMPerformanceModel
import transformers

config = transformers.AutoConfig.from_pretrained("llama-7b")
model = vLLMPerformanceModel.from_transformers_config(
    config=config,
    gpu_memory_capacity_gb=40.0,
    gpu_memory_utilization=0.9,
    max_model_len=2048,
)

# Calculate memory breakdown
breakdown = model.calculate_memory_breakdown()
print(f"KV Cache: {breakdown['kv_cache_memory']:.2f}GB")
print(f"CUDA Graphs: {breakdown['cuda_graph_memory']:.2f}GB")

# Calculate effective batch size
batch_size = model.calculate_effective_batch_size()
print(f"Max concurrent sequences: {batch_size}")
```

### 3. WorkloadProfile

Characterizes expected vLLM workload patterns:

```python
from autoparallel.frameworks.vllm_memory import WorkloadProfile

# Create synthetic workload profiles
chatbot = WorkloadProfile.create_synthetic("chatbot")
batch = WorkloadProfile.create_synthetic("batch_inference") 
interactive = WorkloadProfile.create_synthetic("interactive")

# Custom workload
custom = WorkloadProfile(
    requests_per_second=100,
    batch_size_distribution={1: 0.3, 2: 0.4, 4: 0.2, 8: 0.1},
    sequence_length_distribution={512: 0.4, 1024: 0.4, 2048: 0.2},
    target_metric="latency",
    latency_budget_ms=50,
)
```

### 4. vLLMConfigOptimizer

Searches for optimal vLLM configurations:

```python
from autoparallel.frameworks.vllm_config import vLLMConfigOptimizer

optimizer = vLLMConfigOptimizer(
    model_name="meta-llama/Llama-2-7b-hf",
    gpu_memory_capacity_gb=40.0,
    tuning_params=params,
)

result = optimizer.search_optimal_config(workload=chatbot)

if result["optimal_config"]:
    config = result["optimal_config"]
    predictions = result["predictions"]
    
    print(f"Optimal memory utilization: {config.gpu_memory_utilization}")
    print(f"Effective batch size: {predictions['effective_batch_size']}")
    print(f"Graph coverage: {predictions['graph_coverage']:.1%}")
```

## Cluster Integration

The system integrates with autoparallel for cluster-wide optimization:

```python
from autoparallel.frameworks.vllm_config import optimize_vllm_config_for_cluster

parallelism_strategy = {"tp": 2, "pp": 1, "dp": 4}

result = optimize_vllm_config_for_cluster(
    model_name="meta-llama/Llama-2-7b-hf",
    gpu_memory_capacity_gb=40.0,
    workload=workload,
    parallelism_strategy=parallelism_strategy,
)

print(f"Total cluster throughput: {result['cluster_predictions']['total_throughput']}")
print(f"Deployment recommendations: {result['recommendations']}")
```

## Configuration Validation

Validate configurations before deployment:

```python
validation = optimizer.validate_configuration(config)

print(f"Valid: {validation['valid']}")
print(f"Warnings: {validation['warnings']}")
print(f"Recommendations: {validation['recommendations']}")
```

## Memory Model

The system uses empirically-derived constants for accurate memory estimation:

- **Graph memory scaling (0.1, 0.02)**: Based on CUDA graph capture profiling
- **Compilation multipliers (1.8, 1.0)**: Memory overhead for FULL vs PIECEWISE compilation
- **Activation memory factor (0.3)**: Conservative estimate from inference profiling
- **Performance weights (0.7/0.3, 0.8/0.2)**: Balance between batch size and graph coverage

## Key Benefits

1. **No Engine Instantiation**: Optimize configurations without building full vLLM engines
2. **Memory-Aware**: Explicitly model KV cache vs CUDA graph tradeoffs
3. **Workload-Specific**: Tailor configurations to actual usage patterns  
4. **Cluster-Integrated**: Combine with autoparallel's parallelism optimization
5. **Actionable**: Generate specific deployment recommendations

## Example Usage

See `examples/vllm_config_optimization.py` for a complete demonstration of:

- Single-instance optimization for different workloads
- Cluster-wide optimization with different parallelism strategies
- Configuration validation and recommendations
- Workload type comparison

## Testing

The implementation includes comprehensive tests covering:

- Memory calculation functions
- Performance model creation and predictions
- Configuration optimization and validation
- Cluster integration
- Edge cases and error handling

Run tests with:
```bash
uv run pytest src/autoparallel/frameworks/vllm_config_test.py -v
```
