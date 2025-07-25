# AutoParallel Simplified Public API

## Design Principles

1. **Single Entry Point**: One main function for the common use case
2. **Progressive Disclosure**: Advanced options available but not required
3. **Consistent Types**: Use standard Python types and dataclasses
4. **Clear Errors**: Helpful error messages for common issues

## Core API

### Main Function

```python
def analyze(
    model: str,
    cluster: dict,
    sequence_length: int = 2048,
    batch_size: int = 32,
    max_configs: int = 5
) -> List[dict]:
    """
    Analyze model parallelism options for given hardware.
    
    Args:
        model: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
        cluster: Hardware specification
            {
                "gpu_memory_gb": 80,      # Memory per GPU
                "gpus_per_node": 8,       # GPUs per node
                "num_nodes": 1,           # Number of nodes
                "gpu_architecture": "A100" # Optional: A100, H100, V100
            }
        sequence_length: Max sequence length for memory estimation
        batch_size: Batch size for memory estimation
        max_configs: Maximum configurations to return
        
    Returns:
        List of configuration dictionaries, ranked best to worst:
        [
            {
                "tensor_parallel": 2,
                "pipeline_parallel": 1,  
                "expert_parallel": 1,
                "data_parallel": 4,
                "total_gpus": 8,
                "memory_per_gpu_gb": 45.2,
                "memory_utilization": 0.87,
                "memory_breakdown": {
                    "weights": 12.5,
                    "activations": 15.3,
                    "kv_cache": 12.8,
                    "framework_overhead": 4.6,
                    "total": 45.2
                },
                "deployment_command": "python -m vllm.entrypoints.openai..."
            }
        ]
        
    Raises:
        ModelNotFoundError: If model doesn't exist on Hugging Face
        InsufficientMemoryError: If model can't fit on available hardware
        InvalidConfigurationError: If cluster specification is invalid
    """
```

### Convenience Functions

```python
def best_config(model: str, cluster: dict, objective: str = "minimize_gpus") -> dict:
    """
    Get the single best configuration for given objective.
    
    Args:
        objective: "minimize_gpus" | "maximize_batch" | "balanced"
        
    Returns:
        Single configuration dictionary (same format as analyze())
    """

def check_memory_requirements(model: str, sequence_length: int = 2048) -> dict:
    """
    Check memory requirements without hardware constraints.
    
    Returns:
        {
            "model_size_gb": 13.2,
            "minimum_memory_gb": 25.8,  # With activations + overhead
            "recommended_memory_gb": 32.0,  # With safety margin
            "supports_quantization": True
        }
    """

def estimate_cost(
    model: str, 
    cluster: dict, 
    hours_per_month: int = 730
) -> dict:
    """
    Estimate monthly cloud costs (rough approximation).
    
    Returns:
        {
            "gpus_needed": 8,
            "estimated_monthly_cost_usd": 12000,  # Very rough estimate
            "cost_per_gpu_hour": 2.50,
            "assumptions": "Based on AWS p4d.24xlarge pricing"
        }
    """
```

## Error Handling

```python
class AutoParallelError(Exception):
    """Base exception for AutoParallel."""

class ModelNotFoundError(AutoParallelError):
    """Model not found on Hugging Face Hub."""

class InsufficientMemoryError(AutoParallelError):
    """Model cannot fit on available hardware."""
    
class InvalidConfigurationError(AutoParallelError):
    """Invalid cluster or model configuration."""
```

## Usage Examples

### Basic Usage
```python
import autoparallel

# Analyze Llama-2-7B on single A100 node
configs = autoparallel.analyze(
    model="meta-llama/Llama-2-7b-hf",
    cluster={
        "gpu_memory_gb": 80,
        "gpus_per_node": 8,
        "num_nodes": 1
    }
)

# Get the best configuration
best = configs[0]
print(f"Use {best['tensor_parallel']}x TP, {best['data_parallel']}x DP")
print(f"Memory usage: {best['memory_utilization']:.1%}")
```

### Objective-Based Selection
```python
# Minimize GPU count
minimal = autoparallel.best_config(
    model="meta-llama/Llama-2-70b-hf",
    cluster={"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 4},
    objective="minimize_gpus"
)

# Maximize batch size  
throughput = autoparallel.best_config(
    model="meta-llama/Llama-2-70b-hf", 
    cluster={"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 4},
    objective="maximize_batch"
)
```

### Memory Check
```python
# Check requirements before buying hardware
requirements = autoparallel.check_memory_requirements(
    model="meta-llama/Llama-2-70b-hf",
    sequence_length=4096
)

print(f"Model needs at least {requirements['minimum_memory_gb']} GB")
print(f"Recommend {requirements['recommended_memory_gb']} GB per GPU")
```

### Cost Estimation
```python
cost_info = autoparallel.estimate_cost(
    model="meta-llama/Llama-2-70b-hf",
    cluster={"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 2}
)

print(f"Estimated cost: ${cost_info['estimated_monthly_cost_usd']}/month")
```

## Deployment Integration

The deployment command in the response can be used directly:

```python
config = autoparallel.best_config(model="...", cluster={...})
command = config["deployment_command"]

# Run with subprocess or save to script
import subprocess
subprocess.run(command, shell=True)
```

## Migration from Complex API

For users migrating from more complex APIs:

```python
# Old complex way:
# cluster = Cluster(gpu_memory=80, ...)
# workload = Workload(batch_size=32, ...)
# optimizer = AutoParallel(cluster, workload)
# result = optimizer.optimize(model)

# New simple way:
configs = autoparallel.analyze(model, {"gpu_memory_gb": 80, ...})
best = configs[0]
```

## Future Extensions

Advanced features can be added without breaking the simple API:

```python
# Future: Framework-specific optimization
def optimize_vllm(model: str, cluster: dict, **vllm_args) -> dict:
    """vLLM-specific optimization."""

# Future: Training optimization  
def optimize_training(model: str, cluster: dict, **training_args) -> dict:
    """Training-specific parallelism optimization."""
```

The simplified API prioritizes ease of use for the 95% common case while allowing future extensions for advanced use cases.
