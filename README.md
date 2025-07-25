# AutoParallel

Automatic parallelization strategy optimization for large language models (LLMs). AutoParallel analyzes your model and hardware to recommend the best tensor parallel, pipeline parallel, and data parallel configurations for optimal GPU utilization and performance.

## Overview

AutoParallel simplifies deploying large language models by automatically determining the optimal parallelization strategy for your specific hardware setup. It supports a wide range of transformer architectures and provides memory-accurate estimates to help you make informed deployment decisions.

**Key Features:**
- 🚀 **Simple API** - Single function call for most use cases
- 🧠 **Smart Analysis** - Considers model architecture, hardware constraints, and memory requirements
- 📊 **Memory Estimation** - Accurate GPU memory usage predictions
- ⚡ **Multi-Strategy Support** - Tensor, pipeline, expert, and data parallelism
- 🔧 **Production Ready** - Generate deployment commands for popular frameworks

## Installation

AutoParallel uses the modern Python toolchain with [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install AutoParallel
git clone https://github.com/jvmncs/autoparallel.git
cd autoparallel
uv sync --group dev
```

**System Requirements:**
- Python 3.12+
- Internet connection (for downloading model configurations)

## Quick Start

### Basic Usage

```python
import autoparallel

# Analyze Llama-2-7B on an 8xA100 node
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
print(f"Recommended: {best['tensor_parallel']}x TP, {best['data_parallel']}x DP")
print(f"Memory utilization: {best['memory_utilization']:.1%}")
print(f"GPUs needed: {best['total_gpus']}")
```

### Check Memory Requirements

```python
# Check memory needs before purchasing hardware
requirements = autoparallel.check_memory_requirements(
    model="meta-llama/Llama-2-70b-hf",
    sequence_length=4096
)

print(f"Model size: {requirements['total_memory_gb']:.1f} GB")
print(f"Minimum GPU memory: {requirements['single_gpu_requirements']:.1f} GB")
print(f"Recommended: {requirements['recommended_gpu_memory_gb']:.1f} GB per GPU")
```

### Get Single Best Configuration

```python
# Get optimal config for specific objective
config = autoparallel.best_config(
    model="meta-llama/Llama-2-70b-hf",
    cluster={
        "gpu_memory_gb": 80,
        "gpus_per_node": 8, 
        "num_nodes": 4
    },
    objective="minimize_gpus"  # or "maximize_throughput", "balance"
)

print(f"Deploy with: {config['deployment_command']}")
```

## API Reference

### Core Functions

#### `analyze(model, cluster, **kwargs)`

Main analysis function returning ranked parallelism configurations.

**Parameters:**
- `model` (str): Hugging Face model identifier (e.g., "meta-llama/Llama-2-7b-hf")
- `cluster` (dict): Hardware specification
  - `gpu_memory_gb` (int): Memory per GPU in GB
  - `gpus_per_node` (int): GPUs per node
  - `num_nodes` (int): Number of nodes
  - `gpu_architecture` (str, optional): GPU type ("A100", "H100", "V100")
- `sequence_length` (int, default=2048): Maximum sequence length
- `batch_size` (int, default=1): Batch size per GPU
- `quantization` (str, default="fp16"): Quantization format
- `max_configs` (int, default=10): Maximum configurations to return

**Returns:**
List of configuration dictionaries ranked best to worst:
```python
[
    {
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "expert_parallel": 1,
        "data_parallel": 4,
        "total_gpus": 8,
        "memory_per_gpu_gb": 45.2,
        "memory_utilization": 0.87,
        "memory_breakdown": {...},
        "deployment_command": "python -m vllm.entrypoints.openai..."
    }
]
```

#### `best_config(model, cluster, objective="minimize_gpus", **kwargs)`

Get single best configuration for given objective.

**Objectives:**
- `"minimize_gpus"`: Use fewest GPUs possible
- `"maximize_throughput"`: Optimize for throughput  
- `"balance"`: Balanced resource usage

#### `check_memory_requirements(model, **kwargs)`

Check memory requirements without hardware constraints.

**Returns:**
```python
{
    "total_memory_gb": 13.2,
    "single_gpu_requirements": 25.8,
    "recommended_gpu_memory_gb": 32.0,
    "breakdown": {...},
    "architecture_info": {...}
}
```

### Exception Handling

```python
from autoparallel import ModelNotFoundError, InsufficientMemoryError

try:
    configs = autoparallel.analyze(model, cluster)
except ModelNotFoundError:
    print("Model not found on Hugging Face Hub")
except InsufficientMemoryError:
    print("Model too large for available hardware")
```

## Examples

See the [`examples/`](examples/) directory for detailed usage examples:

- **[vLLM Memory Estimation](examples/vllm_memory_estimation.py)** - Memory analysis and optimization
- **[vLLM Config Optimization](examples/vllm_config_optimization.py)** - Configuration tuning for different workloads

Run examples:
```bash
uv run python examples/vllm_memory_estimation.py
uv run python examples/vllm_config_optimization.py
```

## Supported Models

AutoParallel supports a wide range of transformer architectures through Hugging Face's model hub:

### Model Families

| Family | Models | Precision | Features |
|--------|--------|-----------|----------|
| **Llama** | Llama-3.1/3.2/3.3 (1B-70B), Llama-4-Scout/Maverick | bf16 | Long context (128K-10M), MoE |
| **Qwen** | Qwen3 (8B-32B), Qwen3-235B/480B-A22B/A35B | bf16 | Dense + MoE variants |
| **DeepSeek** | DeepSeek-V3, DeepSeek-R1 | fp8 | 256 experts, 8 active |
| **Kimi** | Kimi-K2-Instruct | fp8 | 384 experts, 8 active |
| **Mixtral** | Mixtral-8x7B, Mixtral-8x22B | fp16/bf16 | MoE architectures |

### Architecture Support

- **Dense Transformers**: Standard attention-based models
- **Mixture of Experts (MoE)**: Expert parallelism optimization
- **Grouped Query Attention (GQA)**: Efficient attention patterns
- **Long Context**: Models with up to 10M token context length

### Quantization Formats

- **fp32** (4 bytes/param) - Full precision
- **fp16/bf16** (2 bytes/param) - Half precision (most common)
- **int8** (1 byte/param) - 8-bit quantization
- **fp8** (1 byte/param) - Specialized for newer models

## Memory Estimation

AutoParallel provides detailed memory breakdowns to help optimize deployments:

### Memory Components

1. **Model Weights**: Parameter storage based on quantization
2. **Activations**: Forward pass intermediate values
3. **KV Cache**: Key-value cache for attention mechanism
4. **Framework Overhead**: Runtime memory overhead (typically 10-15%)

### Memory Optimization Tips

- **Use fp16/bf16**: Reduces model size by 50% vs fp32
- **Consider fp8**: Further reduction for supported models
- **Optimize sequence length**: Longer sequences require more KV cache
- **Balance batch size**: Higher batch utilization vs memory usage

## Troubleshooting

### Common Issues

**Model Not Found**
```python
# Ensure model name is correct and publicly available
configs = autoparallel.analyze("meta-llama/Llama-2-7b-hf", cluster)
```

**Insufficient Memory**
```python
# Check memory requirements first
requirements = autoparallel.check_memory_requirements(model)
print(f"Need at least {requirements['recommended_gpu_memory_gb']} GB per GPU")
```

**No Valid Configurations**
- Increase GPU memory or number of GPUs
- Try different quantization (fp16 → int8)
- Reduce sequence length or batch size

**Import Errors**
```bash
# Ensure dependencies are installed
uv sync --group dev
```

### Getting Help

1. Check the [examples/](examples/) directory
2. Review error messages - they include specific recommendations
3. Use `check_memory_requirements()` to understand constraints
4. Open an issue with your configuration and error details

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/jvmncs/autoparallel.git
cd autoparallel

# Install development dependencies
uv sync --group dev

# Activate environment
. .venv/bin/activate
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check --fix

# Type checking
uv run ty check
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest src/autoparallel/public_api_test.py
```

### Project Structure

```
autoparallel/
├── src/autoparallel/          # Main source code
│   ├── public_api.py          # Simplified public API
│   ├── memory.py              # Memory estimation
│   ├── constraints.py         # Model architecture analysis
│   ├── grid_search.py         # Configuration optimization
│   └── frameworks/            # Framework-specific optimizations
├── examples/                  # Usage examples
├── spec/                      # Project specifications
└── tests/                     # Test suite
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite
5. Submit a pull request

### Development Workflow

```bash
# Create new branch
jj new main

# Make changes
# ... edit files ...

# Test changes
uv run pytest
uv run ruff check

# Commit changes
jj describe -m "Add new feature"

# Push to remote
jj git push --no-pager
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use AutoParallel in your research, please cite:

```bibtex
@software{autoparallel2024,
  title={AutoParallel: Automatic LLM Parallelization Strategy Optimization},
  author={Mancuso, Jason},
  year={2024},
  url={https://github.com/jvmncs/autoparallel}
}
```
