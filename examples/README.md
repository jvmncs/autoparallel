# AutoParallel Examples

This directory contains example scripts demonstrating how to use AutoParallel for various parallelization and memory analysis tasks.

## Memory Requirement Analysis

### `memory_check.py`

A comprehensive example showing how to analyze memory requirements for large language models across different configurations.

**Features demonstrated:**

1. **Basic memory requirement checking** - Check memory needs for popular models like Llama 2 7B
2. **Precision format comparison** - Compare memory usage across fp32, fp16, bf16, int8, and fp8
3. **Sequence length impact** - See how sequence length affects KV cache and activation memory
4. **Minimum GPU requirements** - Find the smallest GPU that can fit different model sizes
5. **Parallelism scaling** - Understand how tensor/pipeline parallelism reduces per-GPU memory
6. **Error handling** - Handle scenarios where models don't fit in available memory
7. **MoE model analysis** - Compare Mixture of Experts vs dense models

**Usage:**

```bash
# Run the complete analysis
uv run python examples/memory_check.py

# Or using your virtual environment
python examples/memory_check.py
```

**Sample output:**

```
AutoParallel Memory Requirement Analysis Examples
============================================================
Example 1: Basic Memory Requirements
============================================================
Model: Llama 2 7B
Total Memory Required: 12.9GB

Memory Breakdown:
  Weights:    9.9GB
  Activations:0.1GB
  KV Cache:   1.0GB
  Framework:  2.0GB
  Total:      12.9GB

GPU Compatibility:
  ✓ A100_40GB
  ✓ A100_80GB
  ✓ H100_80GB
  ✓ RTX_4090_24GB
  ✓ RTX_3090_24GB
```

**Key functions used:**

- `autoparallel.check_memory_requirements()` - High-level memory analysis
- `autoparallel.memory.estimate_memory()` - Detailed memory estimation
- `autoparallel.memory.check_memory_feasibility()` - Check if model fits in GPU
- `autoparallel.memory.get_quantization_bytes()` - Convert precision to bytes per parameter

**Note about the MockConfig class:**

Due to a bug in the current memory estimation code (where `hasattr(model_config, "__getattribute__")` is used to detect config objects vs dictionaries, but dictionaries also have `__getattribute__`), the example includes a `MockConfig` class that works around this issue. In production code, you should use proper Hugging Face `PretrainedConfig` objects.

## Future Examples

This directory will be expanded with additional examples covering:

- Grid search optimization for different cluster configurations
- Cost estimation and deployment planning
- Advanced parallelism strategies for specific model architectures
- Real-world deployment scenarios and best practices

## Requirements

Examples require the core autoparallel package and its dependencies:

```bash
uv sync --group dev
```

Some examples may attempt to load real model configurations from Hugging Face, but will fall back to mock configurations if models are not accessible.
