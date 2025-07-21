# Testing and Validation Strategy

## Overview

This document defines a comprehensive testing strategy for AutoParallel that maximizes verification coverage while minimizing computational overhead. The strategy prioritizes fast, lightweight tests that can run frequently during development, with selective integration testing for validation without requiring expensive GPU clusters or large model instantiation.

## Testing Philosophy

### Core Principles

1. **Meta-Device First**: Leverage Hugging Face's `device_map="meta"` functionality to test model analysis without memory allocation
2. **Architecture-Driven Testing**: Test parallelism constraints using model architectures rather than loaded weights
3. **Synthetic Validation**: Use mathematical models and known constraints to validate calculations
4. **Selective Integration**: Reserve expensive tests (full model loading, vLLM engines) for critical paths and CI/deployment validation
5. **Model Diversity**: Cover architectural variations (dense, MoE, GQA, quantized) across representative model families

## Test Categories

### 1. Unit Tests (Fast, < 1s each)

#### Model Analysis Tests
- **Constraint Detection**: Test attention head divisibility, expert distribution, vocabulary sharding constraints
- **Architecture Parsing**: Validate transformers config parsing for all supported model types
- **Memory Calculations**: Test component-based memory estimation (weights, activations, KV cache, optimizer states)
- **Quantization Handling**: Test memory overhead calculations for GPTQ, AWQ, bitsandbytes, int4/int8/fp16/fp8

#### Configuration Generation Tests
- **Parallelism Enumeration**: Test valid TP/PP/EP/DP combination generation
- **Constraint Validation**: Test invalid configuration rejection (memory, architecture, topology constraints)
- **Parameter Bounds**: Test `MemoryEstimationParameters` and `ParallelismConstraintParameters` validation
- **Edge Cases**: Test single GPU, maximum parallelism, impossible configurations

#### Framework Integration Tests
- **vLLM Configuration**: Test deployment command generation and parameter mapping
- **DeepSpeed Integration**: Test ZeRO optimization configuration generation
- **CUDA Graph Estimation**: Test memory allocation calculations for different graph sizes

### 2. Integration Tests (Medium, 1-30s each)

#### Meta-Device Model Loading
Test model analysis pipeline using meta devices for primary target models:

**Priority Models (Native Precision, Max Context)**:
- `Qwen/Qwen3-14B` (bf16, 32K context, GQA + SwiGLU)
- `Qwen/Qwen3-32B` (bf16, 32K context, GQA + SwiGLU)
- `Qwen/Qwen3-30B-A3B` (bf16, 32K context, MoE 128 experts, 8 active)
- `Qwen/Qwen3-235B-A22B` (bf16, 32K context, MoE 128 experts, 8 active)
- `deepseek-ai/DeepSeek-V3-Base` (finegrained blockwise fp8, 128K context, MoE 256 experts)
- `moonshotai/Kimi-K2-Instruct` (block-fp8, 128K context, MoE 384 experts)
- `meta-llama/Llama-3.1-70B` (bf16, 128K context, GQA)
- `meta-llama/Llama-4-Scout-17B-16E` (bf16, 10M context, multimodal MoE 16 experts)
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct` (bf16/fp8, 1M context, multimodal MoE 128 experts)

**Smaller Validation Models**:
- `meta-llama/Llama-3.2-1B` (bf16, 8K context, GQA)
- `meta-llama/Llama-3.2-3B` (bf16, 128K context, GQA)

#### Memory Estimation Validation
- **Synthetic Workloads**: Test memory calculations against mathematical models
- **Framework Comparison**: Compare memory estimates between vLLM and DeepSpeed configurations
- **Quantization Impact**: Test memory reduction calculations for different quantization schemes
- **Safety Margin Validation**: Test configurable memory safety margins and fragmentation overhead

#### Cluster Analysis Tests
- **Topology Modeling**: Test intra-node vs inter-node communication efficiency calculations
- **GPU Configuration**: Test memory and compute capacity detection for different GPU types
- **Network Constraints**: Test bandwidth-aware parallelism strategy selection

### 3. System Tests (Slow, 30s-5min each)

#### Configuration Feasibility Validation
- **End-to-End Pipeline**: Test complete analysis from model specification to deployment commands
- **Multi-Model Comparison**: Test comparative analysis across different model architectures
- **Workload Optimization**: Test throughput vs latency optimization for different use cases
- **Error Handling**: Test graceful failure modes and error reporting

#### Performance Model Validation
- **Synthetic Benchmarking**: Test throughput/latency predictions against mathematical models
- **Configuration Ranking**: Test optimization target achievement (cost, latency, throughput)
- **Scaling Behavior**: Test prediction accuracy across different cluster sizes

### 4. Acceptance Tests (Expensive, 5min+ each)

#### Limited Model Loading Tests
Run sparingly, only for critical validation:

**Small Model Loading** (CI/deployment only):
- Load actual `Llama-3.2-1B` (bf16, 8K) to validate memory estimation accuracy
- Compare predicted vs actual memory usage for known configurations
- Test vLLM engine instantiation with generated configurations

**Production Validation** (Manual/staging only):
- Test deployment commands on actual GPU clusters with priority models
- Validate memory predictions against real framework memory usage (native precision, max context)
- Performance benchmarking for configuration optimization validation

## Model Coverage Strategy

### Primary Target Models (Native Precision, Max Sequence Length)

| Model Family | Primary Test Model | Native Precision | Max Sequence Length | Architecture |
|--------------|-------------------|------------------|---------------------|--------------|
| Qwen (Dense) | Qwen/Qwen3-14B, Qwen/Qwen3-32B | bf16 | 32K | GQA, SwiGLU |
| Qwen (MoE) | Qwen/Qwen3-30B-A3B, Qwen/Qwen3-235B-A22B | bf16 | 32K | GQA, MoE (128 experts, 8 active) |
| DeepSeek | deepseek-ai/DeepSeek-V3-Base | finegrained blockwise fp8 | 128K | MoE (256 experts, 8 active) + Dense |
| Kimi | moonshotai/Kimi-K2-Instruct | block-fp8 | 128K | MoE (384 experts, 8 active) |
| Llama3 | meta-llama/Llama-3.1-70B | bf16 | 128K | GQA, RMSNorm |
| Llama3 | meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B | bf16 | 8K, 128K | GQA, RMSNorm |
| Llama4 (MoE) | meta-llama/Llama-4-Scout-17B-16E | bf16 | 10M | Multimodal MoE (16 experts, 1 active) |
| Llama4 (MoE) | meta-llama/Llama-4-Maverick-17B-128E-Instruct | bf16/fp8 | 1M | Multimodal MoE (128 experts, ~17B active) |

### Architecture Coverage

- **Attention Mechanisms**: Multi-head (Llama), Grouped Query (Qwen3), Multi-Query (MQA)
- **MLP Variations**: Standard (Llama), SwiGLU (Qwen3, DeepSeek-V3, Kimi-K2)
- **MoE Configurations**: 
  - Qwen3 MoE (128 experts, 8 active)
  - DeepSeek-V3 (256 experts, 8 active) 
  - Kimi-K2 (384 experts, 8 active)
  - Llama4 Scout (16 experts, 1 active)
  - Llama4 Maverick (128 experts, alternating MoE/dense layers)
- **Normalization**: RMSNorm across all target architectures
- **Precision Focus**: bf16 (Llama, Qwen), finegrained blockwise fp8 (DeepSeek-V3), block-fp8 (Kimi-K2), bf16/fp8 (Llama4)
- **Context Lengths**: 8K-10M tokens, including extreme long context models
- **Multimodal**: Llama4 Scout/Maverick with early fusion text+image processing

## Test Infrastructure

### Test Organization
```
tests/
├── unit/
│   ├── model_analysis/
│   │   ├── constraint_detection_test.py
│   │   ├── memory_calculation_test.py
│   │   └── quantization_handling_test.py
│   ├── configuration/
│   │   ├── generation_test.py
│   │   ├── validation_test.py
│   │   └── parameter_bounds_test.py
│   └── framework/
│       ├── vllm_integration_test.py
│       └── deepspeed_integration_test.py
├── integration/
│   ├── meta_device_loading_test.py
│   ├── memory_estimation_validation_test.py
│   └── cluster_analysis_test.py
├── system/
│   ├── end_to_end_pipeline_test.py
│   └── performance_model_test.py
└── acceptance/
    ├── model_loading_validation_test.py
    └── production_deployment_test.py
```

### Test Data Management

**Model Fixtures**: Maintain minimal test fixtures for architecture configurations
**Synthetic Data**: Generate test cases programmatically for parameter sweeps
**Reference Results**: Store known-good results for regression testing
**Performance Baselines**: Track test execution times and memory usage

### Continuous Integration

**Fast Tests** (< 2 minutes total):
- All unit tests
- Meta-device integration tests
- Configuration generation and validation

**Nightly Tests** (< 30 minutes total):
- All fast tests
- System tests
- Architecture stress testing
- Memory estimation validation

**Release Tests** (< 2 hours total):
- All tests
- Limited acceptance tests with small model loading
- Performance regression testing

## Test Implementation Guidelines

### Mock Strategies

**GPU Environment**: Mock CUDA availability and GPU specifications for cluster analysis
**Model Downloads**: Mock Hugging Face model downloads to use cached/local configurations
**Framework Integration**: Mock vLLM and DeepSpeed APIs for configuration testing
**Network Topology**: Mock cluster network configurations for parallelism strategy testing

### Performance Criteria

**Unit Tests**: Each test < 1 second, total suite < 30 seconds
**Integration Tests**: Each test < 30 seconds, total suite < 5 minutes
**System Tests**: Each test < 5 minutes, total suite < 30 minutes
**Memory Usage**: Tests should not exceed 4GB RAM (except acceptance tests)

### Coverage Targets

**Code Coverage**: Minimum 90% line coverage, 85% branch coverage
**Architecture Coverage**: Test all supported model architectures and features
**Configuration Coverage**: Test all valid parallelism strategy combinations
**Error Coverage**: Test all documented error conditions and edge cases

## Validation Methodology

### Memory Estimation Accuracy

**Synthetic Validation**: Compare calculations against mathematical models for known cases
**Relative Validation**: Test scaling behavior across different model sizes and configurations
**Framework Consistency**: Ensure consistent estimates between vLLM and DeepSpeed modes
**Regression Testing**: Track estimation accuracy over time with reference implementations

### Performance Model Validation

**Mathematical Models**: Validate throughput/latency calculations against analytical models
**Scaling Laws**: Test prediction accuracy across different parallelism degrees
**Communication Modeling**: Validate network overhead predictions for different topologies
**Optimization Targets**: Test achievement of specified optimization goals (throughput, latency, cost)

### Configuration Correctness

**Constraint Validation**: Test all architecture constraints are properly enforced
**Feasibility Checking**: Ensure generated configurations are actually valid
**Completeness**: Verify all valid configurations are discovered
**Optimality**: Test that optimal configurations are identified for given targets

## Implementation Priority

### Core Testing Focus
1. Unit tests for memory calculation and constraint detection with verified model architectures
2. Meta-device loading for validated models: Qwen3 Dense/MoE (32K), DeepSeek-V3 (128K), Kimi-K2 (128K), Llama-3.1 (128K), Llama4 Scout (10M), Llama4 Maverick (1M)
3. Configuration generation for diverse MoE architectures:
   - High expert count: Kimi-K2 (384), DeepSeek-V3 (256), Qwen3 MoE (128), Llama4 Maverick (128)
   - Low expert count: Llama4 Scout (16, 1 active)
   - Hybrid architectures: Llama4 Maverick alternating MoE/dense layers
4. Framework integration testing for multiple precision formats: bf16, finegrained blockwise fp8, block-fp8, multimodal fp8
5. Extreme context length testing: Scout (10M tokens), Maverick (1M tokens) for memory scaling validation
6. Multimodal parallelism testing: Early fusion text+image processing for Llama4 models
7. Limited acceptance testing with Llama-3.2-1B (8K context) for memory accuracy validation

This strategy covers the full spectrum of modern LLM architectures: dense transformers, diverse MoE configurations (16-384 experts), multiple precision formats, extreme context lengths, and multimodal processing - providing comprehensive stress testing for parallelism optimization.
