# AutoParallel Library - Comprehensive Implementation Plan

## Executive Summary

AutoParallel is a transformers-native parallelism optimization library that automatically determines optimal GPU cluster configurations for LLM inference and training. This implementation plan details the step-by-step approach to build a production-ready library that eliminates parallelism guesswork through data-driven, architecture-aware recommendations.

## Project Foundation Analysis

### Current State
- **Minimal codebase**: Single placeholder function in `src/autoparallel/__init__.py`
- **Complete specifications**: 5 detailed architectural specifications in `spec/` directory
- **Development environment**: Fully configured Astral stack (uv, ruff, ty, pytest)
- **Ready for implementation**: Comprehensive API design and testing strategy documented

### Architecture Overview
The library consists of four core components:
1. **Memory Estimation Framework**: Unified memory calculations across frameworks
2. **Architecture Constraint Analyzer**: Model-specific parallelization limits
3. **Configuration Generator**: Optimal parallelism strategy enumeration
4. **Framework Integrations**: vLLM and DeepSpeed deployment optimization

## Implementation Phases

### Phase 1: Core Infrastructure

#### 1.1 Memory Estimation Framework
**Priority: Critical**
**Files to implement:**
- `src/autoparallel/memory/estimator.py`
- `src/autoparallel/memory/components.py`
- `src/autoparallel/memory/config.py`

**Key Components:**
```python
# Memory component breakdown
class MemoryComponents:
    weights: int
    activations: int  
    kv_cache: int
    cuda_graphs: int
    optimizer_states: int
    fragmentation_overhead: int

# Configurable estimation parameters
class MemoryConfig:
    utilization_bound: float = 0.85
    fragmentation_overhead: float = 0.10
    safety_margin: float = 0.05
    quantization_format: str = "fp16"
```

**Implementation Details:**
- Unified memory calculation supporting GPTQ, AWQ, bitsandbytes formats
- Component-wise breakdown (weights, activations, KV cache, CUDA graphs)
- Configurable safety margins and fragmentation overhead
- Meta-device analysis for zero-memory model introspection

**Testing Requirements:**
- Unit tests for each memory component calculation
- Validation against different quantization formats
- Edge case handling (extreme model sizes, memory constraints)

#### 1.2 Architecture Constraint Analyzer
**Priority: Critical**
**Files to implement:**
- `src/autoparallel/constraints/analyzer.py`
- `src/autoparallel/constraints/model_support.py`
- `src/autoparallel/constraints/validation.py`

**Key Features:**
```python
# Architecture-specific constraints
class ModelConstraints:
    max_tensor_parallel: int    # Based on attention heads
    max_pipeline_parallel: int  # Based on layer count
    max_expert_parallel: int    # For MoE models
    vocabulary_sharding: bool   # For large vocabularies
    
# Automatic constraint detection
def analyze_model_constraints(model_config) -> ModelConstraints:
    # Detect attention heads, layers, experts, vocab size
    # Calculate optimal parallelization limits
    # Return validated constraints
```

**Model Architecture Support:**
- **Dense Transformers**: Llama, Qwen, Mistral families with GQA support
- **MoE Models**: DeepSeek-V3 (256 experts), Kimi-K2 (384 experts), Qwen3 MoE
- **Multimodal**: Llama4 Scout/Maverick with text+image processing
- **Long Context**: 1M-10M token context length support

**Testing Strategy:**
- Meta-device loading for model config analysis
- Constraint validation across different architectures
- Edge case testing (single layer models, massive MoE)

#### 1.3 Configuration Generator
**Priority: High**
**Files to implement:**
- `src/autoparallel/config/generator.py`
- `src/autoparallel/config/optimizer.py`
- `src/autoparallel/config/validator.py`

**Core Algorithm:**
```python
def generate_valid_configs(
    constraints: ModelConstraints,
    cluster: ClusterConfig,
    workload: WorkloadProfile
) -> List[ParallelismConfig]:
    # Enumerate all valid TP/PP/EP/DP combinations
    # Filter by memory constraints
    # Score by workload-specific metrics
    # Return ranked configurations
```

**Optimization Targets:**
- **Throughput**: Maximize tokens/second for batch inference
- **Latency**: Minimize time-to-first-token for interactive chat
- **Memory Efficiency**: Maximize model size fit within cluster
- **Cost**: Optimize for cloud deployment economics

#### 1.4 Testing Infrastructure
**Priority: High**
**Files to implement:**
- `tests/test_memory_estimation.py`
- `tests/test_constraint_analysis.py`
- `tests/test_config_generation.py`
- `tests/conftest.py` (pytest fixtures)

**Testing Philosophy:**
- **Meta-Device First**: Use HuggingFace meta devices for memory-free testing
- **Architecture-Driven**: Test constraints using model configs, not loaded weights
- **Selective Integration**: Reserve expensive tests for critical validation

**Performance Targets:**
- Unit tests: <1 second each
- Integration tests: <30 seconds
- System tests: <5 minutes
- 90% line coverage, 85% branch coverage

### Phase 2: Framework Integration

#### 2.1 vLLM Integration
**Priority: High**
**Files to implement:**
- `src/autoparallel/frameworks/vllm_optimizer.py`
- `src/autoparallel/frameworks/vllm_config.py`
- `src/autoparallel/deployment/vllm_commands.py`

**Key Features:**
```python
# vLLM-specific optimization
class VLLMOptimizer:
    def optimize_kv_cache_vs_cuda_graphs(self, config) -> VLLMConfig:
        # Balance KV cache size vs CUDA graph memory
        # Optimize for specific GPU architectures
        # Generate vLLM engine configuration
        
    def generate_deployment_command(self, config) -> str:
        # Return ready-to-run vLLM command
        # Include all optimized parameters
```

**Memory Optimization:**
- KV cache vs CUDA graph memory tradeoff analysis
- GPU architecture-specific optimizations (H100, A100, etc.)
- Automatic engine configuration parameter tuning

**Integration Testing:**
- Mock vLLM engine creation for validation
- Configuration parameter verification
- Deployment command generation testing

#### 2.2 DeepSpeed Integration
**Priority: Medium**
**Files to implement:**
- `src/autoparallel/frameworks/deepspeed_optimizer.py`
- `src/autoparallel/frameworks/zero_config.py`
- `src/autoparallel/deployment/deepspeed_commands.py`

**ZeRO Optimization:**
```python
class DeepSpeedOptimizer:
    def optimize_zero_config(self, config) -> DeepSpeedConfig:
        # Configure ZeRO stage (1/2/3) based on memory
        # Optimize gradient accumulation
        # Set communication backend parameters
```

**Features:**
- Automatic ZeRO stage selection based on memory constraints
- Training-specific optimizations (gradient accumulation, communication)
- Integration with activation checkpointing and CPU offloading

#### 2.3 Public API Implementation
**Priority: Critical**
**Files to implement:**
- `src/autoparallel/api/simple.py`
- `src/autoparallel/api/advanced.py`
- `src/autoparallel/__init__.py` (update with public interface)

**Progressive Complexity Design:**
```python
# Simple interface
def analyze(model: str, cluster: dict) -> AnalysisResult:
    """One-line optimal configuration analysis"""

def optimize(model: str, cluster: dict, workload: str) -> OptimizedConfig:
    """Workload-optimized configuration with deployment ready commands"""

# Advanced interface  
class AutoParallel:
    def __init__(self, config: Optional[AutoParallelConfig] = None):
        """Advanced configuration with custom constraints"""
    
    def analyze_model(self, model_config) -> DetailedAnalysis:
        """Detailed analysis with component breakdown"""
```

**Result Interface:**
- Configuration details with memory breakdown
- Performance predictions (throughput, latency estimates)
- Deployment commands for vLLM/DeepSpeed
- Human-readable decision explanations

### Phase 3: Enhanced Features

#### 3.1 Cluster Auto-Detection
**Priority: Medium**
**Files to implement:**
- `src/autoparallel/cluster/detection.py`
- `src/autoparallel/cluster/topology.py`

**Auto-Detection Sources:**
```python
def detect_cluster() -> ClusterConfig:
    # SLURM environment variables
    # torchrun distributed settings  
    # Local GPU detection via nvidia-ml-py
    # Network topology inference
```

**Topology Awareness:**
- Intra-node vs inter-node efficiency modeling
- NVSwitch vs RDMA communication costs
- Automatic NCCL backend optimization

#### 3.2 Workload Profiling
**Priority: Medium**  
**Files to implement:**
- `src/autoparallel/workload/profiler.py`
- `src/autoparallel/workload/patterns.py`

**Workload Types:**
- **Chatbot**: Interactive latency optimization, small batch sizes
- **Batch Inference**: Throughput optimization, large batch processing
- **Training**: Memory optimization, gradient accumulation efficiency

**Profiling Metrics:**
- Request rate patterns and batch size distributions
- Latency requirements and throughput targets
- Memory utilization patterns and growth trends

### Phase 4: Validation and Production Readiness

#### 4.1 Empirical Validation
**Priority: High**
**Files to implement:**
- `tests/validation/memory_accuracy.py`
- `tests/validation/performance_benchmarks.py`
- `scripts/validate_against_real_deployments.py`

**Validation Strategy:**
- Limited model loading for critical accuracy tests
- Framework deployment verification (where feasible)
- Memory estimation accuracy measurement (<10% error target)

**Target Models for Validation:**
- **Dense**: Qwen3-14B/32B (bf16, 32K context), Llama-3.1-70B (bf16, 128K)
- **MoE**: Qwen3-30B-A3B (128 experts), DeepSeek-V3 (256 experts)
- **Extreme**: Llama4-Scout (10M context), Kimi-K2 (384 experts)
- **Multimodal**: Llama4 text+image processing

#### 4.2 Performance Optimization
**Priority: Medium**
**Files to optimize:**
- Memory calculation algorithms
- Configuration enumeration efficiency
- Meta-device loading performance

**Optimization Targets:**
- Analysis time <5 seconds for most models
- Memory footprint <100MB during analysis
- Support for concurrent analysis requests

#### 4.3 Documentation and Examples
**Priority: High**
**Files to create:**
- `README.md` with comprehensive usage examples
- `docs/` directory with detailed guides
- `examples/` directory with real-world scenarios

**Documentation Structure:**
- Quick start guide with common use cases
- API reference with all configuration options
- Framework integration guides (vLLM, DeepSpeed)
- Troubleshooting and FAQ sections

## Implementation Priority Matrix

### Critical Path (Blocks other work):
1. Memory Estimation Framework
2. Architecture Constraint Analyzer
3. Configuration Generator
4. Public API Implementation

### High Priority (Core functionality):
1. vLLM Integration
2. Testing Infrastructure
3. Empirical Validation
4. Documentation

### Medium Priority (Enhancement features):
1. DeepSpeed Integration
2. Cluster Auto-Detection
3. Workload Profiling
4. Performance Optimization

## Risk Mitigation Strategies

### Technical Risks:
1. **Memory Estimation Accuracy**: Implement extensive validation against real deployments
2. **Framework API Changes**: Use stable APIs and version pinning with compatibility layers
3. **Model Architecture Evolution**: Design extensible constraint detection system

### Implementation Risks:
1. **Complexity Underestimation**: Focus on MVP functionality first, iterate rapidly
2. **Testing Coverage**: Prioritize meta-device testing, selective real model validation
3. **Performance Requirements**: Profile early, optimize bottlenecks incrementally

### Dependencies:
1. **HuggingFace Transformers**: Pin stable versions, test meta-device compatibility
2. **PyTorch Evolution**: Monitor memory APIs, adapt to changes proactively
3. **Framework Updates**: Track vLLM/DeepSpeed releases, maintain compatibility

## Success Metrics

### Functional Metrics:
- **Memory Accuracy**: <10% error vs actual framework usage
- **Configuration Coverage**: Support 95% of common parallelism scenarios
- **Analysis Speed**: <30 seconds for any supported model
- **API Usability**: Single function call for basic use cases

### Quality Metrics:
- **Test Coverage**: 90% line coverage, 85% branch coverage
- **Performance**: All unit tests <1s, integration tests <30s
- **Documentation**: Complete API reference, usage examples
- **Type Safety**: Full type annotation coverage with py.typed

### Adoption Metrics:
- **Framework Integration**: Working vLLM and DeepSpeed deployment
- **Model Support**: 20+ popular model architectures validated
- **Community Validation**: External testing and feedback incorporation
- **Production Ready**: Documented deployment patterns and troubleshooting

## Development Workflow

### Daily Development:
1. **Code Quality**: `uv run ruff check --fix && uv run ruff format`
2. **Type Checking**: `uv run ty check`
3. **Test Execution**: `uv run pytest --cov`
4. **Incremental Testing**: Focus on changed components

### Regular Milestones:
1. **Feature Completion**: Complete major components incrementally
2. **Integration Testing**: Cross-component validation
3. **Documentation Updates**: Keep specs and examples current
4. **Performance Review**: Profile and optimize bottlenecks

### Release Preparation:
1. **Comprehensive Testing**: Full test suite execution
2. **Documentation Review**: Complete API reference validation
3. **Example Verification**: All usage examples working
4. **Version Control**: Proper semantic versioning with jj

## Conclusion

This implementation plan provides a systematic approach to building AutoParallel as a production-ready library. The phased approach ensures core functionality is established first, with enhanced features added incrementally. The emphasis on meta-device testing and architecture-driven design enables comprehensive validation without prohibitive computational costs.

The detailed specifications provide a complete foundation, requiring primarily implementation rather than additional design work. Success depends on maintaining focus on the critical path while building a robust, extensible architecture that can evolve with the rapidly changing LLM landscape.

Additional enhancement ideas beyond the current scope are documented in [IDEAS.md](file:///home/jm/autoparallel/IDEAS.md).
