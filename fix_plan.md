# AutoParallel Implementation Plan

> **Structure Note**: This plan is organized as a comprehensive bullet-point list of implementation steps, referencing detailed specifications in `spec/` and current source code state. When editing, maintain this concise bullet-point structure - avoid verbose explanations that duplicate spec content.

## Current State Assessment
- Codebase: Memory estimation framework complete with comprehensive test coverage
- Specifications: Complete architectural docs in `spec/` directory
- Dependencies: Configured Astral stack (uv, ruff, ty, pytest)
- Implementation Status: **Phase 1, 2, 3 & 4 COMPLETED** - public API implementation successfully deployed (41 new tests added, 399 total tests, 96% coverage), ready for Phase 5

## Phase-Based Implementation Plan

### Phase 0: Scaffolding & Tooling (Priority: Critical - NOW) ✅ COMPLETED
- ✅ Create package directories & __init__.py stubs:
  - ✅ `src/autoparallel/memory/`
  - ✅ `src/autoparallel/constraints/`
  - ✅ `src/autoparallel/config/`
  - ✅ `src/autoparallel/api/`
  - ✅ `src/autoparallel/frameworks/`
  - ✅ `src/autoparallel/cluster/`
  - ✅ `src/autoparallel/workload/`
  - ✅ `src/autoparallel/deployment/`
  - ✅ `src/autoparallel/validation/`
- ✅ Add `py.typed` marker for type checking
- Configure entry points in pyproject.toml for pip install -e .
- Setup CI workflow: ruff check+format, ty check, pytest
- ✅ Create conftest.py with basic pytest fixtures

### Phase 1: Memory Estimation Framework (Priority: Critical) ✅ COMPLETED
- ✅ Create `src/autoparallel/memory/estimator.py`
  - ✅ Implement unified memory calculation (weights, activations, KV cache, CUDA graphs)
  - ✅ Support GPTQ, AWQ, bitsandbytes quantization formats
  - ✅ Add configurable safety margins and fragmentation overhead
  - ✅ Implement meta-device analysis for zero-memory introspection
- ✅ Create `src/autoparallel/memory/components.py`
  - ✅ Define MemoryComponents dataclass (weights, activations, kv_cache, cuda_graphs, optimizer_states, fragmentation_overhead)
  - ✅ Implement component-wise memory breakdown calculations
- ✅ Create `src/autoparallel/memory/config.py`
  - ✅ Define MemoryConfig with defaults (utilization_bound=0.85, fragmentation_overhead=0.10, safety_margin=0.05, quantization_format="fp16")
- ✅ Create `src/autoparallel/memory/estimator_test.py`
  - ✅ Created missing estimator_test.py with 99% coverage
  - ✅ Unit tests for each memory component calculation
  - ✅ Edge case handling (extreme model sizes, memory constraints)
- ✅ Create `src/autoparallel/memory/config_test.py`
  - ✅ Created missing config_test.py with 100% coverage
  - ✅ Comprehensive validation of MemoryConfig defaults and validation logic
- ✅ Create `src/autoparallel/memory/components_test.py`
  - ✅ Validation against different quantization formats
  - ✅ Fixed import inconsistencies across constraint files
  - ✅ Fixed type annotation issues

### Phase 2: Architecture Constraint Analyzer (Priority: Critical) ✅ COMPLETED
- ✅ Create `src/autoparallel/constraints/analyzer.py`
  - ✅ Implement ModelConstraints dataclass (max_tensor_parallel, max_pipeline_parallel, max_expert_parallel, vocabulary_sharding)
  - ✅ Add analyze_model_constraints() function for automatic constraint detection
  - ✅ Support dense transformers (Llama, Qwen, Mistral with GQA)
  - ✅ Support MoE models (DeepSeek-V3, Kimi-K2, Qwen3 MoE)
  - ✅ Support multimodal (Llama4 Scout/Maverick)
  - ✅ Support long context (1M-10M tokens)
- ✅ Create `src/autoparallel/constraints/model_support.py`
  - ✅ Architecture-specific constraint calculation logic
  - ✅ Model family detection and constraint mapping
- ✅ Create `src/autoparallel/constraints/validation.py`
  - ✅ Constraint validation and sanity checking
- ✅ Create `src/autoparallel/constraints/analyzer_test.py`
  - ✅ Meta-device loading for model config analysis
- ✅ Create `src/autoparallel/constraints/model_support_test.py`
  - ✅ Constraint validation across different architectures
  - ✅ Edge case testing (single layer models, massive MoE)

### Phase 3: Configuration Generator (Priority: Critical) ✅ COMPLETED
- ✅ Create `src/autoparallel/config/generator.py`
  - ✅ Implement generate_valid_configs() for TP/PP/EP/DP enumeration
  - ✅ Filter configurations by memory constraints
  - ✅ Score by workload-specific metrics (throughput, latency, memory efficiency, cost)
- ✅ Create `src/autoparallel/config/optimizer.py`
  - ✅ Configuration ranking and optimization algorithms
  - ✅ Workload-specific optimization targets
- ✅ Create `src/autoparallel/config/validator.py`
  - ✅ Configuration validation logic
  - ✅ Cross-component constraint checking
- ✅ Create `src/autoparallel/config/generator_test.py`
- ✅ Create `src/autoparallel/config/optimizer_test.py`
- ✅ Create `src/autoparallel/config/validator_test.py`
- ✅ Create `src/conftest.py` (pytest fixtures)
- ✅ **Successfully implemented**: All three major components (generator.py, optimizer.py, validator.py) with 188 Phase 3 tests passing

### Phase 4: Public API Implementation (Priority: Critical) ✅ COMPLETED
- ✅ Create `src/autoparallel/api/simple.py`
  - ✅ Implemented analyze(model: str, cluster: dict) -> AnalysisResult
  - ✅ Implemented optimize(model: str, cluster: dict, workload: str) -> OptimizedConfig
- ✅ Create `src/autoparallel/api/advanced.py`
  - ✅ Implemented AutoParallel class with advanced configuration
  - ✅ Added analyze_model() method with detailed analysis
- ✅ Update `src/autoparallel/__init__.py`
  - ✅ Exported public interface functions and classes
  - ✅ Cleaned up placeholder code
- ✅ **Successfully implemented**: Both simple and advanced APIs with comprehensive tests (41 new tests, all passing)

### Phase 5: vLLM Integration & Autotuning (Priority: High)
- Create `src/autoparallel/frameworks/vllm_optimizer.py`
  - Implement VLLMOptimizer class
  - Add optimize_kv_cache_vs_cuda_graphs() method
  - Add generate_deployment_command() method
  - GPU architecture-specific optimizations (H100, A100)
- Create `src/autoparallel/frameworks/vllm_config.py`
  - vLLM-specific configuration handling
  - Engine parameter optimization
- Create `src/autoparallel/deployment/vllm_commands.py`
  - Ready-to-run vLLM command generation
- Create `src/autoparallel/frameworks/vllm_optimizer_test.py`
  - Mock vLLM engine creation for validation
- Create `src/autoparallel/frameworks/vllm_config_test.py`
  - Configuration parameter verification
- Create `src/autoparallel/deployment/vllm_commands_test.py`
  - Deployment command generation testing

### Phase 6: Empirical Validation & Benchmarks (Priority: High)
- Create `src/validation/memory_accuracy_test.py`
  - Limited model loading for accuracy tests
  - Memory estimation error measurement (<10% target)
- Create `src/validation/performance_benchmarks_test.py`
  - Performance target validation
  - Analysis speed benchmarks (<30 seconds)
- Create `scripts/validate_against_real_deployments.py`
  - Framework deployment verification
  - Real-world validation scenarios
- Test target models:
  - Dense: Qwen3-14B/32B (bf16, 32K), Llama-3.1-70B (bf16, 128K)
  - MoE: Qwen3-30B-A3B (128 experts), DeepSeek-V3 (256 experts)
  - Extreme: Llama4-Scout (10M context), Kimi-K2 (384 experts)
  - Multimodal: Llama4 text+image

### Phase 7: DeepSpeed Integration (Priority: Medium)
- Create `src/autoparallel/frameworks/deepspeed_optimizer.py`
  - Implement DeepSpeedOptimizer class
  - Add optimize_zero_config() method
  - Automatic ZeRO stage selection (1/2/3)
  - Training-specific optimizations
- Create `src/autoparallel/frameworks/zero_config.py`
  - ZeRO configuration handling
  - Gradient accumulation optimization
- Create `src/autoparallel/deployment/deepspeed_commands.py`
  - DeepSpeed command generation

### Phase 8: Cluster Auto-Detection & Workload Profiling (Priority: Medium)
- Create `src/autoparallel/cluster/detection.py`
  - Implement detect_cluster() function
  - SLURM environment variable detection
  - torchrun distributed settings detection
  - Local GPU detection via nvidia-ml-py
- Create `src/autoparallel/cluster/topology.py`
  - Network topology inference
  - Intra-node vs inter-node efficiency modeling
  - NVSwitch vs RDMA communication costs
  - NCCL backend optimization

- Create `src/autoparallel/workload/profiler.py`
  - Request rate pattern analysis
  - Batch size distribution analysis
  - Latency requirement assessment
- Create `src/autoparallel/workload/patterns.py`
  - Workload type definitions (chatbot, batch inference, training)
  - Performance metrics calculation

### Phase 9: Performance Optimization & Documentation (Priority: Medium)
- Optimize memory calculation algorithms (<5 seconds analysis time)
- Optimize configuration enumeration efficiency
- Optimize meta-device loading performance
- Support concurrent analysis requests
- Memory footprint <100MB during analysis

- Update `README.md` with comprehensive usage examples
- Create `docs/` directory with detailed guides
- Create `examples/` directory with real-world scenarios
- Add quick start guide with common use cases
- Create API reference with all configuration options
- Add framework integration guides (vLLM, DeepSpeed)
- Create troubleshooting and FAQ sections

## Testing Strategy
- Meta-device first approach (HuggingFace meta devices for memory-free testing)
- Architecture-driven testing (model configs, not loaded weights)
- Selective integration testing for critical validation
- Performance targets: unit tests <1s, integration tests <30s, system tests <5min
- Coverage targets: 90% line coverage, 85% branch coverage

## Quality Assurance
- Run `uv run ruff check --fix && uv run ruff format` for code quality
- Run `uv run ty check` for type checking
- Run `uv run pytest --cov` for testing with coverage
- Follow Google Python Style Guide conventions
- Use snake_case for variables/functions, PascalCase for classes
- Full type annotation coverage with py.typed

## Success Criteria
- Memory accuracy <10% error vs actual framework usage
- Configuration coverage for 95% of common parallelism scenarios
- Analysis speed <30 seconds for any supported model
- Single function call for basic use cases
- Working vLLM and DeepSpeed deployment
- 20+ popular model architectures validated
