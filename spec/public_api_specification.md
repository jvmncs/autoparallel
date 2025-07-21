# AutoParallel Public API Specification

## Overview

AutoParallel is a library for automatically determining optimal parallelism and deployment configurations for Large Language Models across GPU clusters. This specification defines the public-facing API that prioritizes ease of use, powerful defaults, and gradual exposure of advanced features.

## Design Principles

1. **Simple by default, powerful when needed** - Most users should achieve great results with minimal configuration
2. **Workload-aware optimization** - Different use cases (inference vs training, chatbots vs batch processing) require different optimization strategies
3. **Transparent decision-making** - Users can understand why certain configurations were chosen
4. **Resource-aware** - Considers real hardware constraints, memory limits, and network topology
5. **Framework-agnostic insights** - Core analysis works across vLLM, DeepSpeed, etc.

## Core API

### Primary Interface

```python
import autoparallel

# Simplest possible usage - analyze all viable configurations
result = autoparallel.analyze(
    model="meta-llama/Llama-2-70b-hf",
    cluster={"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 4}
)

# Workload-specific optimization
result = autoparallel.optimize(
    model="microsoft/DialoGPT-medium",
    cluster={"gpu_memory_gb": 40, "gpus_per_node": 4, "num_nodes": 2},
    workload="inference",
    target="throughput"  # or "latency", "min_cost"
)

# Advanced usage with detailed workload specification
result = autoparallel.optimize(
    model="meta-llama/Llama-2-7b-hf",
    cluster=autoparallel.Cluster.from_slurm(),  # auto-detect
    workload=autoparallel.Workload(
        type="chatbot",
        requests_per_second=100,
        batch_size_distribution={1: 0.6, 2: 0.3, 4: 0.1},
        sequence_length_avg=1024,
        target_latency_p95_ms=150
    ),
    preferences=autoparallel.Preferences(
        memory_conservatism="moderate",  # conservative/moderate/aggressive
        precision="auto",  # or fp16, int8, int4
        framework="vllm"  # optional framework hint for optimization
    )
)
```

### Configuration Objects

```python
# Cluster specification - multiple ways to define
cluster = autoparallel.Cluster(
    gpu_memory_gb=80,
    gpus_per_node=8, 
    num_nodes=4,
    gpu_type="A100",  # optional, for better memory/compute estimates
    interconnect="nvlink"  # optional: nvlink, infiniband, ethernet
)

# Auto-detection from environment
cluster = autoparallel.Cluster.from_slurm()
cluster = autoparallel.Cluster.from_torchrun_env()
cluster = autoparallel.Cluster.from_local_gpus()

# Workload specification
workload = autoparallel.Workload(
    type="chatbot",  # chatbot, batch_inference, training, interactive
    requests_per_second=50,
    batch_size_distribution={1: 0.8, 2: 0.2},
    sequence_length_avg=512,
    sequence_length_max=2048,
    target_latency_p95_ms=100
)

# Training workload
workload = autoparallel.Workload(
    type="training",
    global_batch_size=256,
    micro_batch_size=4,
    sequence_length=2048,
    gradient_checkpointing=True
)

# User preferences
preferences = autoparallel.Preferences(
    memory_conservatism="moderate",  # how aggressive with memory usage
    precision="auto",  # quantization preferences  
    optimization_time="fast",  # fast/thorough - how long to spend optimizing
    explain_decisions=True  # include explanations in results
)
```

## Result Interface

```python
# Result provides comprehensive information
class OptimizationResult:
    # Primary recommendation
    recommended: Configuration
    
    # All viable configurations, ranked by target metric
    configurations: List[Configuration] 
    
    # Analysis insights
    insights: AnalysisInsights
    
    # Methods for deployment
    def deploy_vllm(self) -> str:
        """Generate vLLM startup command"""
    
    def deploy_deepspeed(self) -> Dict:
        """Generate DeepSpeed configuration"""
    
    def deploy_torchrun(self) -> str:
        """Generate torchrun command for training"""
        
    def explain(self) -> str:
        """Human-readable explanation of recommendations"""

# Individual configuration
class Configuration:
    # Parallelism strategy
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    expert_parallel: int  # for MoE models
    
    # Resource allocation
    memory_per_gpu_gb: float
    memory_breakdown: MemoryBreakdown
    
    # Performance predictions
    estimated_throughput: Optional[float]
    estimated_latency_p95_ms: Optional[float]
    
    # Framework-specific optimizations
    vllm_config: Optional[VLLMConfig]
    deepspeed_config: Optional[DeepSpeedConfig]
    
    # Validity and constraints
    is_valid: bool
    constraint_violations: List[str]
    warnings: List[str]

# Memory breakdown for transparency
class MemoryBreakdown:
    model_weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    gradients_gb: float  # training only
    optimizer_states_gb: float  # training only
    cuda_graphs_gb: float  # vLLM only
    safety_margin_gb: float
    total_gb: float

# Analysis insights
class AnalysisInsights:
    # Constraints that limited configurations
    limiting_factors: List[str]  # e.g., "attention heads", "gpu memory", "vocabulary size"
    
    # Recommendations for better performance
    suggestions: List[str]  # e.g., "Consider int8 quantization for 2x more models"
    
    # Cluster utilization
    gpu_utilization_percent: float
    memory_utilization_percent: float
    
    # Scaling opportunities
    scaling_recommendations: List[str]
```

## Framework-Specific Integrations

### vLLM Integration

```python
# vLLM-specific optimization
result = autoparallel.optimize_vllm(
    model="meta-llama/Llama-2-13b-hf",
    cluster=cluster,
    workload=workload
)

# Direct deployment
vllm_args = result.recommended.deploy_vllm()
# Returns: "--model meta-llama/Llama-2-13b-hf --tensor-parallel-size 4 --max-model-len 2048 ..."

# Programmatic usage
vllm_config = result.recommended.vllm_config
engine = LLMEngine.from_engine_args(vllm_config.to_engine_args())
```

### DeepSpeed Integration

```python
# DeepSpeed-specific optimization  
result = autoparallel.optimize_deepspeed(
    model="meta-llama/Llama-2-70b-hf",
    cluster=cluster,
    workload=autoparallel.TrainingWorkload(
        global_batch_size=256,
        sequence_length=2048
    )
)

# Direct deployment
deepspeed_config = result.recommended.deploy_deepspeed()
# Returns complete DeepSpeed JSON configuration
```

## Advanced Features

### Custom Constraints

```python
# Add custom constraints for special requirements
constraints = autoparallel.Constraints(
    max_pipeline_stages=4,  # limit pipeline depth
    require_even_expert_split=True,  # for MoE models
    min_batch_size_per_gpu=2,
    max_memory_utilization=0.85,
    require_identical_gpus=True
)

result = autoparallel.optimize(
    model=model,
    cluster=cluster, 
    workload=workload,
    constraints=constraints
)
```

### Multi-Model Analysis

```python
# Compare multiple models on same cluster
comparison = autoparallel.compare_models(
    models=["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"],
    cluster=cluster,
    workload=workload
)

for model, result in comparison.items():
    print(f"{model}: {result.recommended.estimated_throughput} tokens/sec")
```

### Cluster Planning

```python
# Find minimum cluster size for requirements
min_cluster = autoparallel.plan_cluster(
    model="meta-llama/Llama-2-70b-hf",
    workload=workload,
    target_throughput=1000  # tokens/sec
)

# Cost optimization across cloud providers
cost_analysis = autoparallel.cost_optimize(
    model=model,
    workload=workload,
    providers=["aws", "gcp", "azure"],
    budget_per_hour=50
)
```

## Error Handling & Validation

```python
# Graceful handling of impossible configurations
try:
    result = autoparallel.optimize(
        model="meta-llama/Llama-2-70b-hf",
        cluster={"gpu_memory_gb": 16, "gpus_per_node": 1, "num_nodes": 1}  # too small
    )
except autoparallel.InsufficientResourcesError as e:
    print(f"Cannot fit model: {e.message}")
    print(f"Minimum requirements: {e.minimum_cluster}")
    
# Warnings for suboptimal configurations
result = autoparallel.analyze(model, cluster)
if result.warnings:
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

## Integration Patterns

### Jupyter/Research Usage

```python
# Rich display in notebooks
result = autoparallel.optimize(model, cluster, workload)
result.display()  # Rich table with configurations and trade-offs
result.plot_pareto_frontier()  # Throughput vs latency trade-offs
```

### Production Deployment

```python
# Production deployment with monitoring
config = autoparallel.optimize(model, cluster, workload).recommended

# Deploy with automatic health checks
deployment = autoparallel.deploy(
    config=config,
    health_check_endpoint="/health",
    rollback_on_failure=True
)

# Monitor and re-optimize based on actual usage
metrics = deployment.get_metrics(hours=24)
if metrics.p95_latency > workload.target_latency_p95_ms * 1.2:
    new_config = autoparallel.reoptimize(config, actual_workload=metrics)
```

## Key Benefits of This API Design

1. **Progressive complexity** - Simple cases are trivial, complex cases are possible
2. **Workload-aware** - Different optimization strategies for different use cases
3. **Transparent decisions** - Users understand why configurations were chosen  
4. **Framework integration** - Seamless deployment to vLLM, DeepSpeed, etc.
5. **Resource awareness** - Considers real hardware constraints and costs
6. **Production ready** - Error handling, monitoring, and re-optimization support

This API hides the complexity of constraint analysis, memory estimation, and framework-specific details while exposing the insights and control that users actually need for successful LLM deployment.
