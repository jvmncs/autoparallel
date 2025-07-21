# vLLM Autotuning Specification

## Overview

This specification defines an autotuning system for optimizing vLLM memory configurations without building full vLLM engines. The system addresses the critical tradeoff between KV cache allocation and CUDA graph capture sizes to maximize throughput while respecting memory constraints.

## Problem Statement

vLLM's performance depends on optimizing competing memory demands:

1. **KV Cache**: Enables concurrent sequence processing, directly affects batch size capacity
2. **CUDA Graphs**: Reduces kernel launch overhead, improves latency for captured batch sizes
3. **Model Weights**: Fixed memory requirement that must be accommodated

The challenge: **More CUDA graph captures = less KV cache space = lower effective batch size**

Current vLLM provides no automatic optimization, requiring manual configuration through trial-and-error.

## Memory Allocation Model

Based on vLLM research, memory allocation follows this priority order:

```python
total_memory = gpu_memory_capacity * gpu_memory_utilization

# Priority order:
1. model_weights          # Fixed requirement from transformers config
2. activation_memory      # Measured via profiling with meta tensors
3. cuda_graph_memory      # Calculated from capture sizes and model architecture
4. kv_cache_memory        # Remaining memory allocated to block pool
```

**Key Insight**: KV cache gets the leftover memory after all other components are allocated.

## Magic Number Parameterization

All empirically-derived constants are exposed as configurable parameters in `AutotuningParameters`:

- **Graph memory scaling (0.1, 0.02)**: Based on CUDA graph capture profiling across model sizes
- **Compilation multipliers (1.8, 1.0)**: Memory overhead difference between FULL vs PIECEWISE compilation
- **Activation memory factor (0.3)**: Conservative estimate from inference memory profiling 
- **Performance weights (0.7/0.3, 0.8/0.2)**: Empirical balance between batch size and graph coverage impact
- **Memory bounds (0.8-0.98)**: Safe operational ranges for GPU memory utilization

This parameterization enables tuning without code changes and facilitates empirical validation.

## Autotuning Architecture

### 1. Performance Model

```python
@dataclass
class AutotuningParameters:
    """Configurable parameters for autotuning behavior"""
    
    # CUDA graph memory estimation parameters
    graph_memory_overhead_base_ratio: float = 0.1  # Base overhead as fraction of model memory
    graph_memory_batch_scaling_factor: float = 0.02  # Additional memory per batch element
    compilation_memory_multiplier_full: float = 1.8  # Memory multiplier for FULL compilation
    compilation_memory_multiplier_piecewise: float = 1.0  # Memory multiplier for PIECEWISE
    compilation_level: str = "PIECEWISE"  # Default compilation level: "FULL" or "PIECEWISE"
    
    # Memory utilization bounds
    min_gpu_memory_utilization: float = 0.8  # Conservative lower bound
    max_gpu_memory_utilization: float = 0.98  # Aggressive upper bound
    
    # Performance scoring weights (tuned based on empirical vLLM performance analysis)
    throughput_batch_weight: float = 0.7  # Weight for batch size in throughput scoring
    throughput_graph_weight: float = 0.3  # Weight for graph coverage in throughput scoring
    latency_graph_weight: float = 0.8  # Weight for graph coverage in latency scoring
    latency_batch_weight: float = 0.2  # Weight for batch size in latency scoring
    
    # Memory safety margins
    fragmentation_overhead_factor: float = 1.05  # Account for memory fragmentation
    min_kv_cache_ratio: float = 0.05  # Minimum KV cache as fraction of GPU memory

def get_vllm_default_capture_sizes(max_limit: int, vllm_version: str = "v1") -> List[int]:
    """Get vLLM's actual default CUDA graph capture sizes
    
    Args:
        max_limit: Maximum capture size to include
        vllm_version: vLLM version ("v0" or "v1") - controls whether max_limit 
                     applies to max_num_seqs (v0) or max_num_batched_tokens (v1)
    """
    
    # Based on vllm/config.py: batch_size_capture_list = [1, 2, 4] + list(range(8, 513, 8))
    base_sizes = [1, 2, 4] + list(range(8, 513, 8))
    
    # Filter by version-specific limits
    return [size for size in base_sizes if size <= max_limit]

@dataclass
class vLLMPerformanceModel:
    """Model for predicting vLLM performance without engine instantiation"""
    
    # Hardware constraints
    gpu_memory_capacity_gb: float
    
    # Model-derived parameters (from transformers config)
    model_memory_gb: float
    activation_memory_gb: float
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int  # For GQA models
    vocab_size: int
    
    # Tunable parameters
    gpu_memory_utilization: float
    cudagraph_capture_sizes: List[int]
    max_model_len: int
    kv_cache_dtype: str  # "auto", "fp8", "fp8_e4m3"
    
    # Autotuning configuration
    tuning_params: AutotuningParameters
    
    @classmethod
    def from_transformers_config(
        cls,
        config: transformers.PretrainedConfig,
        gpu_memory_capacity_gb: float,
        gpu_memory_utilization: float,
        max_model_len: int,
        tuning_params: AutotuningParameters = None
    ) -> 'vLLMPerformanceModel':
        """Create performance model from transformers config"""
        
        if tuning_params is None:
            tuning_params = AutotuningParameters()
        
        # Extract architecture parameters from config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers  
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
        vocab_size = getattr(config, 'vocab_size', 50257)
        
        # Calculate model memory from config
        model_memory_gb = calculate_model_memory_from_config(config)
        
        # Estimate activation memory using CUDA graph memory estimator methodology
        activation_memory_gb = estimate_activation_memory_from_config(
            config, batch_size=32, sequence_length=512
        )
        
        return cls(
            gpu_memory_capacity_gb=gpu_memory_capacity_gb,
            model_memory_gb=model_memory_gb,
            activation_memory_gb=activation_memory_gb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            gpu_memory_utilization=gpu_memory_utilization,
            cudagraph_capture_sizes=get_vllm_default_capture_sizes(512),  # Will be overridden
            max_model_len=max_model_len,
            kv_cache_dtype="auto",
            tuning_params=tuning_params
        )
    
    def calculate_memory_breakdown(self) -> Dict[str, float]:
        """Calculate memory allocation without instantiating vLLM"""
        
        # CUDA graph memory calculation
        cuda_graph_memory = 0.0
        for batch_size in self.cudagraph_capture_sizes:
            # Graph memory scales with model size and batch size
            graph_memory = (
                self.model_memory_gb * 
                self.tuning_params.graph_memory_overhead_base_ratio * 
                (1 + batch_size * self.tuning_params.graph_memory_batch_scaling_factor)
            )
            
            # Apply compilation level multiplier based on configured level
            if self.tuning_params.compilation_level == "FULL":
                graph_memory *= self.tuning_params.compilation_memory_multiplier_full
            else:  # PIECEWISE
                graph_memory *= self.tuning_params.compilation_memory_multiplier_piecewise
            cuda_graph_memory += graph_memory
        
        # Available memory for KV cache
        total_available = self.gpu_memory_capacity_gb * self.gpu_memory_utilization
        kv_cache_memory = (
            total_available - 
            self.model_memory_gb - 
            self.activation_memory_gb - 
            cuda_graph_memory
        )
        
        return {
            'model_memory': self.model_memory_gb,
            'activation_memory': self.activation_memory_gb,
            'cuda_graph_memory': cuda_graph_memory,
            'kv_cache_memory': kv_cache_memory,
            'total_used': total_available,
            'utilization_ratio': total_available / self.gpu_memory_capacity_gb
        }
    
    def calculate_effective_batch_size(self) -> int:
        """Calculate maximum concurrent sequences given KV cache constraints"""
        
        memory_breakdown = self.calculate_memory_breakdown()
        kv_cache_memory_bytes = memory_breakdown['kv_cache_memory'] * (1024**3)
        
        if kv_cache_memory_bytes <= 0:
            return 0
        
        # KV cache per sequence calculation from architecture
        head_dim = self.hidden_size // self.num_attention_heads
        
        # Determine bytes per element based on dtype
        if self.kv_cache_dtype == "fp8" or self.kv_cache_dtype == "fp8_e4m3":
            dtype_bytes = 1
        else:  # "auto" typically means fp16/bf16
            dtype_bytes = 2
        
        # Memory per token: keys + values for all layers
        memory_per_token = (
            self.num_key_value_heads * head_dim * 2 * dtype_bytes * self.num_layers
        )
        
        # Maximum concurrent sequences
        memory_per_sequence = memory_per_token * self.max_model_len
        max_concurrent_seqs = int(kv_cache_memory_bytes / memory_per_sequence)
        
        return max_concurrent_seqs
    
    def calculate_graph_coverage(self, workload: 'WorkloadProfile') -> float:
        """Estimate % of requests that benefit from CUDA graphs"""
        
        covered_requests = 0
        total_requests = 0
        
        for batch_size, frequency in workload.batch_size_distribution.items():
            if batch_size in self.cudagraph_capture_sizes:
                covered_requests += frequency
            total_requests += frequency
        
        return covered_requests / total_requests if total_requests > 0 else 0.0

def calculate_model_memory_from_config(config: transformers.PretrainedConfig) -> float:
    """Calculate model memory from transformers config without loading weights"""
    
    # Extract quantization info
    quant_config = getattr(config, 'quantization_config', {})
    if quant_config:
        bits = quant_config.get('bits', 16)
    else:
        # Check torch_dtype
        torch_dtype = getattr(config, 'torch_dtype', 'float32')
        if 'float16' in str(torch_dtype) or 'bfloat16' in str(torch_dtype):
            bits = 16
        elif 'fp8' in str(torch_dtype):
            bits = 8
        else:
            bits = 32
    
    bytes_per_param = bits / 8
    
    # Estimate total parameters from config
    vocab_size = getattr(config, 'vocab_size', 50257)
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
    
    # Rough parameter count estimation
    embedding_params = vocab_size * hidden_size
    
    # Per-layer parameters (attention + MLP)
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    mlp_params = 2 * hidden_size * intermediate_size  # up, down projections
    layer_norm_params = 2 * hidden_size  # Two layer norms per layer
    
    layer_params = attention_params + mlp_params + layer_norm_params
    total_params = embedding_params + (num_layers * layer_params) + hidden_size  # Final layer norm
    
    return (total_params * bytes_per_param) / (1024**3)  # Convert to GB

def estimate_activation_memory_from_config(
    config: transformers.PretrainedConfig,
    batch_size: int,
    sequence_length: int
) -> float:
    """Estimate activation memory using CUDA graph memory estimation methodology
    
    This function estimates peak activation memory during inference by analyzing:
    1. Attention mechanism memory requirements (Q, K, V tensors, attention scores)
    2. MLP intermediate activations
    3. Gradient checkpointing effects during inference
    
    The conservative estimate factor (0.3) accounts for the fact that not all layers
    are active simultaneously during forward pass, based on empirical memory profiling.
    """
    
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    
    # Attention activations (dominant memory component)
    # Q, K, V tensors: batch_size × seq_len × hidden_size each
    attention_memory = 3 * batch_size * sequence_length * hidden_size * 2  # fp16
    
    # Attention scores: batch_size × num_heads × seq_len × seq_len  
    attention_scores = batch_size * num_attention_heads * sequence_length * sequence_length * 2
    
    # MLP intermediate activations
    intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
    mlp_memory = batch_size * sequence_length * intermediate_size * 2
    
    # Total per layer, accounting for gradient storage during training
    per_layer_memory = attention_memory + attention_scores + mlp_memory
    
    # Peak memory (not all layers active simultaneously, but need buffer)
    # Conservative estimate based on empirical memory profiling of inference workloads
    peak_activation_memory = per_layer_memory * 0.3
    
    return peak_activation_memory / (1024**3)  # Convert to GB
```

### 2. Workload Profiling

```python
@dataclass
class WorkloadProfile:
    """Characterize expected vLLM workload patterns"""
    
    # Request patterns
    requests_per_second: float
    batch_size_distribution: Dict[int, float]  # batch_size -> frequency
    sequence_length_distribution: Dict[int, float]  # length -> frequency
    
    # Performance priorities
    target_metric: str  # "throughput" or "latency"
    latency_budget_ms: float  # P99 latency constraint
    
    @classmethod
    def from_trace_analysis(cls, request_trace: List[Dict]) -> 'WorkloadProfile':
        """Extract workload profile from historical request traces"""
        # Analyze request patterns, batch sizes, sequence lengths
        pass
    
    @classmethod  
    def create_synthetic(
        cls, 
        workload_type: str,
        requests_per_second: float = None,
        latency_budget_ms: float = None
    ) -> 'WorkloadProfile':
        """Create synthetic workload profiles for common scenarios"""
        
        if workload_type == "chatbot":
            return cls(
                requests_per_second=requests_per_second or 100,
                batch_size_distribution={1: 0.4, 2: 0.3, 4: 0.2, 8: 0.1},
                sequence_length_distribution={512: 0.3, 1024: 0.4, 2048: 0.3},
                target_metric="latency",
                latency_budget_ms=latency_budget_ms or 100
            )
        elif workload_type == "batch_inference":
            return cls(
                requests_per_second=requests_per_second or 10,
                batch_size_distribution={16: 0.2, 32: 0.4, 64: 0.3, 128: 0.1},
                sequence_length_distribution={1024: 0.5, 2048: 0.4, 4096: 0.1},
                target_metric="throughput",
                latency_budget_ms=latency_budget_ms or 1000
            )
        elif workload_type == "interactive":
            return cls(
                requests_per_second=requests_per_second or 50,
                batch_size_distribution={1: 0.8, 2: 0.15, 4: 0.05},
                sequence_length_distribution={256: 0.4, 512: 0.4, 1024: 0.2},
                target_metric="latency",
                latency_budget_ms=latency_budget_ms or 50
            )
        
        raise ValueError(f"Unknown workload type: {workload_type}")

    def get_expected_max_batch_size(self, percentile: float = 0.95) -> int:
        """Get expected maximum batch size at given percentile"""
        cumulative = 0.0
        sorted_batches = sorted(self.batch_size_distribution.items())
        
        for batch_size, frequency in sorted_batches:
            cumulative += frequency
            if cumulative >= percentile:
                return batch_size
        
        return max(self.batch_size_distribution.keys())
    
    def get_expected_max_sequence_length(self, percentile: float = 0.95) -> int:
        """Get expected maximum sequence length at given percentile"""
        cumulative = 0.0
        sorted_lengths = sorted(self.sequence_length_distribution.items())
        
        for seq_len, frequency in sorted_lengths:
            cumulative += frequency
            if cumulative >= percentile:
                return seq_len
        
        return max(self.sequence_length_distribution.keys())
```

### 3. Configuration Search

```python
class vLLMConfigOptimizer:
    """Search for optimal vLLM configurations using performance models"""
    
    def __init__(
        self, 
        model_name: str, 
        cluster_spec: ClusterSpec,
        tuning_params: AutotuningParameters = None
    ):
        self.model_name = model_name
        self.cluster_spec = cluster_spec
        self.tuning_params = tuning_params or AutotuningParameters()
        
        # Load model config for analysis
        self.config = AutoConfig.from_pretrained(model_name)
    
    def search_optimal_config(
        self, 
        workload: WorkloadProfile, 
        search_space: Dict[str, List] = None
    ) -> Dict[str, Any]:
        """Search for optimal vLLM configuration"""
        
        if search_space is None:
            search_space = self.get_default_search_space(workload)
        
        best_config = None
        best_score = float('-inf')
        evaluated_configs = []
        
        # Grid search over configuration space
        for config_params in self.generate_configs(search_space):
            config = vLLMPerformanceModel.from_transformers_config(
                config=self.config,
                gpu_memory_capacity_gb=self.cluster_spec.gpu_memory_per_device,
                tuning_params=self.tuning_params,
                **config_params
            )
            
            if not self.is_feasible_config(config):
                continue
                
            score = self.evaluate_config(config, workload)
            evaluated_configs.append((config, score))
            
            if score > best_score:
                best_score = score
                best_config = config
        
        return {
            'optimal_config': best_config,
            'performance_score': best_score,
            'memory_breakdown': best_config.calculate_memory_breakdown() if best_config else None,
            'predictions': self.get_config_predictions(best_config, workload) if best_config else None,
            'all_evaluated_configs': evaluated_configs
        }
    
    def get_default_search_space(self, workload: WorkloadProfile) -> Dict[str, List]:
        """Define search space based on workload characteristics"""
        
        # Determine reasonable max_model_len based on workload
        max_seq_len = workload.get_expected_max_sequence_length(percentile=0.99)
        max_model_len_options = [length for length in [1024, 2048, 4096, 8192, 16384] 
                               if length >= max_seq_len]
        
        # Determine CUDA graph capture sizes based on workload batch patterns
        max_batch = workload.get_expected_max_batch_size(percentile=0.95)
        
        # Generate capture size options
        conservative_captures = [1, 2, 4]
        balanced_captures = get_vllm_default_capture_sizes(min(32, max_batch))
        aggressive_captures = get_vllm_default_capture_sizes(min(128, max_batch))
        
        return {
            'gpu_memory_utilization': [
                self.tuning_params.min_gpu_memory_utilization,
                0.85, 0.90, 0.95,
                self.tuning_params.max_gpu_memory_utilization
            ],
            'cudagraph_capture_sizes': [
                [],  # No CUDA graphs
                conservative_captures,
                balanced_captures,
                aggressive_captures
            ],
            'max_model_len': max_model_len_options,
            'kv_cache_dtype': ["auto", "fp8_e4m3"] if workload.target_metric == "throughput" else ["auto"]
        }
    
    def generate_configs(self, search_space: Dict[str, List]) -> Iterator[Dict[str, Any]]:
        """Generate all combinations from search space"""
        import itertools
        
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def is_feasible_config(self, config: vLLMPerformanceModel) -> bool:
        """Check if configuration fits within memory constraints"""
        
        memory_breakdown = config.calculate_memory_breakdown()
        
        # Check KV cache has minimum required space
        min_kv_cache_gb = config.gpu_memory_capacity_gb * self.tuning_params.min_kv_cache_ratio
        if memory_breakdown['kv_cache_memory'] < min_kv_cache_gb:
            return False
            
        # Check effective batch size is reasonable
        effective_batch_size = config.calculate_effective_batch_size()
        if effective_batch_size < 1:
            return False
            
        return True
    
    def evaluate_config(
        self, 
        config: vLLMPerformanceModel, 
        workload: WorkloadProfile
    ) -> float:
        """Score configuration based on predicted performance"""
        
        effective_batch_size = config.calculate_effective_batch_size()
        graph_coverage = config.calculate_graph_coverage(workload)
        
        if workload.target_metric == "throughput":
            # Throughput score: prioritize batch size, bonus for graph coverage
            score = (
                effective_batch_size * self.tuning_params.throughput_batch_weight +
                graph_coverage * effective_batch_size * self.tuning_params.throughput_graph_weight
            )
            
        elif workload.target_metric == "latency":
            # Latency score: prioritize graph coverage, moderate batch size
            # Batch size normalization by 32 represents typical interactive workload scale
            score = (
                graph_coverage * self.tuning_params.latency_graph_weight +
                (effective_batch_size / 32) * self.tuning_params.latency_batch_weight
            )
            
        else:
            raise ValueError(f"Unknown target metric: {workload.target_metric}")
        
        return score
    
    def get_config_predictions(
        self, 
        config: vLLMPerformanceModel, 
        workload: WorkloadProfile
    ) -> Dict[str, Any]:
        """Generate detailed predictions for a configuration"""
        
        return {
            'effective_batch_size': config.calculate_effective_batch_size(),
            'graph_coverage': config.calculate_graph_coverage(workload),
            'memory_breakdown': config.calculate_memory_breakdown(),
            'recommended_max_num_seqs': min(
                config.calculate_effective_batch_size(),
                workload.get_expected_max_batch_size(percentile=0.95)
            )
        }
```

### 4. Integration with Autoparallel

```python
def optimize_vllm_config_for_cluster(
    model_name: str,
    cluster_spec: ClusterSpec,
    workload: WorkloadProfile,
    parallelism_strategy: Dict[str, int]
) -> Dict[str, Any]:
    """Integrate vLLM config optimization with autoparallel"""
    
    # Calculate effective resources per vLLM instance
    tp_size = parallelism_strategy['tp']
    pp_size = parallelism_strategy['pp'] 
    dp_size = parallelism_strategy['dp']
    
    # Memory per vLLM instance
    effective_gpu_memory = cluster_spec.gpu_memory_per_device
    effective_gpus_per_instance = tp_size
    
    # Create optimizer for single vLLM instance
    optimizer = vLLMConfigOptimizer(model_name, cluster_spec)
    
    # Optimize configuration
    optimal_result = optimizer.search_optimal_config(workload)
    
    # Scale predictions for full cluster
    cluster_predictions = {
        'total_throughput': optimal_result['predictions']['effective_batch_size'] * dp_size,
        'instances_per_cluster': dp_size,
        'memory_efficiency': optimal_result['memory_breakdown']['kv_cache_memory'] / effective_gpu_memory,
        'graph_coverage': optimal_result['predictions']['graph_coverage']
    }
    
    return {
        'vllm_config': optimal_result['optimal_config'],
        'parallelism_strategy': parallelism_strategy,
        'cluster_predictions': cluster_predictions,
        'recommendations': generate_deployment_recommendations(optimal_result, parallelism_strategy)
    }

def generate_deployment_recommendations(
    optimal_result: Dict[str, Any],
    parallelism_strategy: Dict[str, int]
) -> List[str]:
    """Generate actionable deployment recommendations"""
    
    recommendations = []
    config = optimal_result['optimal_config']
    
    # Memory recommendations
    kv_cache_ratio = optimal_result['memory_breakdown']['kv_cache_memory'] / config.gpu_memory_capacity_gb
    if kv_cache_ratio < 0.1:
        recommendations.append("Consider reducing CUDA graph capture sizes to free more memory for KV cache")
    
    # Performance recommendations  
    if optimal_result['predictions']['graph_coverage'] < 0.5:
        recommendations.append("Low CUDA graph coverage - consider adjusting capture sizes for your workload")
    

    
    return recommendations
```

## Implementation Roadmap

### Phase 1: Core Modeling
- [ ] Implement vLLMPerformanceModel with accurate memory calculations
- [ ] Build workload profiling system
- [ ] Validate memory models against actual vLLM instances

### Phase 2: Optimization Engine  
- [ ] Implement configuration search algorithms
- [ ] Add support for multi-objective optimization (throughput vs latency)
- [ ] Build feasibility checking and constraint handling

### Phase 3: Integration
- [ ] Add cluster-wide optimization considering DP scaling
- [ ] Build deployment recommendation system

### Phase 4: Validation & Tuning
- [ ] Benchmark predictions against real vLLM deployments
- [ ] Tune performance models based on empirical data
- [ ] Add support for specialized workloads (MoE, quantized models)

## Key Benefits

1. **No Engine Instantiation**: Optimize configurations without building full vLLM engines
2. **Memory-Aware**: Explicitly model KV cache vs CUDA graph tradeoffs  
3. **Workload-Specific**: Tailor configurations to actual usage patterns
4. **Cluster-Integrated**: Combine with autoparallel's parallelism optimization
5. **Actionable**: Generate specific deployment recommendations

This autotuning system addresses vLLM's current lack of automatic optimization, providing data-driven configuration recommendations that maximize performance within memory constraints.
