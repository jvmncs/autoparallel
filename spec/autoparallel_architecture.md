# AutoParallel Architecture Specification

## Overview

AutoParallel is a tool for determining optimal parallelism configurations for LLM inference and training across GPU clusters. Given a Hugging Face transformers model identifier and cluster description, it analyzes all viable parallelism strategies considering tensor parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), and data parallelism (DP).

The tool is specifically designed to work with Hugging Face transformers models, leveraging the transformers config system, model architectures, and quantization formats for precise analysis while respecting architectural constraints that limit parallelization options.

## Core Architecture Components

### 1. Model Profiler

**Purpose**: Precise memory and computational analysis of Hugging Face transformers models using meta devices.

**Key Functions**:
- **Transformers-Native Loading**: Use `AutoConfig.from_pretrained()` and `AutoModel.from_pretrained()` with `device_map="meta"` for zero-memory model analysis
- **Memory Estimation**: Delegate to unified memory estimation framework for precise calculations
- **Architecture Constraint Analysis**: Determine parallelization limits based on model architecture
- **Quantization-Aware Analysis**: Detect and handle GPTQ, AWQ, bitsandbytes, and native precision formats
- **MoE Detection**: Programmatically identify MoE models via config inspection
- **Architecture-Specific Patterns**: Extract model-specific parameters from transformers configs

### 2. Architecture Constraints Analyzer

**Purpose**: Determine valid parallelism configurations based on model architecture limitations.

```python
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import math

@dataclass
class ParallelismConstraintParameters:
    """Configurable parameters for parallelism constraints"""
    
    # Default constraints
    default_min_layers_per_stage: int = 2
    default_max_tensor_parallel: int = 64
    min_experts_per_device: int = 1
    vocab_large_threshold: int = 100000
    vocab_medium_threshold: int = 50000
    vocab_large_divisibility: int = 8
    vocab_medium_divisibility: int = 4
    vocab_small_divisibility: int = 2

@dataclass
class ArchitectureConstraints:
    """Model architecture constraints that limit parallelization"""
    
    # Tensor parallelism constraints
    max_tensor_parallel: int  # Maximum TP size based on architecture
    tensor_parallel_divisors: Set[int]  # Valid TP sizes
    
    # Expert parallelism constraints (for MoE)
    max_expert_parallel: int  # Maximum EP size
    expert_parallel_divisors: Set[int]  # Valid EP sizes
    
    # Pipeline parallelism constraints
    max_pipeline_parallel: int  # Maximum PP size (typically num_layers)
    min_layers_per_stage: int  # Minimum layers per pipeline stage
    
    # Additional constraints
    requires_tied_embeddings: bool  # Input/output embeddings are tied
    supports_grouped_query_attention: bool  # Has GQA
    vocab_divisibility_requirement: int  # Vocab size divisibility for TP
    
    def get_valid_tensor_parallel_sizes(self, max_gpus: int) -> List[int]:
        """Get valid tensor parallel sizes up to max_gpus"""
        valid_sizes = []
        for size in self.tensor_parallel_divisors:
            if size <= min(max_gpus, self.max_tensor_parallel):
                valid_sizes.append(size)
        return sorted(valid_sizes)
    
    def get_valid_expert_parallel_sizes(self, max_gpus: int) -> List[int]:
        """Get valid expert parallel sizes up to max_gpus"""
        if self.max_expert_parallel == 0:
            return [1]  # Not an MoE model
        
        valid_sizes = []
        for size in self.expert_parallel_divisors:
            if size <= min(max_gpus, self.max_expert_parallel):
                valid_sizes.append(size)
        return sorted(valid_sizes)
    
    def get_valid_pipeline_parallel_sizes(self, max_nodes: int) -> List[int]:
        """Get valid pipeline parallel sizes up to max_nodes"""
        max_pp = min(max_nodes, self.max_pipeline_parallel)
        valid_sizes = []
        
        for size in range(1, max_pp + 1):
            layers_per_stage = self.max_pipeline_parallel / size
            if layers_per_stage >= self.min_layers_per_stage:
                valid_sizes.append(size)
        
        return valid_sizes

def analyze_architecture_constraints(
    config: transformers.PretrainedConfig,
    constraint_params: ParallelismConstraintParameters = None
) -> ArchitectureConstraints:
    """Analyze model architecture to determine parallelization constraints"""
    
    if constraint_params is None:
        constraint_params = ParallelismConstraintParameters()
    
    # Extract basic architecture parameters
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    num_layers = config.num_hidden_layers
    vocab_size = getattr(config, 'vocab_size', 50257)
    intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
    
    # Tensor parallelism constraints
    tp_constraints = _analyze_tensor_parallel_constraints(
        hidden_size, num_attention_heads, num_key_value_heads, 
        vocab_size, intermediate_size, constraint_params
    )
    
    # Expert parallelism constraints (MoE specific)
    ep_constraints = _analyze_expert_parallel_constraints(config, constraint_params)
    
    # Pipeline parallelism constraints
    pp_constraints = _analyze_pipeline_parallel_constraints(num_layers, constraint_params)
    
    # Additional architectural features
    tied_embeddings = _check_tied_embeddings(config)
    gqa_support = num_key_value_heads != num_attention_heads
    vocab_divisibility = _determine_vocab_divisibility_requirement(vocab_size, constraint_params)
    
    return ArchitectureConstraints(
        max_tensor_parallel=tp_constraints['max_size'],
        tensor_parallel_divisors=tp_constraints['valid_sizes'],
        max_expert_parallel=ep_constraints['max_size'],
        expert_parallel_divisors=ep_constraints['valid_sizes'],
        max_pipeline_parallel=pp_constraints['max_size'],
        min_layers_per_stage=pp_constraints['min_layers_per_stage'],
        requires_tied_embeddings=tied_embeddings,
        supports_grouped_query_attention=gqa_support,
        vocab_divisibility_requirement=vocab_divisibility
    )

def _analyze_tensor_parallel_constraints(
    hidden_size: int, 
    num_attention_heads: int, 
    num_key_value_heads: int,
    vocab_size: int,
    intermediate_size: int,
    constraint_params: ParallelismConstraintParameters
) -> Dict[str, any]:
    """Analyze tensor parallelism constraints based on model dimensions"""
    
    # Key constraint: attention heads must be divisible by TP size
    attention_head_constraint = num_attention_heads
    
    # For GQA models, KV heads are the limiting factor
    kv_head_constraint = num_key_value_heads
    
    # Hidden size should be divisible (for efficient sharding)
    hidden_size_divisors = _get_divisors(hidden_size)
    
    # Intermediate size constraint (for MLP sharding)
    intermediate_divisors = _get_divisors(intermediate_size)
    
    # Vocab size constraint (for embedding sharding)
    vocab_divisors = _get_efficient_divisors(vocab_size, max_divisor=constraint_params.default_max_tensor_parallel)
    
    # Find intersection of all constraints
    valid_tp_sizes = set(range(1, attention_head_constraint + 1))
    valid_tp_sizes &= set(range(1, kv_head_constraint + 1))
    valid_tp_sizes &= set(hidden_size_divisors)
    valid_tp_sizes &= set(intermediate_divisors)
    valid_tp_sizes &= set(vocab_divisors)
    
    # Practical maximum (configurable)
    practical_max = min(constraint_params.default_max_tensor_parallel, max(valid_tp_sizes) if valid_tp_sizes else 1)
    valid_tp_sizes = {size for size in valid_tp_sizes if size <= practical_max}
    
    return {
        'max_size': practical_max,
        'valid_sizes': valid_tp_sizes
    }

def _analyze_expert_parallel_constraints(
    config: transformers.PretrainedConfig,
    constraint_params: ParallelismConstraintParameters
) -> Dict[str, any]:
    """Analyze expert parallelism constraints for MoE models"""
    
    # Check if this is an MoE model
    num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', 0))
    
    if num_experts == 0:
        return {'max_size': 0, 'valid_sizes': {1}}
    
    # Expert parallelism: experts must be distributable across devices
    expert_divisors = _get_divisors(num_experts)
    
    # Practical constraint: configurable minimum experts per device
    max_ep_size = num_experts // constraint_params.min_experts_per_device
    
    valid_ep_sizes = {size for size in expert_divisors if size <= max_ep_size}
    
    return {
        'max_size': max_ep_size,
        'valid_sizes': valid_ep_sizes
    }

def _analyze_pipeline_parallel_constraints(
    num_layers: int,
    constraint_params: ParallelismConstraintParameters
) -> Dict[str, any]:
    """Analyze pipeline parallelism constraints"""
    
    # Pipeline parallelism: distribute layers across stages
    # Minimum layers per stage for efficiency (configurable)
    max_pp_size = num_layers // constraint_params.default_min_layers_per_stage
    
    return {
        'max_size': max_pp_size,
        'min_layers_per_stage': constraint_params.default_min_layers_per_stage
    }

def _get_divisors(n: int, max_divisor: int = None) -> List[int]:
    """Get all divisors of n up to max_divisor"""
    if max_divisor is None:
        max_divisor = n
    
    divisors = []
    for i in range(1, min(int(n**0.5) + 1, max_divisor + 1)):
        if n % i == 0:
            divisors.append(i)
            if i != n // i and n // i <= max_divisor:
                divisors.append(n // i)
    
    return sorted(divisors)

def _get_efficient_divisors(n: int, max_divisor: int = 64) -> List[int]:
    """Get divisors that are powers of 2 or have small prime factors (more efficient)"""
    all_divisors = _get_divisors(n, max_divisor)
    
    # Prefer powers of 2 and numbers with small prime factors
    efficient_divisors = []
    for d in all_divisors:
        if d <= max_divisor and _is_efficient_divisor(d):
            efficient_divisors.append(d)
    
    return efficient_divisors

def _is_efficient_divisor(n: int) -> bool:
    """Check if a number is an efficient divisor (power of 2 or has small prime factors)"""
    if n & (n - 1) == 0:  # Power of 2
        return True
    
    # Check if all prime factors are small (≤ 7)
    temp = n
    for prime in [2, 3, 5, 7]:
        while temp % prime == 0:
            temp //= prime
    
    return temp == 1

def _check_tied_embeddings(config: transformers.PretrainedConfig) -> bool:
    """Check if model has tied input/output embeddings"""
    return getattr(config, 'tie_word_embeddings', False)

def _determine_vocab_divisibility_requirement(
    vocab_size: int,
    constraint_params: ParallelismConstraintParameters
) -> int:
    """Determine vocabulary divisibility requirement for efficient sharding"""
    # For large vocabularies, require divisibility by larger numbers for efficiency
    if vocab_size >= constraint_params.vocab_large_threshold:
        return constraint_params.vocab_large_divisibility
    elif vocab_size >= constraint_params.vocab_medium_threshold:
        return constraint_params.vocab_medium_divisibility
    else:
        return constraint_params.vocab_small_divisibility
```

**Memory Components Tracked**:
```python
total_memory = model_weights + peak_activations + kv_cache + cuda_graphs + minimal_fragmentation_overhead
```

Where:
- `model_weights = sum(p.numel() * p.element_size() for p in model.parameters())`
- `peak_activations` calculated via meta tensor forward pass with memory hooks
- `kv_cache = max_seqs × context_len × num_layers × hidden_dim × 2 × precision_bytes`
- `cuda_graphs` estimated based on compilation level and operation count
- `minimal_fragmentation_overhead` as small percentage (5-15%)

### 3. Cluster Analyzer

**Purpose**: Characterize available GPU resources and network topology.

```python
@dataclass
class NetworkTopologyParameters:
    """Configurable network topology parameters"""
    
    # Default bandwidth assumptions (parameterizable)
    intra_node_bandwidth_gbps: float = 900.0  # NVSwitch default
    inter_node_bandwidth_gbps: float = 200.0  # RoCE RDMA default
    
    # Efficiency thresholds (configurable)
    intra_node_efficiency_threshold: float = 0.8  # Efficiency loss threshold for intra-node
    inter_node_efficiency_threshold: float = 0.6  # Efficiency loss threshold for inter-node
    
    # Communication patterns
    tensor_parallel_bandwidth_requirement: float = 0.8  # Fraction of peak bandwidth needed for TP
    pipeline_parallel_bandwidth_requirement: float = 0.3  # Fraction of peak bandwidth needed for PP

@dataclass 
class ClusterSpec:
    """GPU cluster specification"""
    
    gpu_memory_gb: float  # Standardized parameter name
    gpus_per_node: int
    num_nodes: int
    gpu_model: str = "H100"  # For architecture-specific optimizations
    network_params: NetworkTopologyParameters = None
    
    def __post_init__(self):
        if self.network_params is None:
            self.network_params = NetworkTopologyParameters()
    
    @property
    def total_gpus(self) -> int:
        return self.gpus_per_node * self.num_nodes
    
    @property 
    def total_memory_gb(self) -> float:
        return self.gpu_memory_gb * self.total_gpus
    
    def get_topology_constraints(self) -> Dict[str, int]:
        """Get topology-based constraints for parallelism"""
        return {
            'max_intra_node_parallel': self.gpus_per_node,
            'max_inter_node_parallel': self.num_nodes,
            'prefer_intra_node_up_to': self.gpus_per_node  # Prefer intra-node communication
        }
```

### 4. Configuration Generator

**Purpose**: Generate all valid parallelism configurations within cluster and architecture constraints.

```python
@dataclass
class ParallelismOptimizationParameters:
    """Configurable parameters for parallelism optimization"""
    
    # Memory utilization constraints
    min_gpu_memory_utilization: float = 0.80  # Conservative lower bound
    max_gpu_memory_utilization: float = 0.95  # Aggressive upper bound
    target_gpu_memory_utilization: float = 0.90  # Preferred target
    
    # Efficiency preferences
    prefer_tensor_parallel_within_node: bool = True
    prefer_pipeline_parallel_across_nodes: bool = True
    
    # Performance thresholds (configurable)
    min_effective_batch_size: int = 1  # Minimum viable batch size
    inter_node_tensor_parallel_penalty: float = 0.7  # Efficiency penalty for inter-node TP
    
    # Expert parallelism preferences (MoE)
    min_experts_per_device: int = 1  # Minimum experts per device for EP
    prefer_expert_parallel_over_data_parallel: bool = True  # For MoE models

class ConfigurationGenerator:
    """Generate valid parallelism configurations"""
    
    def __init__(
        self, 
        model_name: str,
        cluster_spec: ClusterSpec,
        optimization_params: ParallelismOptimizationParameters = None
    ):
        self.model_name = model_name
        self.cluster_spec = cluster_spec
        self.optimization_params = optimization_params or ParallelismOptimizationParameters()
        
        # Load model configuration and analyze constraints
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        self.arch_constraints = analyze_architecture_constraints(self.config)
        
        # Create memory estimator
        from spec.memory_estimation_framework import create_memory_estimator
        self.memory_estimator = create_memory_estimator(self.config)
    
    def generate_all_valid_configurations(
        self, 
        workload_config: Dict[str, int] = None
    ) -> List[Dict[str, int]]:
        """Generate all valid parallelism configurations"""
        
        if workload_config is None:
            workload_config = {
                'batch_size': 32,
                'sequence_length': 2048,
                'max_num_seqs': 256,
                'max_model_len': 4096
            }
        
        valid_configs = []
        
        # Get valid parallelism sizes from constraints
        max_gpus_per_node = self.cluster_spec.gpus_per_node
        max_nodes = self.cluster_spec.num_nodes
        
        valid_tp_sizes = self.arch_constraints.get_valid_tensor_parallel_sizes(max_gpus_per_node)
        valid_ep_sizes = self.arch_constraints.get_valid_expert_parallel_sizes(max_gpus_per_node)
        valid_pp_sizes = self.arch_constraints.get_valid_pipeline_parallel_sizes(max_nodes)
        
        # Generate all combinations
        for tp_size in valid_tp_sizes:
            for ep_size in valid_ep_sizes:
                for pp_size in valid_pp_sizes:
                    
                    # Calculate required GPUs for this parallelism combination
                    gpus_needed = tp_size * ep_size * pp_size
                    
                    if gpus_needed > self.cluster_spec.total_gpus:
                        continue
                    
                    # Calculate data parallelism
                    dp_size = self.cluster_spec.total_gpus // gpus_needed
                    
                    config = {
                        'tensor_parallel_size': tp_size,
                        'expert_parallel_size': ep_size,
                        'pipeline_parallel_size': pp_size,
                        'data_parallel_size': dp_size
                    }
                    
                    # Validate configuration
                    if self._is_valid_configuration(config, workload_config):
                        valid_configs.append(config)
        
        return valid_configs
    
    def _is_valid_configuration(
        self, 
        parallelism_config: Dict[str, int], 
        workload_config: Dict[str, int]
    ) -> bool:
        """Check if a parallelism configuration is valid"""
        
        # Check GPU allocation
        total_gpus_used = (
            parallelism_config['tensor_parallel_size'] *
            parallelism_config['expert_parallel_size'] * 
            parallelism_config['pipeline_parallel_size'] *
            parallelism_config['data_parallel_size']
        )
        
        if total_gpus_used != self.cluster_spec.total_gpus:
            return False
        
        # Check memory constraints
        memory_breakdown = self.memory_estimator.estimate_total_memory(
            workload_config=workload_config,
            parallelism_config=parallelism_config
        )
        
        memory_per_gpu = memory_breakdown.total_gb
        
        if memory_per_gpu > self.cluster_spec.gpu_memory_gb * self.optimization_params.max_gpu_memory_utilization:
            return False
        
        if memory_per_gpu < self.cluster_spec.gpu_memory_gb * self.optimization_params.min_gpu_memory_utilization:
            return False  # Underutilizing memory
        
        # Check topology constraints
        if not self._respects_topology_constraints(parallelism_config):
            return False
        
        return True
    
    def _respects_topology_constraints(self, parallelism_config: Dict[str, int]) -> bool:
        """Check if configuration respects cluster topology"""
        
        tp_size = parallelism_config['tensor_parallel_size']
        pp_size = parallelism_config['pipeline_parallel_size']
        
        # Tensor parallelism should prefer intra-node placement
        if tp_size > self.cluster_spec.gpus_per_node:
            # Inter-node tensor parallelism - check if it's worth the penalty
            if not self._is_inter_node_tp_justified(parallelism_config):
                return False
        
        # Pipeline parallelism should align with node boundaries when possible
        if pp_size > self.cluster_spec.num_nodes:
            return False  # Can't have more pipeline stages than nodes
        
        return True
    
    def _is_inter_node_tp_justified(self, parallelism_config: Dict[str, int]) -> bool:
        """Check if inter-node tensor parallelism is justified"""
        
        # This is a heuristic - inter-node TP is rarely optimal
        # Only allow if model is very large and no other options exist
        tp_size = parallelism_config['tensor_parallel_size']
        
        # Allow inter-node TP only for very large models that can't fit otherwise
        try:
            # Try intra-node alternative
            intra_node_config = parallelism_config.copy()
            intra_node_config['tensor_parallel_size'] = self.cluster_spec.gpus_per_node
            intra_node_config['pipeline_parallel_size'] = max(
                1, tp_size // self.cluster_spec.gpus_per_node
            )
            
            # Recalculate data parallelism
            gpus_needed = (
                intra_node_config['tensor_parallel_size'] *
                intra_node_config['expert_parallel_size'] *
                intra_node_config['pipeline_parallel_size']
            )
            
            if gpus_needed <= self.cluster_spec.total_gpus:
                return False  # Intra-node alternative exists
            
        except:
            pass
        
        return True  # No better alternative found
```

## Memory Calculation Integration

**Note**: Memory calculations now delegate to the unified memory estimation framework defined in `spec/memory_estimation_framework.md`. This eliminates magic constants and provides precise, configurable memory estimates.

```python
# Example usage with memory estimation framework
def calculate_memory_requirements(
    model_name: str,
    parallelism_config: Dict[str, int],
    workload_config: Dict[str, int],
    framework: str = "vllm"
) -> MemoryBreakdown:
    """Calculate memory requirements using unified framework"""
    
    from spec.memory_estimation_framework import create_memory_estimator, vLLMMemoryEstimator
    
    config = transformers.AutoConfig.from_pretrained(model_name)
    
    if framework == "vllm":
        estimator = vLLMMemoryEstimator(config)
        # Framework-specific config can include CUDA graph parameters
        framework_config = {
            'cudagraph_capture_sizes': workload_config.get('cudagraph_capture_sizes', []),
            'graph_memory_overhead_ratio': 0.1,  # Configurable
            'graph_batch_scaling_factor': 0.02   # Configurable
        }
    else:
        estimator = create_memory_estimator(config)
        framework_config = {}
    
    return estimator.estimate_total_memory(
        workload_config=workload_config,
        parallelism_config=parallelism_config,
        framework_config=framework_config
    )
```

### 4. MoE-Specific Handling

**Expert Parallelism Memory Model**:
```python
# Expert memory distributed across EP replicas
expert_memory_per_gpu = (total_experts * expert_size) / ep_size

# Dense layer memory still replicated across DP
dense_memory_per_gpu = dense_params * (1 / tp_size)

# Critical insight: Expert sharing between DP replicas
# Experts can be shared across DP replicas via all-to-all communication
# Unlike dense layers which must be fully replicated for each DP replica
total_expert_memory_cluster = total_experts * expert_size  # NOT multiplied by DP
total_dense_memory_cluster = dense_params * dp_size
```

**MoE Configuration Constraints**:
```python
# Expert parallelism typically within node for routing efficiency
ep_size <= gpus_per_node

# Combined constraints for MoE
tp_size * ep_size * pp_size * dp_size <= total_gpus
```

## Model Analysis Interface

### Transformers Model Interface

```python
def get_model_analysis_interface(model_name_or_path):
    """Get interface for model analysis using memory estimation framework"""
    from spec.memory_estimation_framework import create_memory_estimator
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name_or_path)
    estimator = create_memory_estimator(config)
    
    return {
        'config': config,
        'memory_estimator': estimator,
        'moe_info': estimator.get_moe_info() if hasattr(estimator, 'get_moe_info') else {'is_moe': False},
        'quantization_info': estimator.get_quantization_info() if hasattr(estimator, 'get_quantization_info') else None
    }
```



## Parallelism Strategy Selection

Parallelism strategy selection leverages the unified memory estimation framework and architectural constraints to determine optimal configurations. The selection algorithm:

1. **Constraint Analysis**: Use `analyze_architecture_constraints()` with configurable parameters
2. **Memory Estimation**: Delegate to memory estimation framework for precise calculations  
3. **Strategy Selection**: Apply topology-aware selection using configurable efficiency thresholds
4. **Validation**: Ensure all configurations respect architectural and hardware constraints

## Configuration Optimization Goals

### 1. Minimum Cluster Size Determination
- Calculate smallest cluster that can fit model without pipeline parallelism
- Prioritize intra-node tensor parallelism for efficiency
- Respect architecture constraints (attention heads, expert divisibility, etc.)

### 2. Throughput Optimization  
- Maximize data parallelism within memory constraints
- Leverage expert sharing for MoE models to reduce memory replication
- Balance computation efficiency vs communication overhead

### 3. Latency Optimization
- Minimize pipeline stages to reduce pipeline bubbles
- Prefer tensor parallelism over pipeline parallelism when memory allows
- Consider disaggregated prefill for interactive workloads

## Key Design Principles

1. **Transformers-Native Integration**: Leverage Hugging Face transformers config system, model architectures, and quantization formats for precise analysis
2. **Architecture Constraint Awareness**: Respect model divisibility constraints (attention heads, experts, vocabulary size) that limit valid parallelism configurations  
3. **Precision over Approximation**: Use meta device analysis with transformers models for exact calculations rather than rough estimates
4. **Quantization Awareness**: Handle diverse precision formats (GPTQ, AWQ, native bf16/fp8) with proper memory calculations
5. **MoE Specialization**: Detect and optimize for MoE architectures (Mixtral, Switch Transformers, DeepSeek) with expert sharing
6. **Memory Estimation Framework Integration**: Delegate memory calculations to unified framework for configurability and accuracy
7. **Topology Awareness**: Respect network topology constraints (intra-node vs inter-node efficiency)
8. **Configurable Parameters**: Parameterize all behavior through dedicated configuration classes
9. **Comprehensive Constraint Validation**: Ensure all generated configurations respect both hardware and architectural limits

## Memory Calculation Integration

**Note**: All memory calculations are delegated to the unified memory estimation framework defined in `spec/memory_estimation_framework.md`. This provides precise, configurable memory estimates and eliminates magic constants.

This architecture provides a foundation for building a precise, transformers-native autoparallel tool that can guide optimal cluster configuration decisions for modern LLM deployments across inference and training frameworks while respecting all architectural constraints that limit parallelization options.
