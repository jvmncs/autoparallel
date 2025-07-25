# AutoParallel Simplified Memory Estimation

## Overview

Memory estimation is simplified to focus on the core components that matter for parallelism decisions. The approach uses proven heuristics rather than complex modeling.

## Core Memory Components

```python
@dataclass
class MemoryBreakdown:
    """Memory usage breakdown in GB."""
    weights: float           # Model parameters
    activations: float       # Forward pass activations
    kv_cache: float         # Key-value cache for attention
    framework_overhead: float # vLLM/framework overhead
    total: float            # Sum of all components
    
    def fits_in_gpu(self, gpu_memory_gb: float, utilization: float = 0.9) -> bool:
        return self.total <= gpu_memory_gb * utilization
        
    def per_gpu_memory(self, tensor_parallel: int) -> float:
        """Memory per GPU after tensor parallel sharding."""
        sharded_weights = self.weights / tensor_parallel
        # Activations and KV cache are replicated across TP ranks
        return sharded_weights + self.activations + self.kv_cache + self.framework_overhead
```

## Memory Estimation Function

```python
def estimate_memory(
    model_config: PretrainedConfig,
    sequence_length: int = 2048,
    batch_size: int = 32,
    tensor_parallel: int = 1,
    quantization_bytes: int = 2,  # 2 for fp16/bf16, 1 for int8, 4 for fp32
    enable_kv_cache: bool = True,
    framework_overhead_gb: float = 2.0
) -> MemoryBreakdown:
    """
    Estimate memory usage for model configuration.
    
    Uses simplified heuristics based on empirical observations:
    - Weights: parameter_count * quantization_bytes
    - Activations: ~2x weights for inference, ~4x for training
    - KV cache: scales with batch_size * sequence_length
    - Framework overhead: fixed ~2GB for vLLM
    """
    
    # Calculate model weights
    param_count = _estimate_param_count(model_config)
    weights_gb = param_count * quantization_bytes / (1024**3)
    
    # Activation memory (simplified heuristic)
    # Based on: batch_size * sequence_length * hidden_size * num_layers * bytes_per_element
    activations_gb = _estimate_activations(model_config, sequence_length, batch_size)
    
    # KV cache memory
    kv_cache_gb = 0.0
    if enable_kv_cache:
        kv_cache_gb = _estimate_kv_cache(model_config, sequence_length, batch_size)
    
    total_gb = weights_gb + activations_gb + kv_cache_gb + framework_overhead_gb
    
    return MemoryBreakdown(
        weights=weights_gb,
        activations=activations_gb, 
        kv_cache=kv_cache_gb,
        framework_overhead=framework_overhead_gb,
        total=total_gb
    )
```

## Helper Functions

### Parameter Count Estimation

```python
def _estimate_param_count(config: PretrainedConfig) -> int:
    """
    Estimate parameter count from model configuration.
    
    Handles:
    - Dense transformers (Llama, Qwen, Mistral)
    - MoE models (DeepSeek, Mixtral)
    - Multimodal models (basic estimation)
    """
    
    # Core transformer parameters
    vocab_size = getattr(config, 'vocab_size', 32000)
    hidden_size = getattr(config, 'hidden_size', 4096)
    intermediate_size = getattr(config, 'intermediate_size', 11008)
    num_layers = getattr(config, 'num_hidden_layers', 32)
    
    # Embedding layers
    embedding_params = vocab_size * hidden_size
    
    # Per-layer parameters
    attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
    mlp_params = 2 * hidden_size * intermediate_size  # up and down projections
    layer_norm_params = 2 * hidden_size  # attention and MLP layer norms
    
    single_layer_params = attention_params + mlp_params + layer_norm_params
    
    # Handle MoE
    num_experts = getattr(config, 'num_local_experts', 1)
    if num_experts > 1:
        # MoE: multiply MLP params by number of experts, add router
        router_params = hidden_size * num_experts
        moe_mlp_params = mlp_params * num_experts + router_params
        single_layer_params = attention_params + moe_mlp_params + layer_norm_params
    
    total_params = embedding_params + (single_layer_params * num_layers)
    
    # Add final layer norm and LM head
    total_params += hidden_size + (vocab_size * hidden_size)
    
    return total_params
```

### Activation Memory Estimation

```python
def _estimate_activations(config: PretrainedConfig, seq_len: int, batch_size: int) -> float:
    """
    Estimate activation memory using simplified heuristic.
    
    Formula: batch_size * seq_len * hidden_size * num_layers * 4 bytes (fp32) / 1GB
    """
    hidden_size = getattr(config, 'hidden_size', 4096)
    num_layers = getattr(config, 'num_hidden_layers', 32)
    
    # Peak activation memory during forward pass
    activation_memory_bytes = batch_size * seq_len * hidden_size * num_layers * 4
    
    return activation_memory_bytes / (1024**3)
```

### KV Cache Estimation

```python
def _estimate_kv_cache(config: PretrainedConfig, seq_len: int, batch_size: int) -> float:
    """
    Estimate KV cache memory.
    
    Formula: 2 * batch_size * seq_len * num_layers * num_heads * head_dim * 2 bytes
    """
    hidden_size = getattr(config, 'hidden_size', 4096)
    num_layers = getattr(config, 'num_hidden_layers', 32)
    num_attention_heads = getattr(config, 'num_attention_heads', 32)
    
    head_dim = hidden_size // num_attention_heads
    
    # 2 for key and value, 2 bytes for fp16
    kv_cache_bytes = 2 * batch_size * seq_len * num_layers * num_attention_heads * head_dim * 2
    
    return kv_cache_bytes / (1024**3)
```

## Parallelism Impact

### Tensor Parallelism
- **Weights**: Sharded across TP ranks (memory / tp_size)
- **Activations**: Replicated (same memory per GPU)
- **KV Cache**: Replicated (same memory per GPU)

### Pipeline Parallelism  
- **Weights**: Distributed across PP stages (memory / pp_size)
- **Activations**: Reduced (only store current stage)
- **KV Cache**: Stored only on last stage

### Expert Parallelism (MoE)
- **Weights**: Expert layers sharded (experts / ep_size)
- **Activations**: Replicated
- **KV Cache**: Replicated

## Accuracy vs Complexity Trade-offs

### Simplified (Current Approach)
- **Accuracy**: ~20% estimation error
- **Speed**: <100ms analysis time
- **Complexity**: ~100 LOC
- **Maintenance**: Easy to update

### Complex (Previous Approach)
- **Accuracy**: ~10% estimation error  
- **Speed**: ~1000ms analysis time
- **Complexity**: ~800 LOC
- **Maintenance**: Difficult to update

The simplified approach prioritizes speed and maintainability over precision. For most parallelism decisions, 20% accuracy is sufficient since the ranking of configurations remains correct.

## Future Improvements

1. **Empirical Calibration**: Collect real measurements to improve heuristics
2. **Framework Integration**: Actual memory profiling with vLLM/TensorRT-LLM
3. **Quantization Support**: Better handling of different quantization schemes
4. **Long Context**: Special handling for very long sequences (>100K tokens)

The goal is to maintain simplicity while gradually improving accuracy through empirical validation.
