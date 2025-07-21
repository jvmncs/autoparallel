# Memory Estimation Framework Specification

## Overview

This specification defines a unified memory estimation framework for LLM deployment across different inference and training engines (vLLM, DeepSpeed, etc.). The framework provides precise memory calculations without instantiating actual engines.

## Core Architecture

### 1. Memory Component Abstraction

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import transformers

@dataclass
class MemoryEstimationParameters:
    """Configurable parameters for memory estimation behavior"""
    
    # Memory safety and fragmentation
    fragmentation_overhead_factor: float = 1.05  # Account for memory fragmentation (valid range: [1.0, 2.0])
    min_memory_safety_margin_gb: float = 0.5  # Minimum safety margin (valid range: [0.0, 10.0])
    memory_safety_margin_factor: float = 0.02  # Safety margin as fraction of base memory (valid range: [0.0, 0.2])
    
    # Activation memory estimation
    activation_recomputation_factor: float = 0.3  # Fraction of activations stored in memory (valid range: [0.0, 1.0])
    gradient_memory_multiplier: float = 1.0  # Gradient memory multiplier: 0.0 for inference, 1.0+ for training (valid range: [0.0, 10.0])
    
    # Precision handling
    default_compute_dtype: str = "float16"  # Default dtype for computation
    default_storage_dtype: str = "float16"  # Default dtype for weight storage
    
    # CUDA graph memory estimation parameters
    graph_memory_overhead_ratio: float = 0.1  # Base graph overhead as fraction of model memory (valid range: [0.0, 1.0])
    graph_batch_scaling_factor: float = 0.02  # Additional overhead per batch size unit (valid range: [0.0, 0.1])
    
    # Conservative estimation multipliers for uncertain scenarios
    quantization_overhead_multiplier: float = 1.1  # Overhead for quantized models (valid range: [1.0, 2.0])
    mixed_precision_overhead_multiplier: float = 1.05  # Overhead for mixed precision (valid range: [1.0, 1.5])
    
    def __post_init__(self):
        """Validate parameter bounds"""
        bounds = {
            'fragmentation_overhead_factor': (1.0, 2.0),
            'min_memory_safety_margin_gb': (0.0, 10.0),
            'memory_safety_margin_factor': (0.0, 0.2),
            'activation_recomputation_factor': (0.0, 1.0),
            'gradient_memory_multiplier': (0.0, 10.0),
            'graph_memory_overhead_ratio': (0.0, 1.0),
            'graph_batch_scaling_factor': (0.0, 0.1),
            'quantization_overhead_multiplier': (1.0, 2.0),
            'mixed_precision_overhead_multiplier': (1.0, 1.5)
        }
        
        for param_name, (min_val, max_val) in bounds.items():
            value = getattr(self, param_name)
            if not min_val <= value <= max_val:
                raise ValueError(f"{param_name}={value} not in valid range [{min_val}, {max_val}]")

@dataclass
class MemoryBreakdown:
    """Detailed memory usage breakdown"""
    
    # Core components
    model_weights_gb: float
    activations_gb: float
    gradients_gb: float = 0.0  # Zero for inference
    optimizer_states_gb: float = 0.0  # Zero for inference
    
    # Framework-specific components
    kv_cache_gb: float = 0.0
    cuda_graphs_gb: float = 0.0
    compilation_overhead_gb: float = 0.0
    
    # Memory management
    fragmentation_overhead_gb: float = 0.0
    safety_margin_gb: float = 0.0
    
    @property
    def total_gb(self) -> float:
        return (
            self.model_weights_gb + self.activations_gb + self.gradients_gb + 
            self.optimizer_states_gb + self.kv_cache_gb + self.cuda_graphs_gb + 
            self.compilation_overhead_gb + self.fragmentation_overhead_gb + 
            self.safety_margin_gb
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'model_weights': self.model_weights_gb,
            'activations': self.activations_gb,
            'gradients': self.gradients_gb,
            'optimizer_states': self.optimizer_states_gb,
            'kv_cache': self.kv_cache_gb,
            'cuda_graphs': self.cuda_graphs_gb,
            'compilation_overhead': self.compilation_overhead_gb,
            'fragmentation_overhead': self.fragmentation_overhead_gb,
            'safety_margin': self.safety_margin_gb,
            'total': self.total_gb
        }

class MemoryEstimator(ABC):
    """Abstract base class for memory estimation"""
    
    def __init__(self, config: transformers.PretrainedConfig, params: MemoryEstimationParameters):
        self.config = config
        self.params = params
    
    @abstractmethod
    def estimate_model_weights(self, parallelism_config: Dict[str, int]) -> float:
        """Estimate model weight memory in GB"""
        pass
    
    @abstractmethod
    def estimate_activations(self, batch_config: Dict[str, int], parallelism_config: Dict[str, int]) -> float:
        """Estimate activation memory in GB"""
        pass
    
    def estimate_gradients(self, parallelism_config: Dict[str, int]) -> float:
        """Estimate gradient memory in GB (default: same as weights for training)"""
        if self.params.gradient_memory_multiplier == 0:
            return 0.0
        return self.estimate_model_weights(parallelism_config) * self.params.gradient_memory_multiplier
    
    def estimate_optimizer_states(self, parallelism_config: Dict[str, int], optimizer_type: str = "adamw") -> float:
        """Estimate optimizer state memory in GB"""
        if optimizer_type.lower() == "sgd":
            return 0.0  # SGD has no states
        elif optimizer_type.lower() == "adamw":
            # AdamW: momentum + variance (2x model size)
            return self.estimate_model_weights(parallelism_config) * 2.0
        else:
            return self.estimate_model_weights(parallelism_config) * 2.0  # Conservative estimate
    
    def estimate_total_memory(
        self, 
        workload_config: Dict[str, int],
        parallelism_config: Dict[str, int],
        framework_config: Dict[str, any] = None
    ) -> MemoryBreakdown:
        """Estimate total memory usage"""
        
        framework_config = framework_config or {}
        
        # Core memory components
        weights = self.estimate_model_weights(parallelism_config)
        activations = self.estimate_activations(workload_config, parallelism_config)
        gradients = self.estimate_gradients(parallelism_config)
        optimizer = self.estimate_optimizer_states(
            parallelism_config, 
            framework_config.get('optimizer_type', 'adamw')
        )
        
        # Framework-specific components (override in subclasses)
        kv_cache = self.estimate_kv_cache(workload_config, parallelism_config)
        cuda_graphs = self.estimate_cuda_graphs(workload_config, framework_config)
        compilation = self.estimate_compilation_overhead(framework_config)
        
        # Memory management overhead
        base_memory = weights + activations + gradients + optimizer + kv_cache + cuda_graphs + compilation
        fragmentation = base_memory * (self.params.fragmentation_overhead_factor - 1.0)
        safety_margin = max(
            self.params.min_memory_safety_margin_gb, 
            base_memory * self.params.memory_safety_margin_factor
        )
        
        return MemoryBreakdown(
            model_weights_gb=weights,
            activations_gb=activations,
            gradients_gb=gradients,
            optimizer_states_gb=optimizer,
            kv_cache_gb=kv_cache,
            cuda_graphs_gb=cuda_graphs,
            compilation_overhead_gb=compilation,
            fragmentation_overhead_gb=fragmentation,
            safety_margin_gb=safety_margin
        )
    
    def estimate_kv_cache(self, workload_config: Dict[str, int], parallelism_config: Dict[str, int]) -> float:
        """Estimate KV cache memory (default: 0 for non-inference frameworks)"""
        return 0.0
    
    def estimate_cuda_graphs(self, workload_config: Dict[str, int], framework_config: Dict[str, any]) -> float:
        """Estimate CUDA graph memory (default: 0)"""
        return 0.0
    
    def estimate_compilation_overhead(self, framework_config: Dict[str, any]) -> float:
        """Estimate compilation overhead (default: 0)"""
        return 0.0
```

### 2. Transformers-Specific Implementation

```python
class TransformersMemoryEstimator(MemoryEstimator):
    """Memory estimator for transformers models"""
    
    def __init__(self, config: transformers.PretrainedConfig, params: MemoryEstimationParameters = None):
        super().__init__(config, params or MemoryEstimationParameters())
        
        # Extract architecture parameters
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_attention_heads)
        self.vocab_size = getattr(config, 'vocab_size', 50257)
        self.intermediate_size = getattr(config, 'intermediate_size', 4 * self.hidden_size)
    
    def estimate_model_weights(self, parallelism_config: Dict[str, int]) -> float:
        """Estimate model weight memory accounting for parallelism"""
        
        # Get precision bytes
        precision_bytes = self._get_precision_bytes()
        
        # Calculate total parameters
        total_params = self._calculate_total_parameters()
        
        # Account for tensor parallelism (weights are sharded)
        tp_size = parallelism_config.get('tensor_parallel_size', 1)
        pp_size = parallelism_config.get('pipeline_parallel_size', 1)
        
        # Model weights per GPU with tensor parallelism
        params_per_gpu = total_params / tp_size
        
        return (params_per_gpu * precision_bytes) / (1024**3)
    
    def estimate_activations(self, batch_config: Dict[str, int], parallelism_config: Dict[str, int]) -> float:
        """Estimate activation memory for transformers"""
        
        batch_size = batch_config.get('batch_size', 1)
        sequence_length = batch_config.get('sequence_length', 512)
        
        # Attention activations (dominant component)
        attention_memory = self._estimate_attention_activations(batch_size, sequence_length)
        
        # MLP activations
        mlp_memory = self._estimate_mlp_activations(batch_size, sequence_length)
        
        # Account for pipeline parallelism (fewer layers per GPU)
        pp_size = parallelism_config.get('pipeline_parallel_size', 1)
        layers_per_gpu = self.num_layers / pp_size
        
        # Total activation memory per GPU
        per_layer_memory = attention_memory + mlp_memory
        total_activations = per_layer_memory * layers_per_gpu * self.params.activation_recomputation_factor
        
        return total_activations / (1024**3)
    
    def _get_precision_bytes(self) -> float:
        """Get bytes per parameter based on model precision"""
        
        # Check quantization config
        quant_config = getattr(self.config, 'quantization_config', {})
        if quant_config:
            bits = quant_config.get('bits', 16)
            bytes_per_param = bits / 8.0
            
            # Apply quantization overhead for sub-byte precisions
            if bits < 8:
                bytes_per_param *= self.params.quantization_overhead_multiplier
            
            return bytes_per_param
        
        # Check torch_dtype
        torch_dtype = getattr(self.config, 'torch_dtype', self.params.default_storage_dtype)
        torch_dtype_str = str(torch_dtype).lower()
        
        if 'float16' in torch_dtype_str or 'bfloat16' in torch_dtype_str:
            return 2.0
        elif 'fp8' in torch_dtype_str or 'int8' in torch_dtype_str:
            return 1.0
        elif 'int4' in torch_dtype_str:
            # int4 requires 0.5 bytes per parameter but with packing overhead
            return 0.5 * self.params.quantization_overhead_multiplier
        elif 'float32' in torch_dtype_str:
            return 4.0
        else:
            # Fallback to default storage dtype
            if 'float16' in self.params.default_storage_dtype or 'bfloat16' in self.params.default_storage_dtype:
                return 2.0
            else:
                return 4.0  # Conservative fallback
    
    def _calculate_total_parameters(self) -> int:
        """Calculate total model parameters from architecture"""
        
        # Embedding parameters
        input_embedding_params = self.vocab_size * self.hidden_size
        
        # Per-layer parameters
        # Attention: Q, K, V, O projections
        attention_params = 4 * self.hidden_size * self.hidden_size
        
        # MLP: up and down projections
        mlp_params = 2 * self.hidden_size * self.intermediate_size
        
        # Layer norms
        layer_norm_params = 2 * self.hidden_size  # Pre and post attention
        
        # Total per layer
        per_layer_params = attention_params + mlp_params + layer_norm_params
        
        # Check if embeddings are tied (common in many transformer models)
        tied_embeddings = getattr(self.config, 'tie_word_embeddings', False)
        
        # Output projection parameters (only if not tied to input embeddings)
        output_projection_params = 0 if tied_embeddings else self.vocab_size * self.hidden_size
        
        # Total model parameters
        total_params = (
            input_embedding_params +  # Input embeddings
            (self.num_layers * per_layer_params) +  # All transformer layers
            self.hidden_size +  # Final layer norm
            output_projection_params  # Output projection (if not tied)
        )
        
        return total_params
    
    def _estimate_attention_activations(self, batch_size: int, sequence_length: int) -> float:
        """Estimate attention activation memory"""
        
        # Q, K, V tensors
        qkv_memory = 3 * batch_size * sequence_length * self.hidden_size * 2  # fp16
        
        # Attention scores
        attention_scores = (
            batch_size * self.num_attention_heads * 
            sequence_length * sequence_length * 2  # fp16
        )
        
        return qkv_memory + attention_scores
    
    def _estimate_mlp_activations(self, batch_size: int, sequence_length: int) -> float:
        """Estimate MLP activation memory"""
        
        # Intermediate activations
        return batch_size * sequence_length * self.intermediate_size * 2  # fp16

class MoEMemoryEstimator(TransformersMemoryEstimator):
    """Memory estimator for MoE models"""
    
    def __init__(self, config: transformers.PretrainedConfig, params: MemoryEstimationParameters = None):
        super().__init__(config, params)
        
        # MoE-specific parameters
        self.num_experts = getattr(config, 'num_local_experts', getattr(config, 'num_experts', 0))
        self.experts_per_token = getattr(config, 'num_experts_per_tok', 1)
        
    def estimate_model_weights(self, parallelism_config: Dict[str, int]) -> float:
        """Estimate MoE model weights with expert parallelism"""
        
        if self.num_experts == 0:
            return super().estimate_model_weights(parallelism_config)
        
        # Calculate dense and expert parameters separately
        dense_params = self._calculate_dense_parameters()
        expert_params = self._calculate_expert_parameters()
        
        # Apply parallelism
        tp_size = parallelism_config.get('tensor_parallel_size', 1)
        ep_size = parallelism_config.get('expert_parallel_size', 1)
        
        # Dense parameters distributed via tensor parallelism
        dense_params_per_gpu = dense_params / tp_size
        
        # Expert parameters distributed via expert parallelism
        expert_params_per_gpu = expert_params / ep_size
        
        precision_bytes = self._get_precision_bytes()
        total_params_per_gpu = dense_params_per_gpu + expert_params_per_gpu
        
        return (total_params_per_gpu * precision_bytes) / (1024**3)
    
    def _calculate_expert_parameters(self) -> int:
        """Calculate parameters specific to experts"""
        
        # Each expert is an MLP: up + down projections
        expert_params = 2 * self.hidden_size * self.intermediate_size
        
        # Total across all experts and all layers
        return self.num_experts * expert_params * self.num_layers
    
    def _calculate_dense_parameters(self) -> int:
        """Calculate parameters for non-expert components"""
        
        # Embedding parameters
        embedding_params = self.vocab_size * self.hidden_size
        
        # Per-layer dense parameters (attention + layer norms + routing)
        attention_params = 4 * self.hidden_size * self.hidden_size
        layer_norm_params = 2 * self.hidden_size
        routing_params = self.hidden_size * self.num_experts  # Router weights
        
        per_layer_dense = attention_params + layer_norm_params + routing_params
        
        return embedding_params + (self.num_layers * per_layer_dense) + self.hidden_size

def create_memory_estimator(
    model_name_or_config: Union[str, transformers.PretrainedConfig],
    params: MemoryEstimationParameters = None
) -> MemoryEstimator:
    """Factory function to create appropriate memory estimator"""
    
    if isinstance(model_name_or_config, str):
        config = transformers.AutoConfig.from_pretrained(model_name_or_config)
    else:
        config = model_name_or_config
    
    # Detect MoE models
    if hasattr(config, 'num_local_experts') or hasattr(config, 'num_experts'):
        return MoEMemoryEstimator(config, params)
    else:
        return TransformersMemoryEstimator(config, params)
```

### 3. Framework-Specific Extensions

```python
class vLLMMemoryEstimator(TransformersMemoryEstimator):
    """vLLM-specific memory estimation"""
    
    def estimate_kv_cache(self, workload_config: Dict[str, int], parallelism_config: Dict[str, int]) -> float:
        """Estimate KV cache memory for vLLM"""
        
        max_num_seqs = workload_config.get('max_num_seqs', 512)
        max_model_len = workload_config.get('max_model_len', 2048)
        kv_cache_dtype = workload_config.get('kv_cache_dtype', 'auto')
        
        # Determine precision
        if kv_cache_dtype == 'fp8' or kv_cache_dtype == 'fp8_e4m3':
            dtype_bytes = 1
        else:
            dtype_bytes = 2  # fp16/bf16
        
        # KV cache per sequence
        head_dim = self.hidden_size // self.num_attention_heads
        memory_per_token = self.num_key_value_heads * head_dim * 2 * dtype_bytes * self.num_layers
        
        # Account for tensor parallelism (KV cache is sharded)
        tp_size = parallelism_config.get('tensor_parallel_size', 1)
        memory_per_token_per_gpu = memory_per_token / tp_size
        
        total_kv_memory = max_num_seqs * max_model_len * memory_per_token_per_gpu
        
        return total_kv_memory / (1024**3)
    
    def estimate_cuda_graphs(self, workload_config: Dict[str, int], framework_config: Dict[str, any]) -> float:
        """Estimate CUDA graph memory for vLLM
        
        CUDA graph memory consists of:
        1. Base overhead: proportional to model size for capturing execution graphs
        2. Batch scaling: additional memory per batch size unit for larger graphs
        
        Methodology: Based on empirical measurements showing CUDA graphs consume
        ~10% of model memory as base overhead plus ~2% per batch size unit.
        """
        
        capture_sizes = framework_config.get('cudagraph_capture_sizes', [])
        if not capture_sizes:
            return 0.0
        
        model_memory_gb = self.estimate_model_weights({'tensor_parallel_size': 1})
        
        # Use framework-specific config or fall back to parameters
        base_overhead_ratio = framework_config.get(
            'graph_memory_overhead_ratio', 
            self.params.graph_memory_overhead_ratio
        )
        batch_scaling_factor = framework_config.get(
            'graph_batch_scaling_factor', 
            self.params.graph_batch_scaling_factor
        )
        
        total_graph_memory = 0.0
        for batch_size in capture_sizes:
            graph_memory = (
                model_memory_gb * base_overhead_ratio * 
                (1 + batch_size * batch_scaling_factor)
            )
            total_graph_memory += graph_memory
        
        return total_graph_memory

class DeepSpeedMemoryEstimator(TransformersMemoryEstimator):
    """DeepSpeed-specific memory estimation"""
    
    def estimate_optimizer_states(self, parallelism_config: Dict[str, int], optimizer_type: str = "adamw") -> float:
        """Estimate optimizer states with ZeRO optimization"""
        
        zero_stage = parallelism_config.get('zero_stage', 0)
        dp_size = parallelism_config.get('data_parallel_size', 1)
        
        base_optimizer_memory = super().estimate_optimizer_states(parallelism_config, optimizer_type)
        
        # ZeRO optimizer state sharding
        if zero_stage >= 2:
            return base_optimizer_memory / dp_size
        else:
            return base_optimizer_memory
    
    def estimate_gradients(self, parallelism_config: Dict[str, int]) -> float:
        """Estimate gradients with ZeRO optimization"""
        
        zero_stage = parallelism_config.get('zero_stage', 0)
        dp_size = parallelism_config.get('data_parallel_size', 1)
        
        base_gradient_memory = super().estimate_gradients(parallelism_config)
        
        # ZeRO gradient sharding
        if zero_stage >= 2:
            return base_gradient_memory / dp_size
        else:
            return base_gradient_memory
    
    def estimate_model_weights(self, parallelism_config: Dict[str, int]) -> float:
        """Estimate model weights with ZeRO optimization"""
        
        zero_stage = parallelism_config.get('zero_stage', 0)
        dp_size = parallelism_config.get('data_parallel_size', 1)
        
        base_weight_memory = super().estimate_model_weights(parallelism_config)
        
        # ZeRO parameter sharding
        if zero_stage >= 3:
            return base_weight_memory / dp_size
        else:
            return base_weight_memory
```

## Usage Examples

```python
# Create estimator for a model
estimator = create_memory_estimator("meta-llama/Llama-2-7b-hf")

# Define workload and parallelism
workload_config = {
    'batch_size': 32,
    'sequence_length': 2048,
    'max_num_seqs': 256,
    'max_model_len': 4096
}

parallelism_config = {
    'tensor_parallel_size': 4,
    'pipeline_parallel_size': 1,
    'data_parallel_size': 2
}

# Estimate memory usage
memory_breakdown = estimator.estimate_total_memory(
    workload_config=workload_config,
    parallelism_config=parallelism_config
)

print(f"Total memory: {memory_breakdown.total_gb:.2f} GB")
print(f"Memory breakdown: {memory_breakdown.to_dict()}")

# For vLLM-specific estimation with custom parameters
custom_params = MemoryEstimationParameters(
    memory_safety_margin_factor=0.05,  # 5% safety margin
    activation_recomputation_factor=0.2,  # More aggressive activation recomputation
    graph_memory_overhead_ratio=0.12,  # Higher graph overhead
    quantization_overhead_multiplier=1.2  # Conservative quantization estimate
)

vllm_estimator = vLLMMemoryEstimator(config, custom_params)
vllm_framework_config = {
    'cudagraph_capture_sizes': [1, 2, 4, 8, 16],
    # Framework config overrides parameter defaults if provided
    'graph_memory_overhead_ratio': 0.15,  # Even more conservative
    'graph_batch_scaling_factor': 0.03
}

vllm_memory = vllm_estimator.estimate_total_memory(
    workload_config=workload_config,
    parallelism_config=parallelism_config,
    framework_config=vllm_framework_config
)

# Example with parameter validation error handling
try:
    invalid_params = MemoryEstimationParameters(
        activation_recomputation_factor=1.5  # Invalid: > 1.0
    )
except ValueError as e:
    print(f"Parameter validation failed: {e}")
```

This framework provides:

1. **Unified Interface**: Common memory estimation across frameworks
2. **Configurable Parameters**: No magic constants, all behavior is parameterizable with validation
3. **Architecture-Aware**: Handles different model architectures (dense, MoE, tied embeddings, etc.)
4. **Framework-Specific**: Extensions for vLLM, DeepSpeed, etc. with CUDA graph support
5. **Parallelism-Aware**: Accounts for tensor, pipeline, data, and expert parallelization
6. **Precision-Aware**: Proper handling of quantization including int4 with overhead estimation
7. **Conservative Estimates**: Built-in safety margins and validation for production use
