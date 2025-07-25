#!/usr/bin/env python3
"""
Memory Requirement Checking Examples

This script demonstrates how to use autoparallel's memory estimation capabilities
to analyze model memory requirements across different configurations, precision
formats, and sequence lengths.
"""

from autoparallel import check_memory_requirements
from autoparallel.memory import (
    estimate_memory,
    check_memory_feasibility,
    get_quantization_bytes,
    MemoryBreakdown,
)


class MockConfig:
    """Mock configuration class that works with autoparallel memory estimation.
    
    Note: There's a bug in autoparallel.memory.estimate_memory where it checks
    hasattr(model_config, "__getattribute__") to distinguish between config objects
    and dictionaries, but dictionaries also have __getattribute__. This causes
    dictionaries to be treated as config objects, using getattr instead of dict access.
    """
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def format_gb(bytes_value: int) -> str:
    """Format bytes as GB with 1 decimal place."""
    return f"{bytes_value / (1024**3):.1f}GB"


def print_memory_breakdown(breakdown: dict, title: str = "Memory Breakdown"):
    """Print formatted memory breakdown."""
    print(f"\n{title}:")
    print(f"  Weights:    {breakdown['weights_gb']:.1f}GB")
    print(f"  Activations:{breakdown['activations_gb']:.1f}GB")
    print(f"  KV Cache:   {breakdown['kv_cache_gb']:.1f}GB")
    print(f"  Framework:  {breakdown['framework_overhead_gb']:.1f}GB")
    print(f"  Total:      {sum(breakdown.values()):.1f}GB")


def example_1_basic_memory_check():
    """Example 1: Basic memory requirement checking for a 7B model."""
    print("=" * 60)
    print("Example 1: Basic Memory Requirements")
    print("=" * 60)
    
    try:
        # Check memory requirements for Llama 2 7B
        result = check_memory_requirements(
            model="meta-llama/Llama-2-7b-hf",
            sequence_length=2048,
            batch_size=1,
            quantization="fp16"
        )
        
        print(f"Model: Llama 2 7B")
        print(f"Total Memory Required: {result['total_memory_gb']:.1f}GB")
        
        print_memory_breakdown(result['breakdown'])
        
        # Show GPU compatibility
        print("\nGPU Compatibility:")
        gpus = result['single_gpu_requirements']['fits_in_common_gpus']
        for gpu, fits in gpus.items():
            status = "✓" if fits else "✗"
            print(f"  {status} {gpu}")
        
        # Architecture info
        arch = result['architecture_info']
        print(f"\nModel Architecture:")
        print(f"  Parameters: ~{arch['num_parameters_estimate']}")
        print(f"  Layers: {arch['num_layers']}")
        print(f"  Hidden Size: {arch['hidden_size']}")
        print(f"  Vocab Size: {arch['vocab_size']:,}")
        
    except Exception as e:
        print(f"Error checking Llama 2 7B: {e}")
        print("Using mock configuration instead...")
        
        # Fallback to mock config for demonstration
        mock_config = MockConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
        )
        
        memory = estimate_memory(
            model_config=mock_config,
            sequence_length=2048,
            batch_size=1,
            quantization_bytes=2,  # fp16
        )
        
        print(f"\nMock 7B Model Memory Requirements:")
        print(f"  Weights: {format_gb(memory.weights)}")
        print(f"  Activations: {format_gb(memory.activations)}")
        print(f"  KV Cache: {format_gb(memory.kv_cache)}")
        print(f"  Framework: {format_gb(memory.framework_overhead)}")
        print(f"  Total: {format_gb(memory.total)}")


def example_2_precision_comparison():
    """Example 2: Compare memory usage across different precision formats."""
    print("\n" + "=" * 60)
    print("Example 2: Precision Format Comparison")
    print("=" * 60)
    
    # 7B model configuration
    config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )
    
    precisions = ["fp32", "fp16", "bf16", "int8", "fp8"]
    
    print("Memory usage by precision (7B model, seq_len=2048):")
    print(f"{'Precision':<10} {'Bytes/Param':<12} {'Total Memory':<15} {'Fits A100-40GB'}")
    print("-" * 55)
    
    for precision in precisions:
        quant_bytes = get_quantization_bytes(precision)
        
        memory = estimate_memory(
            model_config=config,
            sequence_length=2048,
            batch_size=1,
            quantization_bytes=quant_bytes,
        )
        
        fits_a100_40gb = memory.fits_in_gpu(40.0)
        fits_status = "✓" if fits_a100_40gb else "✗"
        
        print(f"{precision:<10} {quant_bytes:<12} {format_gb(memory.total):<15} {fits_status}")


def example_3_sequence_length_impact():
    """Example 3: Show how sequence length affects memory usage."""
    print("\n" + "=" * 60)
    print("Example 3: Sequence Length Impact on Memory")
    print("=" * 60)
    
    config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )
    
    sequence_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    print("Memory scaling with sequence length (7B model, fp16):")
    print(f"{'Seq Length':<12} {'KV Cache':<12} {'Activations':<12} {'Total':<12}")
    print("-" * 50)
    
    for seq_len in sequence_lengths:
        memory = estimate_memory(
            model_config=config,
            sequence_length=seq_len,
            batch_size=1,
            quantization_bytes=2,  # fp16
        )
        
        print(f"{seq_len:<12} {format_gb(memory.kv_cache):<12} "
              f"{format_gb(memory.activations):<12} {format_gb(memory.total):<12}")


def example_4_find_minimum_gpu():
    """Example 4: Find minimum GPU requirements for different models."""
    print("\n" + "=" * 60)
    print("Example 4: Minimum GPU Requirements")
    print("=" * 60)
    
    # Different model sizes (using MockConfig)
    models = {
        "1B": MockConfig(
            vocab_size=32000,
            hidden_size=2048,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=5504,
        ),
        "7B": MockConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
        ),
        "13B": MockConfig(
            vocab_size=32000,
            hidden_size=5120,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            intermediate_size=13824,
        ),
        "70B": MockConfig(
            vocab_size=32000,
            hidden_size=8192,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=64,
            intermediate_size=28672,
        ),
    }
    
    gpu_options = [
        ("RTX 3090", 24.0),
        ("RTX 4090", 24.0),
        ("A100-40GB", 40.0),
        ("A100-80GB", 80.0),
        ("H100-80GB", 80.0),
    ]
    
    print("Minimum GPU requirements (fp16, seq_len=2048):")
    print(f"{'Model':<8} {'Weights':<10} {'Total Mem':<10} {'Recommended GPU'}")
    print("-" * 50)
    
    for model_name, config in models.items():
        
        memory = estimate_memory(
            model_config=config,
            sequence_length=2048,
            batch_size=1,
            quantization_bytes=2,  # fp16
            framework_overhead_gb=1.0,  # Reduce framework overhead to see differences
        )
        
        weights_gb = memory.weights / (1024**3)
        memory_gb = memory.total / (1024**3)
        
        # Find minimum GPU
        recommended_gpu = "None available"
        for gpu_name, gpu_memory in gpu_options:
            if memory.fits_in_gpu(gpu_memory):
                recommended_gpu = gpu_name
                break
        
        print(f"{model_name:<8} {weights_gb:.1f}GB{'':<5} {memory_gb:.1f}GB{'':<5} {recommended_gpu}")


def example_5_parallelism_scaling():
    """Example 5: Show how parallelism reduces per-GPU memory requirements."""
    print("\n" + "=" * 60)
    print("Example 5: Parallelism Memory Scaling")
    print("=" * 60)
    
    # 70B model configuration
    config = MockConfig(
        vocab_size=32000,
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=64,
        intermediate_size=28672,
    )
    
    # Base memory estimation
    base_memory = estimate_memory(
        model_config=config,
        sequence_length=2048,
        batch_size=1,
        quantization_bytes=2,  # fp16
    )
    
    print("70B Model memory scaling with tensor parallelism (fp16, seq_len=2048):")
    print(f"{'TP Size':<8} {'Weights/GPU':<12} {'Total/GPU':<12} {'Fits A100-80GB'}")
    print("-" * 50)
    
    for tp_size in [1, 2, 4, 8]:
        scaled_memory = base_memory.scale_by_parallelism(tensor_parallel=tp_size)
        fits_a100_80gb = scaled_memory.fits_in_gpu(80.0)
        fits_status = "✓" if fits_a100_80gb else "✗"
        
        print(f"{tp_size:<8} {format_gb(scaled_memory.weights):<12} "
              f"{format_gb(scaled_memory.total):<12} {fits_status}")


def example_6_error_handling():
    """Example 6: Demonstrate error handling for insufficient memory scenarios."""
    print("\n" + "=" * 60)
    print("Example 6: Error Handling and Memory Warnings")
    print("=" * 60)
    
    # Large model that won't fit on common GPUs (175B-like)
    large_config = MockConfig(
        vocab_size=128000,
        hidden_size=12288,
        num_hidden_layers=96,
        num_attention_heads=96,
        num_key_value_heads=96,
        intermediate_size=49152,
    )
    
    # Check against different GPU memory sizes
    gpu_sizes = [8.0, 16.0, 24.0, 40.0, 80.0]
    
    print("Large model (175B-like) memory feasibility:")
    print(f"{'GPU Memory':<12} {'Fits?':<8} {'Utilization':<12} {'Recommendations'}")
    print("-" * 60)
    
    for gpu_memory in gpu_sizes:
        try:
            feasibility = check_memory_feasibility(
                model_config=large_config,
                gpu_memory_gb=gpu_memory,
                sequence_length=2048,
                batch_size=1,
                quantization_bytes=2,  # fp16
            )
            
            fits = feasibility["fits_in_single_gpu"]
            utilization = feasibility["memory_utilization"]
            
            if not fits:
                if utilization > 2.0:
                    recommendation = "Use tensor parallelism"
                elif utilization > 1.5:
                    recommendation = "Try int8 quantization"
                else:
                    recommendation = "Reduce sequence length"
            else:
                if utilization > 0.9:
                    recommendation = "High utilization"
                else:
                    recommendation = "Good fit"
            
            fits_symbol = "✓" if fits else "✗"
            print(f"{gpu_memory:.0f}GB{'':<8} {fits_symbol:<8} "
                  f"{utilization:.1f}x{'':<8} {recommendation}")
            
        except Exception as e:
            print(f"{gpu_memory:.0f}GB{'':<8} Error{'':<4} --{'':<10} {str(e)[:30]}...")


def example_7_moe_model():
    """Example 7: Memory estimation for Mixture of Experts (MoE) models."""
    print("\n" + "=" * 60)
    print("Example 7: Mixture of Experts (MoE) Memory Estimation")
    print("=" * 60)
    
    # MoE model configuration (similar to Switch Transformer)
    moe_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        num_experts=8,  # 8 experts per layer
    )
    
    # Dense equivalent for comparison
    dense_config = MockConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
    )
    
    print("Comparing MoE vs Dense models:")
    print(f"{'Model Type':<12} {'Weights':<12} {'Total Memory':<15} {'Parameters'}")
    print("-" * 55)
    
    for name, config in [("Dense", dense_config), ("MoE (8 exp)", moe_config)]:
        memory = estimate_memory(
            model_config=config,
            sequence_length=2048,
            batch_size=1,
            quantization_bytes=2,  # fp16
        )
        
        # Use the built-in parameter estimation
        from autoparallel.memory import _estimate_param_count
        
        total_params = _estimate_param_count(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_experts=getattr(config, "num_experts", 0),
        )
        
        print(f"{name:<12} {format_gb(memory.weights):<12} "
              f"{format_gb(memory.total):<15} ~{total_params/1e9:.1f}B")
    
    print("\nNote: MoE models have more parameters but similar activation memory")
    print("Expert parallelism can distribute experts across GPUs")


def main():
    """Run all memory checking examples."""
    print("AutoParallel Memory Requirement Analysis Examples")
    print("=" * 60)
    print("This script demonstrates various memory estimation capabilities")
    print("including precision formats, sequence length scaling, and GPU requirements.")
    
    try:
        example_1_basic_memory_check()
        example_2_precision_comparison()
        example_3_sequence_length_impact()
        example_4_find_minimum_gpu()
        example_5_parallelism_scaling()
        example_6_error_handling()
        example_7_moe_model()
        
        print("\n" + "=" * 60)
        print("Memory Analysis Complete!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("• fp16/bf16 halves memory vs fp32, int8/fp8 quarters it")
        print("• KV cache and activations scale quadratically with sequence length")
        print("• Tensor parallelism reduces weights and KV cache per GPU")
        print("• Pipeline parallelism reduces activations per GPU")
        print("• Expert parallelism distributes MoE experts across GPUs")
        print("• Always include safety margin (10-20%) for real deployments")
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("This may indicate missing dependencies or model access issues")


if __name__ == "__main__":
    main()
