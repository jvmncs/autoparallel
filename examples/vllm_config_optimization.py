#!/usr/bin/env python3
"""Example demonstrating vLLM configuration optimization."""

from autoparallel.frameworks.vllm_config import (
    AutotuningParameters,
    optimize_vllm_config_for_cluster,
    vLLMConfigOptimizer,
)
from autoparallel.frameworks.vllm_memory import WorkloadProfile


def demonstrate_vllm_config_optimization():
    """Demonstrate vLLM configuration optimization for different scenarios."""

    print("=" * 60)
    print("vLLM Configuration Optimization Demo")
    print("=" * 60)

    # Configuration parameters
    model_name = "microsoft/DialoGPT-small"  # Small model for demo
    gpu_memory_capacity_gb = 24.0  # A100 24GB

    # Create different workload profiles
    chatbot_workload = WorkloadProfile.create_synthetic("chatbot")
    batch_workload = WorkloadProfile.create_synthetic("batch_inference")

    print(f"Model: {model_name}")
    print(f"GPU Memory: {gpu_memory_capacity_gb}GB")
    print()

    # Configure autotuning parameters
    tuning_params = AutotuningParameters(
        min_gpu_memory_utilization=0.8,
        max_gpu_memory_utilization=0.95,
        throughput_batch_weight=0.7,
        throughput_graph_weight=0.3,
    )

    # Demonstrate single-instance optimization
    print("1. Single Instance Optimization")
    print("-" * 40)

    optimizer = vLLMConfigOptimizer(
        model_name=model_name,
        gpu_memory_capacity_gb=gpu_memory_capacity_gb,
        tuning_params=tuning_params,
    )

    # Optimize for chatbot workload
    print("Optimizing for chatbot workload...")
    chatbot_result = optimizer.search_optimal_config(chatbot_workload)

    if chatbot_result["optimal_config"]:
        config = chatbot_result["optimal_config"]
        predictions = chatbot_result["predictions"]

        print("✓ Optimal configuration found")
        print(f"  - GPU Memory Utilization: {config.gpu_memory_utilization:.1%}")
        print(f"  - Max Model Length: {config.max_model_len}")
        print(f"  - KV Cache Data Type: {config.kv_cache_dtype}")
        print(f"  - CUDA Graph Captures: {len(config.cudagraph_capture_sizes)} sizes")
        print(f"  - Effective Batch Size: {predictions['effective_batch_size']}")
        print(f"  - Graph Coverage: {predictions['graph_coverage']:.1%}")
        print(f"  - Performance Score: {chatbot_result['performance_score']:.2f}")

        # Show memory breakdown
        memory = chatbot_result["memory_breakdown"]
        print("  Memory Breakdown:")
        print(f"    - Model: {memory['model_memory']:.2f}GB")
        print(f"    - Activations: {memory['activation_memory']:.2f}GB")
        print(f"    - CUDA Graphs: {memory['cuda_graph_memory']:.2f}GB")
        print(f"    - KV Cache: {memory['kv_cache_memory']:.2f}GB")

    print()

    # Demonstrate cluster optimization
    print("2. Cluster Optimization")
    print("-" * 40)

    # Different parallelism strategies
    strategies = [
        {"tp": 1, "pp": 1, "dp": 8},  # Data parallel only
        {"tp": 2, "pp": 1, "dp": 4},  # Tensor parallel + data parallel
        {"tp": 4, "pp": 1, "dp": 2},  # High tensor parallel
    ]

    for i, strategy in enumerate(strategies, 1):
        print(
            f"Strategy {i}: TP={strategy['tp']}, PP={strategy['pp']}, DP={strategy['dp']}"
        )

        result = optimize_vllm_config_for_cluster(
            model_name=model_name,
            gpu_memory_capacity_gb=gpu_memory_capacity_gb,
            workload=batch_workload,
            parallelism_strategy=strategy,
            tuning_params=tuning_params,
        )

        if result["vllm_config"]:
            cluster_pred = result["cluster_predictions"]
            print(f"  - Total Throughput: {cluster_pred['total_throughput']} sequences")
            print(f"  - Instances: {cluster_pred['instances_per_cluster']}")
            print(f"  - Memory Efficiency: {cluster_pred['memory_efficiency']:.1%}")
            print(f"  - Graph Coverage: {cluster_pred['graph_coverage']:.1%}")

            # Show recommendations
            recommendations = result["recommendations"]
            if recommendations:
                print("  Recommendations:")
                for rec in recommendations[:2]:  # Show first 2
                    print(f"    - {rec}")
        else:
            print("  ✗ No feasible configuration found")

        print()

    print("3. Configuration Validation")
    print("-" * 40)

    # Create a test configuration
    if chatbot_result["optimal_config"]:
        config = chatbot_result["optimal_config"]
        validation = optimizer.validate_configuration(config)

        print(f"Configuration valid: {validation['valid']}")
        print(f"Effective batch size: {validation['effective_batch_size']}")

        if validation["warnings"]:
            print("Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        if validation["recommendations"]:
            print("Recommendations:")
            for rec in validation["recommendations"]:
                print(f"  - {rec}")

    print()
    print("=" * 60)
    print("Demo completed!")


def demonstrate_workload_comparison():
    """Compare optimization results for different workload types."""

    print("=" * 60)
    print("Workload Type Comparison")
    print("=" * 60)

    model_name = "microsoft/DialoGPT-small"
    gpu_memory_capacity_gb = 24.0

    workloads = {
        "Chatbot": WorkloadProfile.create_synthetic("chatbot"),
        "Batch Inference": WorkloadProfile.create_synthetic("batch_inference"),
        "Interactive": WorkloadProfile.create_synthetic("interactive"),
    }

    optimizer = vLLMConfigOptimizer(model_name, gpu_memory_capacity_gb)

    print(f"{'Workload':<15} {'Batch Size':<12} {'Graph Coverage':<15} {'Score':<8}")
    print("-" * 60)

    for name, workload in workloads.items():
        result = optimizer.search_optimal_config(workload)

        if result["optimal_config"] and result["predictions"]:
            batch_size = result["predictions"]["effective_batch_size"]
            coverage = result["predictions"]["graph_coverage"]
            score = result["performance_score"]

            print(f"{name:<15} {batch_size:<12} {coverage:<14.1%} {score:<8.2f}")
        else:
            print(f"{name:<15} {'No solution':<12} {'':<15} {'':<8}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    try:
        demonstrate_vllm_config_optimization()
        print()
        demonstrate_workload_comparison()
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This example requires internet access to download model configs")
