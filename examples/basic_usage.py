#!/usr/bin/env python3
"""
Basic Usage Examples for AutoParallel

This script demonstrates the autoparallel simplified API with progressive disclosure:
- Start simple with basic analysis
- Show cluster configuration options
- Demonstrate result interpretation
- Include proper error handling
- Show advanced options for power users

Run with: uv run python examples/basic_usage.py
"""

import sys

# Add src to path for example execution
sys.path.insert(0, "src")

import autoparallel


def example_1_basic_analysis():
    """Example 1: Basic model analysis with minimal configuration."""
    print("=" * 60)
    print("Example 1: Basic Analysis")
    print("=" * 60)

    # Define a simple cluster - just specify GPU count and memory
    cluster = {
        "gpu_count": 4,
        "gpu_memory_gb": 24.0,  # RTX 4090 / RTX 3090 specs
        "gpu_type": "RTX_4090",  # Optional: for documentation
    }

    # Analyze a lightweight conversational model
    model = "microsoft/DialoGPT-medium"

    try:
        print(f"Analyzing model: {model}")
        print(f"Cluster: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB GPUs")
        print()

        # Get all valid configurations (ranked by efficiency)
        configs = autoparallel.analyze(model, cluster)

        print(f"Found {len(configs)} valid configurations:")
        print()

        # Show top 3 configurations
        for i, config in enumerate(configs[:3], 1):
            print(f"Configuration {i}:")
            print(f"  Tensor Parallel: {config['tensor_parallel']}")
            print(f"  Pipeline Parallel: {config['pipeline_parallel']}")
            print(f"  Data Parallel: {config['data_parallel']}")
            print(f"  Total GPUs: {config['total_gpus']}")
            print(f"  Memory per GPU: {config['memory_per_gpu_gb']:.1f} GB")
            print(f"  Memory Utilization: {config['memory_utilization']:.1%}")
            print(f"  Efficiency Score: {config['score']:.1f}")
            print()

    except autoparallel.ModelNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have internet access and the model name is correct.")
    except autoparallel.InsufficientMemoryError as e:
        print(f"Error: {e}")
        print("Try a smaller model or increase GPU memory.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def example_2_single_best_config():
    """Example 2: Get single best configuration for different objectives."""
    print("=" * 60)
    print("Example 2: Single Best Configuration")
    print("=" * 60)

    # Larger cluster for demonstration
    cluster = {
        "gpu_count": 8,
        "gpu_memory_gb": 80.0,  # A100 80GB specs
        "gpu_type": "A100_80GB",
    }

    # Try a larger model that benefits from parallelism
    model = "microsoft/DialoGPT-large"

    objectives = [
        ("minimize_gpus", "Use fewest GPUs possible"),
        ("maximize_throughput", "Optimize for maximum throughput"),
        ("balance", "Balanced resource usage"),
    ]

    print(f"Analyzing model: {model}")
    print(f"Cluster: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB GPUs")
    print()

    for objective, description in objectives:
        try:
            config = autoparallel.best_config(
                model=model, cluster=cluster, objective=objective
            )

            print(f"Objective: {objective} ({description})")
            print(
                f"  Strategy: TP={config['tensor_parallel']}, "
                f"PP={config['pipeline_parallel']}, "
                f"DP={config['data_parallel']}"
            )
            print(f"  GPUs Used: {config['total_gpus']}")
            print(f"  Memory per GPU: {config['memory_per_gpu_gb']:.1f} GB")
            print(f"  Memory Utilization: {config['memory_utilization']:.1%}")
            print()

        except Exception as e:
            print(f"Objective: {objective} - Error: {e}")
            print()


def example_3_memory_analysis():
    """Example 3: Understand memory requirements before deployment."""
    print("=" * 60)
    print("Example 3: Memory Requirements Analysis")
    print("=" * 60)

    model = "microsoft/DialoGPT-medium"

    try:
        # Check memory requirements without hardware constraints
        memory_info = autoparallel.check_memory_requirements(
            model=model,
            sequence_length=1024,  # Shorter sequences for chat
            batch_size=1,
            quantization="fp16",
        )

        print(f"Memory analysis for {model}:")
        print(f"Total memory required: {memory_info['total_memory_gb']:.2f} GB")
        print()

        print("Memory breakdown:")
        breakdown = memory_info["breakdown"]
        for component, size_gb in breakdown.items():
            print(f"  {component}: {size_gb:.2f} GB")
        print()

        print("GPU compatibility:")
        gpu_compat = memory_info["single_gpu_requirements"]["fits_in_common_gpus"]
        for gpu, fits in gpu_compat.items():
            status = "✓" if fits else "✗"
            print(f"  {gpu}: {status}")
        print()

        print("Architecture info:")
        arch = memory_info["architecture_info"]
        print(f"  Model type: {arch['model_type']}")
        print(f"  Parameters: {arch['num_parameters_estimate']}")
        print(f"  Layers: {arch['num_layers']}")
        print(f"  Hidden size: {arch['hidden_size']}")
        print(f"  Attention heads: {arch['attention_heads']}")
        print()

    except Exception as e:
        print(f"Error analyzing memory: {e}")


def example_4_cluster_configurations():
    """Example 4: Different cluster configurations and their trade-offs."""
    print("=" * 60)
    print("Example 4: Cluster Configuration Comparison")
    print("=" * 60)

    model = "microsoft/DialoGPT-medium"

    # Different cluster setups to compare
    clusters = [
        {
            "name": "Budget Setup",
            "cluster": {"gpu_count": 2, "gpu_memory_gb": 24.0, "gpu_type": "RTX_4090"},
        },
        {
            "name": "Workstation",
            "cluster": {"gpu_count": 4, "gpu_memory_gb": 24.0, "gpu_type": "RTX_4090"},
        },
        {
            "name": "Cloud Instance",
            "cluster": {"gpu_count": 8, "gpu_memory_gb": 40.0, "gpu_type": "A100_40GB"},
        },
        {
            "name": "High-end Cloud",
            "cluster": {"gpu_count": 8, "gpu_memory_gb": 80.0, "gpu_type": "A100_80GB"},
        },
    ]

    print(f"Comparing cluster setups for {model}:")
    print()

    for setup in clusters:
        name = setup["name"]
        cluster = setup["cluster"]

        try:
            config = autoparallel.best_config(
                model=model, cluster=cluster, objective="balance"
            )

            print(f"{name}:")
            print(
                f"  Hardware: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB {cluster['gpu_type']}"
            )
            print(
                f"  Strategy: TP={config['tensor_parallel']}, "
                f"PP={config['pipeline_parallel']}, "
                f"DP={config['data_parallel']}"
            )
            print(f"  GPUs Used: {config['total_gpus']}/{cluster['gpu_count']}")
            print(f"  Memory Utilization: {config['memory_utilization']:.1%}")
            print(f"  Efficiency Score: {config['score']:.1f}")
            print()

        except autoparallel.InsufficientMemoryError:
            print(f"{name}:")
            print(
                f"  Hardware: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB {cluster['gpu_type']}"
            )
            print("  Result: Insufficient memory for this model")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}")
            print()


def example_5_advanced_options():
    """Example 5: Advanced options for power users."""
    print("=" * 60)
    print("Example 5: Advanced Configuration Options")
    print("=" * 60)

    cluster = {"gpu_count": 4, "gpu_memory_gb": 24.0, "gpu_type": "RTX_4090"}

    model = "microsoft/DialoGPT-medium"

    # Show different quantization options
    quantizations = ["fp32", "fp16", "bf16", "int8"]

    print(f"Quantization comparison for {model}:")
    print()

    for quant in quantizations:
        try:
            config = autoparallel.best_config(
                model=model,
                cluster=cluster,
                objective="minimize_gpus",
                sequence_length=2048,
                batch_size=1,
                quantization=quant,
            )

            print(f"{quant.upper()}:")
            print(f"  GPUs needed: {config['total_gpus']}")
            print(f"  Memory per GPU: {config['memory_per_gpu_gb']:.1f} GB")
            print(f"  Memory utilization: {config['memory_utilization']:.1%}")
            print()

        except Exception as e:
            print(f"{quant.upper()}: Error - {e}")
            print()

    # Show batch size and sequence length effects
    print("Batch size and sequence length effects:")
    print()

    configs_to_test = [
        {"batch_size": 1, "sequence_length": 512, "desc": "Small batch, short seq"},
        {"batch_size": 1, "sequence_length": 2048, "desc": "Small batch, long seq"},
        {"batch_size": 4, "sequence_length": 1024, "desc": "Medium batch, medium seq"},
    ]

    for test_config in configs_to_test:
        try:
            config = autoparallel.best_config(
                model=model,
                cluster=cluster,
                objective="balance",
                sequence_length=test_config["sequence_length"],
                batch_size=test_config["batch_size"],
                quantization="fp16",
            )

            print(f"{test_config['desc']}:")
            print(
                f"  Settings: batch_size={test_config['batch_size']}, "
                f"seq_len={test_config['sequence_length']}"
            )
            print(f"  GPUs needed: {config['total_gpus']}")
            print(f"  Memory per GPU: {config['memory_per_gpu_gb']:.1f} GB")
            print()

        except Exception as e:
            print(f"{test_config['desc']}: Error - {e}")
            print()


def example_6_cost_estimation():
    """Example 6: Cost estimation for deployment planning."""
    print("=" * 60)
    print("Example 6: Cost Estimation")
    print("=" * 60)

    cluster = {"gpu_count": 8, "gpu_memory_gb": 40.0, "gpu_type": "A100_40GB"}

    model = "microsoft/DialoGPT-medium"

    try:
        # Estimate costs for a cloud deployment
        cost_analysis = autoparallel.estimate_cost(
            model=model,
            cluster=cluster,
            hours_per_month=730,  # Full month
            cost_per_gpu_hour=2.50,  # Approximate cloud cost
            sequence_length=1024,
            batch_size=1,
            quantization="fp16",
        )

        print(f"Cost analysis for {model} deployment:")
        print(
            f"Assumptions: ${cost_analysis['assumptions']['cost_per_gpu_hour']}/GPU/hour, "
            f"{cost_analysis['assumptions']['hours_per_month']} hours/month"
        )
        print()

        for objective, analysis in cost_analysis["cost_analysis"].items():
            if "error" in analysis:
                print(f"{objective}: {analysis['error']}")
                continue

            print(f"{objective.replace('_', ' ').title()}:")
            print(f"  GPUs used: {analysis['gpus_used']}")
            print(f"  Cost per hour: ${analysis['cost_per_hour']:.2f}")
            print(f"  Cost per month: ${analysis['cost_per_month']:.2f}")
            print(f"  Memory utilization: {analysis['memory_utilization']:.1%}")
            print()

    except Exception as e:
        print(f"Error in cost estimation: {e}")


def example_7_error_handling():
    """Example 7: Comprehensive error handling patterns."""
    print("=" * 60)
    print("Example 7: Error Handling Examples")
    print("=" * 60)

    # Example of handling various error conditions
    error_scenarios = [
        {
            "name": "Invalid model name",
            "model": "nonexistent/model-that-does-not-exist",
            "cluster": {"gpu_count": 4, "gpu_memory_gb": 24.0},
        },
        {
            "name": "Insufficient memory",
            "model": "microsoft/DialoGPT-medium",
            "cluster": {"gpu_count": 1, "gpu_memory_gb": 1.0},  # Too small
        },
        {
            "name": "Invalid cluster config",
            "model": "microsoft/DialoGPT-medium",
            "cluster": {"gpu_count": 0},  # Missing gpu_memory_gb
        },
    ]

    for scenario in error_scenarios:
        print(f"Testing: {scenario['name']}")

        try:
            config = autoparallel.best_config(
                model=scenario["model"],
                cluster=scenario["cluster"],
                objective="balance",
            )
            print(f"  Success: {config['total_gpus']} GPUs needed")

        except autoparallel.ModelNotFoundError as e:
            print(f"  Model Error: {str(e).split('.')[0]}...")  # First sentence only

        except autoparallel.InsufficientMemoryError as e:
            print(f"  Memory Error: {str(e).split('.')[0]}...")

        except ValueError as e:
            print(f"  Configuration Error: {str(e).split('.')[0]}...")

        except Exception as e:
            print(f"  Unexpected Error: {type(e).__name__}: {str(e)[:50]}...")

        print()


def example_8_find_minimum_requirements():
    """Example 8: Find minimum GPU requirements for a model."""
    print("=" * 60)
    print("Example 8: Find Minimum GPU Requirements")
    print("=" * 60)

    model = "microsoft/DialoGPT-medium"
    gpu_types = [
        {"name": "RTX 4090", "memory_gb": 24.0},
        {"name": "A100 40GB", "memory_gb": 40.0},
        {"name": "A100 80GB", "memory_gb": 80.0},
    ]

    print(f"Finding minimum GPU requirements for {model}:")
    print()

    for gpu in gpu_types:
        try:
            result = autoparallel.find_minimum_gpus(
                model=model,
                gpu_memory_gb=gpu["memory_gb"],
                sequence_length=1024,
                batch_size=1,
                quantization="fp16",
            )

            print(f"{gpu['name']} ({gpu['memory_gb']}GB):")
            print(f"  Minimum GPUs needed: {result['min_gpus']}")
            print(f"  Memory per GPU: {result['memory_per_gpu_gb']:.1f} GB")
            print(f"  Memory utilization: {result['memory_utilization']:.1%}")
            config = result["configuration"]
            print(
                f"  Strategy: TP={config['tensor_parallel']}, "
                f"PP={config['pipeline_parallel']}, "
                f"DP={config['data_parallel']}"
            )
            print()

        except Exception as e:
            print(f"{gpu['name']}: Error - {e}")
            print()


def main():
    """Run all examples with error handling."""
    print("AutoParallel API Examples")
    print("=" * 60)
    print("This script demonstrates the autoparallel simplified API")
    print("with progressive disclosure from basic to advanced usage.")
    print()

    examples = [
        example_1_basic_analysis,
        example_2_single_best_config,
        example_3_memory_analysis,
        example_4_cluster_configurations,
        example_5_advanced_options,
        example_6_cost_estimation,
        example_7_error_handling,
        example_8_find_minimum_requirements,
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except KeyboardInterrupt:
            print("\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"Error in example {i}: {e}")
            print()

        if i < len(examples):
            print("Continuing to next example...\n")
            print()

    print("=" * 60)
    print("Examples completed!")
    print()
    print("Key takeaways:")
    print("• Start with autoparallel.analyze() for exploration")
    print("• Use autoparallel.best_config() for single recommendations")
    print("• Check memory requirements with autoparallel.check_memory_requirements()")
    print("• Compare costs with autoparallel.estimate_cost()")
    print("• Always include proper error handling")
    print("• Adjust quantization, batch size, and sequence length for your needs")


if __name__ == "__main__":
    main()
