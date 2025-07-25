#!/usr/bin/env python3
"""Example demonstrating objective-based configuration selection.

This example shows how to use autoparallel.best_config() with different objectives
and demonstrates cost estimation, trade-offs, and decision-making scenarios.
"""

import autoparallel


def compare_objectives_for_model():
    """Compare all three objectives for the same model and cluster."""
    
    print("=" * 70)
    print("Objective-Based Configuration Selection Demo")
    print("=" * 70)
    
    # Model and cluster setup
    model = "microsoft/DialoGPT-small"  # Small model for reliable demo
    cluster = {"gpu_count": 8, "gpu_memory_gb": 24.0}  # 8x A100 24GB
    
    print(f"Model: {model}")
    print(f"Cluster: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB GPUs")
    print()
    
    # Test each objective
    objectives = ["minimize_gpus", "maximize_throughput", "balance"]
    configs = {}
    
    print(f"{'Objective':<18} {'GPUs':<6} {'Memory Use':<12} {'Config':<25} {'Score':<8}")
    print("-" * 75)
    
    for objective in objectives:
        try:
            config = autoparallel.best_config(
                model=model,
                cluster=cluster,
                objective=objective,
                sequence_length=2048,
                batch_size=1,
                quantization="fp16"
            )
            
            configs[objective] = config
            
            # Format parallelism config
            config_str = f"TP={config['tensor_parallel']}, PP={config['pipeline_parallel']}, EP={config['expert_parallel']}, DP={config['data_parallel']}"
            
            print(f"{objective:<18} {config['total_gpus']:<6} {config['memory_utilization']:<11.1%} {config_str:<25} {config['score']:<8.2f}")
            
        except Exception as e:
            print(f"{objective:<18} Error: {str(e)}")
    
    print()
    return configs


def demonstrate_cost_estimation():
    """Show cost estimation for different objectives."""
    
    print("Cost Analysis Example")
    print("-" * 30)
    
    model = "microsoft/DialoGPT-small"
    cluster = {"gpu_count": 16, "gpu_memory_gb": 40.0}  # 16x A100 40GB
    
    # Get cost estimates
    cost_analysis = autoparallel.estimate_cost(
        model=model,
        cluster=cluster,
        hours_per_month=730,  # 24/7 operation
        cost_per_gpu_hour=3.0,  # $3/hour per A100
        sequence_length=2048,
        batch_size=1
    )
    
    print(f"Model: {model}")
    print(f"Cluster: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB GPUs")
    print(f"Assumptions: 730 hours/month at $3/GPU/hour")
    print()
    
    print(f"{'Objective':<18} {'GPUs':<6} {'$/Hour':<10} {'$/Month':<12} {'Memory %':<10}")
    print("-" * 60)
    
    for objective, analysis in cost_analysis["cost_analysis"].items():
        if "error" not in analysis:
            print(f"{objective:<18} {analysis['gpus_used']:<6} ${analysis['cost_per_hour']:<9.0f} ${analysis['cost_per_month']:<11.0f} {analysis['memory_utilization']:<9.1%}")
        else:
            print(f"{objective:<18} Error: {analysis['error']}")
    
    print()
    return cost_analysis


def analyze_scaling_scenarios():
    """Analyze different scaling scenarios and when to use each objective."""
    
    print("Scaling Scenarios & Objective Selection Guide")
    print("-" * 50)
    
    scenarios = [
        {
            "name": "Development/Testing",
            "cluster": {"gpu_count": 2, "gpu_memory_gb": 24.0},
            "recommended_objective": "minimize_gpus",
            "reason": "Cost-effective for dev work, minimal resource usage"
        },
        {
            "name": "Production Inference",
            "cluster": {"gpu_count": 8, "gpu_memory_gb": 40.0},
            "recommended_objective": "maximize_throughput",
            "reason": "Optimize for serving performance and latency"
        },
        {
            "name": "Batch Processing",
            "cluster": {"gpu_count": 16, "gpu_memory_gb": 80.0},
            "recommended_objective": "balance",
            "reason": "Balance resource efficiency with reasonable performance"
        },
        {
            "name": "Resource-Constrained",
            "cluster": {"gpu_count": 4, "gpu_memory_gb": 16.0},
            "recommended_objective": "minimize_gpus",
            "reason": "Make best use of limited GPU resources"
        }
    ]
    
    model = "microsoft/DialoGPT-small"
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Cluster: {scenario['cluster']['gpu_count']}x {scenario['cluster']['gpu_memory_gb']}GB")
        print(f"Recommended objective: {scenario['recommended_objective']}")
        print(f"Reason: {scenario['reason']}")
        
        try:
            # Get recommended config
            config = autoparallel.best_config(
                model=model,
                cluster=scenario["cluster"],
                objective=scenario["recommended_objective"]
            )
            
            print(f"Result: Uses {config['total_gpus']} GPUs ({config['memory_utilization']:.1%} memory)")
            print(f"Config: TP={config['tensor_parallel']}, DP={config['data_parallel']}")
            
        except Exception as e:
            print(f"Result: {str(e)}")


def demonstrate_trade_off_analysis():
    """Show detailed trade-off analysis between objectives."""
    
    print("\nTrade-off Analysis")
    print("-" * 25)
    
    model = "microsoft/DialoGPT-small"
    cluster = {"gpu_count": 8, "gpu_memory_gb": 40.0}
    
    print(f"Model: {model}")
    print(f"Cluster: {cluster['gpu_count']}x {cluster['gpu_memory_gb']}GB GPUs")
    print()
    
    # Analyze all objectives
    objectives = ["minimize_gpus", "maximize_throughput", "balance"]
    results = {}
    
    for objective in objectives:
        try:
            config = autoparallel.best_config(
                model=model,
                cluster=cluster,
                objective=objective
            )
            results[objective] = config
        except Exception as e:
            print(f"Error with {objective}: {e}")
            continue
    
    if not results:
        print("No valid configurations found")
        return
    
    # Compare key metrics
    print("Trade-off Analysis:")
    print()
    
    print("Resource Efficiency:")
    for obj, config in results.items():
        efficiency = config['total_gpus'] / cluster['gpu_count']
        print(f"  {obj:<18}: {config['total_gpus']}/{cluster['gpu_count']} GPUs ({efficiency:.1%})")
    
    print("\nMemory Utilization:")
    for obj, config in results.items():
        print(f"  {obj:<18}: {config['memory_utilization']:.1%}")
    
    print("\nParallelism Strategy:")
    for obj, config in results.items():
        strategy = f"TP={config['tensor_parallel']}, PP={config['pipeline_parallel']}, DP={config['data_parallel']}"
        print(f"  {obj:<18}: {strategy}")
    
    print("\nDecision Guidelines:")
    print("• minimize_gpus: Choose when GPU resources are limited or expensive")
    print("• maximize_throughput: Choose for production inference where performance matters")
    print("• balance: Choose for general-purpose workloads or when unsure")


def demonstrate_memory_requirements():
    """Show memory requirements analysis before configuration selection."""
    
    print("\nMemory Requirements Analysis")
    print("-" * 35)
    
    model = "microsoft/DialoGPT-small"
    
    # Get memory requirements first
    memory_req = autoparallel.check_memory_requirements(
        model=model,
        sequence_length=2048,
        batch_size=1,
        quantization="fp16"
    )
    
    print(f"Model: {model}")
    print(f"Total memory required: {memory_req['total_memory_gb']:.2f}GB")
    print(f"Recommended GPU memory: {memory_req['single_gpu_requirements']['recommended_memory_gb']:.1f}GB")
    print()
    
    print("GPU Compatibility:")
    gpu_compat = memory_req['single_gpu_requirements']['fits_in_common_gpus']
    for gpu, fits in gpu_compat.items():
        status = "✓" if fits else "✗"
        print(f"  {status} {gpu}")
    
    print(f"\nMemory Breakdown:")
    breakdown = memory_req['breakdown']
    for component, size_gb in breakdown.items():
        print(f"  {component}: {size_gb:.2f}GB")
    
    # Show minimum GPU requirements
    try:
        min_gpu_info = autoparallel.find_minimum_gpus(
            model=model,
            gpu_memory_gb=24.0  # A100 24GB
        )
        print(f"\nMinimum GPUs needed (24GB each): {min_gpu_info['min_gpus']}")
        print(f"Memory utilization: {min_gpu_info['memory_utilization']:.1%}")
    except Exception as e:
        print(f"\nCould not determine minimum GPUs: {e}")


def demonstrate_real_world_scenarios():
    """Show realistic decision-making scenarios."""
    
    print("\nReal-World Decision Scenarios")
    print("-" * 40)
    
    # Scenario 1: Startup with limited budget
    print("Scenario 1: Early-stage startup")
    print("• Limited budget, need to minimize costs")
    print("• Small team, development and testing workloads")
    print("• Recommendation: minimize_gpus objective")
    
    cluster = {"gpu_count": 4, "gpu_memory_gb": 24.0}
    try:
        config = autoparallel.best_config(
            model="microsoft/DialoGPT-small",
            cluster=cluster,
            objective="minimize_gpus"
        )
        print(f"• Result: {config['total_gpus']} GPUs, {config['memory_utilization']:.1%} memory use")
    except Exception as e:
        print(f"• Error: {e}")
    
    print()
    
    # Scenario 2: Production service
    print("Scenario 2: Production AI service")
    print("• High request volume, latency requirements")
    print("• Budget allows for performance optimization")
    print("• Recommendation: maximize_throughput objective")
    
    cluster = {"gpu_count": 16, "gpu_memory_gb": 40.0}
    try:
        config = autoparallel.best_config(
            model="microsoft/DialoGPT-small",
            cluster=cluster,
            objective="maximize_throughput"
        )
        print(f"• Result: {config['total_gpus']} GPUs, {config['memory_utilization']:.1%} memory use")
    except Exception as e:
        print(f"• Error: {e}")
    
    print()
    
    # Scenario 3: Research institution
    print("Scenario 3: Research institution")
    print("• Mixed workloads (training, inference, experiments)")
    print("• Need to balance efficiency and performance")
    print("• Recommendation: balance objective")
    
    cluster = {"gpu_count": 8, "gpu_memory_gb": 80.0}
    try:
        config = autoparallel.best_config(
            model="microsoft/DialoGPT-small",
            cluster=cluster,
            objective="balance"
        )
        print(f"• Result: {config['total_gpus']} GPUs, {config['memory_utilization']:.1%} memory use")
    except Exception as e:
        print(f"• Error: {e}")


def main():
    """Run all batch optimization examples."""
    
    try:
        # Basic objective comparison
        configs = compare_objectives_for_model()
        
        # Cost analysis
        cost_analysis = demonstrate_cost_estimation()
        
        # Scaling scenarios
        analyze_scaling_scenarios()
        
        # Trade-off analysis
        demonstrate_trade_off_analysis()
        
        # Memory requirements
        demonstrate_memory_requirements()
        
        # Real-world scenarios
        demonstrate_real_world_scenarios()
        
        print("\n" + "=" * 70)
        print("Summary: Objective Selection Guidelines")
        print("=" * 70)
        print()
        print("🎯 minimize_gpus:")
        print("   • Use when: Limited GPU budget, development/testing, resource constraints")
        print("   • Benefits: Lowest cost, minimal resource usage")
        print("   • Trade-offs: May sacrifice some performance")
        print()
        print("🚀 maximize_throughput:")
        print("   • Use when: Production inference, high request volume, performance critical")
        print("   • Benefits: Best performance, lowest latency")
        print("   • Trade-offs: Higher resource usage and cost")
        print()
        print("⚖️  balance:")
        print("   • Use when: Mixed workloads, general-purpose, uncertain requirements")
        print("   • Benefits: Good compromise between cost and performance")
        print("   • Trade-offs: May not optimize for specific use case")
        print()
        print("💡 Pro tip: Start with 'balance' objective, then specialize based on needs")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This example requires internet access to download model configs")


if __name__ == "__main__":
    main()
