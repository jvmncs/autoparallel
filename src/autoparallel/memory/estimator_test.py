"""Tests for memory estimators."""

import pytest

from .components import MemoryComponents
from .config import MemoryConfig
from .estimator import MemoryEstimator, MoEMemoryEstimator, TransformersMemoryEstimator

# Mock model configurations based on real architectures
LLAMA_7B_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 11008,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
}

LLAMA_13B_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "intermediate_size": 13824,
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
}

GPT_6B_CONFIG = {
    "vocab_size": 50256,
    "hidden_size": 4096,
    "num_hidden_layers": 24,
    "intermediate_size": 16384,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
}

MIXTRAL_8X7B_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # Grouped query attention
    "num_experts": 8,
    "num_experts_per_token": 2,
}


class TestMemoryEstimator:
    """Test suite for base MemoryEstimator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator = ConcreteEstimator()
        assert isinstance(estimator.config, MemoryConfig)
        assert estimator.config.utilization_bound == 0.85
        assert estimator.config.quantization_format == "fp16"

    def test_init_custom_config(self):
        """Test initialization with custom config."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        config = MemoryConfig(
            utilization_bound=0.9,
            quantization_format="int8",
            fragmentation_overhead=0.15,
        )
        estimator = ConcreteEstimator(config)
        assert estimator.config is config
        assert estimator.config.utilization_bound == 0.9
        assert estimator.config.quantization_format == "int8"
        assert estimator.config.fragmentation_overhead == 0.15

    def test_estimate_weights_memory_llama_7b(self):
        """Test weights memory estimation for Llama 7B."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator = ConcreteEstimator()
        memory = estimator.estimate_weights_memory(LLAMA_7B_CONFIG)

        # Verify calculation manually
        vocab_size = 32000
        hidden_size = 4096
        num_layers = 32
        intermediate_size = 11008

        # Embedding
        embedding_params = vocab_size * hidden_size

        # Per layer
        attention_params = 4 * hidden_size * hidden_size
        mlp_params = 2 * hidden_size * intermediate_size
        layernorm_params = 2 * hidden_size
        layer_params = attention_params + mlp_params + layernorm_params

        # Total
        total_params = (
            embedding_params
            + num_layers * layer_params
            + hidden_size  # Final layer norm
            + vocab_size * hidden_size  # Output projection
        )

        expected_memory = total_params * 2  # fp16 = 2 bytes per param
        assert memory == expected_memory
        assert memory > 0

    def test_estimate_weights_memory_different_quantizations(self):
        """Test weights memory with different quantization formats."""
        config_fp16 = MemoryConfig(quantization_format="fp16")
        config_int8 = MemoryConfig(quantization_format="int8")
        config_int4 = MemoryConfig(quantization_format="int4")

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator_fp16 = ConcreteEstimator(config_fp16)
        estimator_int8 = ConcreteEstimator(config_int8)
        estimator_int4 = ConcreteEstimator(config_int4)

        memory_fp16 = estimator_fp16.estimate_weights_memory(LLAMA_7B_CONFIG)
        memory_int8 = estimator_int8.estimate_weights_memory(LLAMA_7B_CONFIG)
        memory_int4 = estimator_int4.estimate_weights_memory(LLAMA_7B_CONFIG)

        # Verify scaling relationships
        assert memory_int8 == memory_fp16 // 2  # int8 = 1 byte vs fp16 = 2 bytes
        assert memory_int4 == memory_fp16 // 4  # int4 = 0.5 bytes vs fp16 = 2 bytes

    def test_estimate_activations_memory(self):
        """Test activations memory estimation."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator = ConcreteEstimator()
        memory = estimator.estimate_activations_memory(
            LLAMA_7B_CONFIG, sequence_length=2048, batch_size=1
        )

        # Verify memory is calculated
        assert memory > 0

        # Test scaling with batch size
        memory_batch_2 = estimator.estimate_activations_memory(
            LLAMA_7B_CONFIG, sequence_length=2048, batch_size=2
        )
        assert memory_batch_2 == memory * 2

        # Test scaling with sequence length (quadratic for attention)
        memory_seq_4096 = estimator.estimate_activations_memory(
            LLAMA_7B_CONFIG, sequence_length=4096, batch_size=1
        )
        assert memory_seq_4096 > memory * 2  # Should be > 2x due to quadratic attention

    def test_estimate_kv_cache_memory(self):
        """Test KV cache memory estimation."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator = ConcreteEstimator()
        memory = estimator.estimate_kv_cache_memory(
            LLAMA_7B_CONFIG, sequence_length=2048, batch_size=1
        )

        # Verify calculation manually
        hidden_size = 4096
        num_layers = 32
        num_kv_heads = 32
        sequence_length = 2048
        batch_size = 1
        head_dim = hidden_size // LLAMA_7B_CONFIG["num_attention_heads"]

        expected_elements = (
            2 * batch_size * num_kv_heads * sequence_length * head_dim * num_layers
        )
        expected_memory = expected_elements * 2  # fp16

        assert memory == expected_memory

        # Test scaling with batch size
        memory_batch_2 = estimator.estimate_kv_cache_memory(
            LLAMA_7B_CONFIG, sequence_length=2048, batch_size=2
        )
        assert memory_batch_2 == memory * 2

    def test_estimate_kv_cache_memory_grouped_query_attention(self):
        """Test KV cache with grouped query attention (fewer KV heads)."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator = ConcreteEstimator()

        # Compare regular attention vs grouped query attention
        memory_regular = estimator.estimate_kv_cache_memory(
            LLAMA_7B_CONFIG, sequence_length=2048, batch_size=1
        )
        memory_gqa = estimator.estimate_kv_cache_memory(
            MIXTRAL_8X7B_CONFIG, sequence_length=2048, batch_size=1
        )

        # GQA should use less memory (8 KV heads vs 32)
        assert memory_gqa < memory_regular

    def test_estimate_cuda_graphs_memory(self):
        """Test CUDA graphs memory estimation."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        config = MemoryConfig(cuda_graph_overhead_mb=256)
        estimator = ConcreteEstimator(config)

        memory = estimator.estimate_cuda_graphs_memory(LLAMA_7B_CONFIG)
        expected = 256 * 1024 * 1024  # 256 MB in bytes
        assert memory == expected

    def test_estimate_optimizer_memory_training(self):
        """Test optimizer memory estimation for training."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        config = MemoryConfig(optimizer_memory_fraction=2.0)
        estimator = ConcreteEstimator(config)

        weights_memory = estimator.estimate_weights_memory(LLAMA_7B_CONFIG)
        optimizer_memory = estimator.estimate_optimizer_memory(
            LLAMA_7B_CONFIG, is_training=True
        )

        assert optimizer_memory == weights_memory * 2.0

    def test_estimate_optimizer_memory_inference(self):
        """Test optimizer memory estimation for inference."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        estimator = ConcreteEstimator()
        optimizer_memory = estimator.estimate_optimizer_memory(
            LLAMA_7B_CONFIG, is_training=False
        )

        assert optimizer_memory == 0

    def test_estimate_fragmentation_overhead(self):
        """Test fragmentation overhead estimation."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        config = MemoryConfig(fragmentation_overhead=0.15)
        estimator = ConcreteEstimator(config)

        base_memory = 1000
        overhead = estimator.estimate_fragmentation_overhead(base_memory)
        assert overhead == 150  # 15% of 1000

    def test_estimate_safety_margin(self):
        """Test safety margin estimation."""

        class ConcreteEstimator(MemoryEstimator):
            def estimate_memory(self, *args, **kwargs):
                pass

        config = MemoryConfig(safety_margin=0.1)
        estimator = ConcreteEstimator(config)

        base_memory = 2000
        margin = estimator.estimate_safety_margin(base_memory)
        assert margin == 200  # 10% of 2000


class TestTransformersMemoryEstimator:
    """Test suite for TransformersMemoryEstimator class."""

    def test_estimate_memory_inference(self):
        """Test memory estimation for inference."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            is_training=False,
        )

        assert isinstance(components, MemoryComponents)
        assert components.weights > 0
        assert components.activations > 0
        assert components.kv_cache > 0
        assert components.cuda_graphs > 0
        assert components.optimizer_states == 0  # No optimizer for inference
        assert components.fragmentation_overhead > 0
        assert components.framework_overhead > 0
        assert components.safety_margin > 0
        assert components.total_memory > 0

    def test_estimate_memory_training(self):
        """Test memory estimation for training."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            is_training=True,
        )

        assert components.optimizer_states > 0  # Should have optimizer memory
        assert components.optimizer_states == components.weights * 2  # Default 200%

    def test_estimate_memory_tensor_parallelism(self):
        """Test memory estimation with tensor parallelism."""
        estimator = TransformersMemoryEstimator()

        components_tp1 = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            tensor_parallel_size=1,
        )

        components_tp2 = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            tensor_parallel_size=2,
        )

        # With TP=2, weights should be roughly half
        assert components_tp2.weights < components_tp1.weights
        assert components_tp2.activations < components_tp1.activations
        assert components_tp2.kv_cache < components_tp1.kv_cache

        # CUDA graphs don't scale with TP
        assert components_tp2.cuda_graphs == components_tp1.cuda_graphs

    def test_estimate_memory_pipeline_parallelism(self):
        """Test memory estimation with pipeline parallelism."""
        estimator = TransformersMemoryEstimator()

        components_pp1 = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            pipeline_parallel_size=1,
        )

        components_pp2 = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            pipeline_parallel_size=2,
        )

        # Pipeline parallelism doesn't affect per-device memory in current impl
        # since scale_by_parallelism only considers tensor parallelism
        # This test verifies the current behavior
        assert components_pp2.weights == components_pp1.weights
        assert components_pp2.activations == components_pp1.activations
        assert components_pp2.kv_cache == components_pp1.kv_cache

    def test_estimate_memory_different_models(self):
        """Test memory estimation with different model sizes."""
        estimator = TransformersMemoryEstimator()

        components_7b = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        components_13b = estimator.estimate_memory(
            LLAMA_13B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # 13B model should use more memory than 7B
        assert components_13b.weights > components_7b.weights
        assert components_13b.total_memory > components_7b.total_memory

    def test_estimate_memory_batch_scaling(self):
        """Test memory scaling with batch size."""
        estimator = TransformersMemoryEstimator()

        components_b1 = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        components_b4 = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=4,
        )

        # Weights don't scale with batch size
        assert components_b4.weights == components_b1.weights

        # Activations and KV cache scale with batch size
        assert components_b4.activations > components_b1.activations
        assert components_b4.kv_cache > components_b1.kv_cache

    def test_estimate_memory_sequence_scaling(self):
        """Test memory scaling with sequence length."""
        estimator = TransformersMemoryEstimator()

        components_s1k = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=1024,
            batch_size=1,
        )

        components_s4k = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=4096,
            batch_size=1,
        )

        # Weights don't scale with sequence length
        assert components_s4k.weights == components_s1k.weights

        # Activations and KV cache scale with sequence length
        assert components_s4k.activations > components_s1k.activations
        assert components_s4k.kv_cache > components_s1k.kv_cache

    def test_estimate_memory_custom_config(self):
        """Test memory estimation with custom configuration."""
        config = MemoryConfig(
            quantization_format="int8",
            fragmentation_overhead=0.2,
            safety_margin=0.1,
            cuda_graph_overhead_mb=1024,
        )
        estimator = TransformersMemoryEstimator(config)

        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # Verify config effects
        assert estimator.config.quantization_format == "int8"
        assert components.cuda_graphs == 1024 * 1024 * 1024  # 1GB

        # Compare with default config
        default_estimator = TransformersMemoryEstimator()
        default_components = default_estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # int8 weights should be half of fp16
        assert components.weights == default_components.weights // 2


class TestMoEMemoryEstimator:
    """Test suite for MoEMemoryEstimator class."""

    def test_estimate_memory_moe_basic(self):
        """Test basic MoE memory estimation."""
        estimator = MoEMemoryEstimator()
        components = estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        assert isinstance(components, MemoryComponents)
        assert components.weights > 0
        assert components.activations > 0
        assert components.kv_cache > 0
        assert components.total_memory > 0

    def test_estimate_memory_moe_vs_transformer(self):
        """Test MoE vs regular transformer memory differences."""
        moe_estimator = MoEMemoryEstimator()
        transformer_estimator = TransformersMemoryEstimator()

        moe_components = moe_estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # Create comparable transformer config (same base dimensions)
        transformer_config = MIXTRAL_8X7B_CONFIG.copy()
        del transformer_config["num_experts"]
        del transformer_config["num_experts_per_token"]

        transformer_components = transformer_estimator.estimate_memory(
            transformer_config,
            sequence_length=2048,
            batch_size=1,
        )

        # MoE should have different weight distribution but similar activations
        assert moe_components.activations == transformer_components.activations
        assert moe_components.kv_cache == transformer_components.kv_cache

    def test_estimate_memory_expert_parallelism(self):
        """Test memory estimation with expert parallelism."""
        estimator = MoEMemoryEstimator()

        components_ep1 = estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            expert_parallel_size=1,
        )

        components_ep2 = estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            expert_parallel_size=2,
        )

        # With EP=2, expert weights should be distributed
        assert components_ep2.weights <= components_ep1.weights

        # Non-expert components should be similar
        assert components_ep2.activations == components_ep1.activations
        assert components_ep2.kv_cache == components_ep1.kv_cache

    def test_estimate_memory_moe_training(self):
        """Test MoE memory estimation for training."""
        estimator = MoEMemoryEstimator()

        components_training = estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            expert_parallel_size=2,
            is_training=True,
        )

        components_inference = estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            expert_parallel_size=2,
            is_training=False,
        )

        # Training should have optimizer states
        assert components_training.optimizer_states > 0
        assert components_inference.optimizer_states == 0

        # Optimizer states should scale with expert parallelism
        assert components_training.optimizer_states > 0

    def test_estimate_memory_router_overhead(self):
        """Test router overhead in MoE models."""
        estimator = MoEMemoryEstimator()
        components = estimator.estimate_memory(
            MIXTRAL_8X7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # Router overhead should be included in framework overhead
        assert components.framework_overhead > 0

    def test_estimate_memory_different_expert_configs(self):
        """Test MoE with different expert configurations."""
        estimator = MoEMemoryEstimator()

        # Test with different num_experts_per_token
        config_2_experts = MIXTRAL_8X7B_CONFIG.copy()
        config_2_experts["num_experts_per_token"] = 2

        config_4_experts = MIXTRAL_8X7B_CONFIG.copy()
        config_4_experts["num_experts_per_token"] = 4

        components_2 = estimator.estimate_memory(
            config_2_experts,
            sequence_length=2048,
            batch_size=1,
        )

        components_4 = estimator.estimate_memory(
            config_4_experts,
            sequence_length=2048,
            batch_size=1,
        )

        # Using more experts per token should affect memory
        assert components_4.weights != components_2.weights


class TestMemoryEstimatorEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_zero_batch_size(self):
        """Test handling of zero batch size."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=0,
        )

        # Should handle gracefully (activations and KV cache should be 0)
        assert components.activations == 0
        assert components.kv_cache == 0
        assert components.weights > 0  # Weights don't depend on batch size

    def test_zero_sequence_length(self):
        """Test handling of zero sequence length."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=0,
            batch_size=1,
        )

        # Should handle gracefully
        assert components.activations == 0
        assert components.kv_cache == 0
        assert components.weights > 0  # Weights don't depend on sequence length

    def test_minimal_model_config(self):
        """Test with minimal model configuration."""
        minimal_config = {}  # Should use defaults

        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            minimal_config,
            sequence_length=2048,
            batch_size=1,
        )

        # Should work with defaults
        assert components.total_memory > 0

    def test_large_parallelism_values(self):
        """Test with large parallelism values."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            tensor_parallel_size=32,
            pipeline_parallel_size=8,
        )

        # Should scale appropriately without errors
        assert components.total_memory > 0
        assert components.weights > 0  # Should still have some memory per device

    def test_extreme_quantization_formats(self):
        """Test with extreme quantization formats."""
        config_int4 = MemoryConfig(quantization_format="int4")
        estimator = TransformersMemoryEstimator(config_int4)

        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # Should work with int4 quantization
        assert components.weights > 0
        assert components.total_memory > 0

    def test_memory_component_consistency(self):
        """Test that memory components are internally consistent."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
            is_training=True,
        )

        # Verify total memory is sum of all components
        expected_total = (
            components.weights
            + components.activations
            + components.kv_cache
            + components.cuda_graphs
            + components.optimizer_states
            + components.fragmentation_overhead
            + components.framework_overhead
            + components.safety_margin
        )

        assert components.total_memory == expected_total

    def test_inheritance_abstract_method(self):
        """Test that MemoryEstimator is properly abstract."""
        with pytest.raises(TypeError):
            MemoryEstimator()  # Should fail - abstract class

    def test_framework_overhead_calculation(self):
        """Test framework overhead calculation."""
        estimator = TransformersMemoryEstimator()
        components = estimator.estimate_memory(
            LLAMA_7B_CONFIG,
            sequence_length=2048,
            batch_size=1,
        )

        # Framework overhead should be 5% of base memory
        base_memory = (
            components.weights
            + components.activations
            + components.kv_cache
            + components.cuda_graphs
            + components.optimizer_states
        )

        expected_framework_overhead = base_memory * 0.05
        assert components.framework_overhead == int(expected_framework_overhead)


class TestMemoryEstimatorIntegration:
    """Integration tests between estimators and configuration."""

    def test_config_parameter_effects(self):
        """Test that configuration parameters properly affect estimates."""
        # Test different configurations
        configs = [
            MemoryConfig(fragmentation_overhead=0.05),
            MemoryConfig(fragmentation_overhead=0.15),
            MemoryConfig(safety_margin=0.02),
            MemoryConfig(safety_margin=0.10),
            MemoryConfig(cuda_graph_overhead_mb=256),
            MemoryConfig(cuda_graph_overhead_mb=1024),
        ]

        components_list = []
        for config in configs:
            estimator = TransformersMemoryEstimator(config)
            components = estimator.estimate_memory(
                LLAMA_7B_CONFIG,
                sequence_length=2048,
                batch_size=1,
            )
            components_list.append(components)

        # Verify different configs produce different results
        for i in range(len(components_list)):
            for j in range(i + 1, len(components_list)):
                # At least some component should be different
                assert (
                    components_list[i].fragmentation_overhead
                    != components_list[j].fragmentation_overhead
                    or components_list[i].safety_margin
                    != components_list[j].safety_margin
                    or components_list[i].cuda_graphs != components_list[j].cuda_graphs
                )

    def test_quantization_format_integration(self):
        """Test integration with all quantization formats."""
        formats = ["fp16", "bf16", "fp8", "int8", "int4", "gptq", "awq", "bitsandbytes"]

        components_by_format = {}
        for fmt in formats:
            config = MemoryConfig(quantization_format=fmt)
            estimator = TransformersMemoryEstimator(config)
            components = estimator.estimate_memory(
                LLAMA_7B_CONFIG,
                sequence_length=2048,
                batch_size=1,
            )
            components_by_format[fmt] = components

        # Verify expected relationships
        assert (
            components_by_format["fp16"].weights == components_by_format["bf16"].weights
        )  # Same size

        assert (
            components_by_format["int8"].weights
            == components_by_format["bitsandbytes"].weights
        )  # Same size

        assert (
            components_by_format["int4"].weights
            == components_by_format["gptq"].weights
            == components_by_format["awq"].weights
        )  # Same size

        # Size ordering
        assert (
            components_by_format["fp16"].weights > components_by_format["int8"].weights
        )
        assert (
            components_by_format["int8"].weights > components_by_format["int4"].weights
        )

    def test_realistic_model_scenarios(self):
        """Test realistic deployment scenarios."""
        scenarios = [
            # Small model inference
            {
                "config": LLAMA_7B_CONFIG,
                "seq_len": 2048,
                "batch_size": 1,
                "tp": 1,
                "pp": 1,
                "training": False,
            },
            # Large model with TP
            {
                "config": LLAMA_13B_CONFIG,
                "seq_len": 4096,
                "batch_size": 2,
                "tp": 2,
                "pp": 1,
                "training": False,
            },
            # Training scenario
            {
                "config": LLAMA_7B_CONFIG,
                "seq_len": 2048,
                "batch_size": 4,
                "tp": 1,
                "pp": 2,
                "training": True,
            },
            # MoE scenario
            {
                "config": MIXTRAL_8X7B_CONFIG,
                "seq_len": 2048,
                "batch_size": 1,
                "tp": 2,
                "pp": 1,
                "expert_parallel_size": 2,
                "training": False,
            },
        ]

        for i, scenario in enumerate(scenarios):
            if "num_experts" in scenario["config"]:
                estimator = MoEMemoryEstimator()
            else:
                estimator = TransformersMemoryEstimator()

            components = estimator.estimate_memory(
                scenario["config"],
                sequence_length=scenario["seq_len"],
                batch_size=scenario["batch_size"],
                tensor_parallel_size=scenario.get("tp", 1),
                pipeline_parallel_size=scenario.get("pp", 1),
                expert_parallel_size=scenario.get("expert_parallel_size", 1),
                is_training=scenario.get("training", False),
            )

            # Basic sanity checks
            assert components.total_memory > 0, f"Scenario {i} failed"
            assert components.weights > 0, f"Scenario {i} failed"

            if scenario.get("training", False):
                assert components.optimizer_states > 0, f"Scenario {i} failed"
            else:
                assert components.optimizer_states == 0, f"Scenario {i} failed"
