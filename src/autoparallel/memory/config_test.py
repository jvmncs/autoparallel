"""Tests for memory configuration."""

import pytest

from autoparallel.memory.config import MemoryConfig


class TestMemoryConfig:
    """Tests for MemoryConfig initialization and validation."""

    def test_default_initialization(self):
        """Test MemoryConfig with default values."""
        config = MemoryConfig()

        assert config.utilization_bound == 0.85
        assert config.fragmentation_overhead == 0.10
        assert config.safety_margin == 0.05
        assert config.quantization_format == "fp16"
        assert config.min_kv_cache_fraction == 0.05
        assert config.cuda_graph_overhead_mb == 512.0
        assert config.optimizer_memory_fraction == 2.0

    def test_custom_initialization(self):
        """Test MemoryConfig with custom parameter values."""
        config = MemoryConfig(
            utilization_bound=0.90,
            fragmentation_overhead=0.15,
            safety_margin=0.10,
            quantization_format="bf16",
            min_kv_cache_fraction=0.10,
            cuda_graph_overhead_mb=1024.0,
            optimizer_memory_fraction=1.5,
        )

        assert config.utilization_bound == 0.90
        assert config.fragmentation_overhead == 0.15
        assert config.safety_margin == 0.10
        assert config.quantization_format == "bf16"
        assert config.min_kv_cache_fraction == 0.10
        assert config.cuda_graph_overhead_mb == 1024.0
        assert config.optimizer_memory_fraction == 1.5

    def test_production_deployment_configs(self):
        """Test realistic production deployment configurations."""
        # Conservative production config
        conservative_config = MemoryConfig(
            utilization_bound=0.80,
            fragmentation_overhead=0.15,
            safety_margin=0.10,
            quantization_format="fp16",
            min_kv_cache_fraction=0.10,
            cuda_graph_overhead_mb=1024.0,
            optimizer_memory_fraction=2.0,
        )
        assert abs(conservative_config.effective_utilization_bound - 0.72) < 1e-10

        # Aggressive config for max performance
        aggressive_config = MemoryConfig(
            utilization_bound=0.95,
            fragmentation_overhead=0.05,
            safety_margin=0.02,
            quantization_format="int4",
            min_kv_cache_fraction=0.03,
            cuda_graph_overhead_mb=256.0,
            optimizer_memory_fraction=1.0,
        )
        assert abs(aggressive_config.effective_utilization_bound - 0.931) < 1e-10

        # Quantized inference config
        inference_config = MemoryConfig(
            utilization_bound=0.85,
            fragmentation_overhead=0.08,
            safety_margin=0.05,
            quantization_format="int8",
            min_kv_cache_fraction=0.15,
            cuda_graph_overhead_mb=512.0,
            optimizer_memory_fraction=0.0,  # No optimizer for inference
        )
        assert inference_config.quantization_dtype_bytes == 1.0


class TestMemoryConfigValidation:
    """Tests for MemoryConfig parameter validation."""

    def test_valid_utilization_bound_values(self):
        """Test valid utilization bound values."""
        # Boundary values that should work
        MemoryConfig(utilization_bound=0.01)  # Minimum valid
        MemoryConfig(utilization_bound=0.50)  # Mid-range
        MemoryConfig(utilization_bound=1.0)  # Maximum valid

    def test_invalid_utilization_bound_values(self):
        """Test invalid utilization bound values raise ValueError."""
        error_msg = "utilization_bound must be in \\(0.0, 1.0\\]"
        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(utilization_bound=0.0)  # Exactly 0 is invalid

        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(utilization_bound=-0.1)  # Negative value

        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(utilization_bound=1.1)  # Greater than 1

    def test_valid_fragmentation_overhead_values(self):
        """Test valid fragmentation overhead values."""
        MemoryConfig(fragmentation_overhead=0.0)  # Minimum valid
        MemoryConfig(fragmentation_overhead=0.25)  # Mid-range
        MemoryConfig(fragmentation_overhead=0.99)  # Just under 1.0

    def test_invalid_fragmentation_overhead_values(self):
        """Test invalid fragmentation overhead values raise ValueError."""
        error_msg = "fragmentation_overhead must be in \\[0.0, 1.0\\)"
        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(fragmentation_overhead=-0.1)  # Negative value

        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(fragmentation_overhead=1.0)  # Exactly 1.0 is invalid

        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(fragmentation_overhead=1.5)  # Greater than 1.0

    def test_valid_safety_margin_values(self):
        """Test valid safety margin values."""
        MemoryConfig(safety_margin=0.0)  # Minimum valid
        MemoryConfig(safety_margin=0.15)  # Mid-range
        MemoryConfig(safety_margin=0.99)  # Just under 1.0

    def test_invalid_safety_margin_values(self):
        """Test invalid safety margin values raise ValueError."""
        with pytest.raises(ValueError, match="safety_margin must be in \\[0.0, 1.0\\)"):
            MemoryConfig(safety_margin=-0.05)  # Negative value

        with pytest.raises(ValueError, match="safety_margin must be in \\[0.0, 1.0\\)"):
            MemoryConfig(safety_margin=1.0)  # Exactly 1.0 is invalid

        with pytest.raises(ValueError, match="safety_margin must be in \\[0.0, 1.0\\)"):
            MemoryConfig(safety_margin=1.2)  # Greater than 1.0

    def test_valid_min_kv_cache_fraction_values(self):
        """Test valid min KV cache fraction values."""
        MemoryConfig(min_kv_cache_fraction=0.0)  # Minimum valid
        MemoryConfig(min_kv_cache_fraction=0.25)  # Mid-range
        MemoryConfig(min_kv_cache_fraction=1.0)  # Maximum valid

    def test_invalid_min_kv_cache_fraction_values(self):
        """Test invalid min KV cache fraction values raise ValueError."""
        error_msg = "min_kv_cache_fraction must be in \\[0.0, 1.0\\]"
        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(min_kv_cache_fraction=-0.1)  # Negative value

        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(min_kv_cache_fraction=1.5)  # Greater than 1.0

    def test_valid_cuda_graph_overhead_values(self):
        """Test valid CUDA graph overhead values."""
        MemoryConfig(cuda_graph_overhead_mb=0.0)  # Minimum valid
        MemoryConfig(cuda_graph_overhead_mb=512.0)  # Default value
        MemoryConfig(cuda_graph_overhead_mb=2048.0)  # Large value

    def test_invalid_cuda_graph_overhead_values(self):
        """Test invalid CUDA graph overhead values raise ValueError."""
        with pytest.raises(ValueError, match="cuda_graph_overhead_mb must be >= 0"):
            MemoryConfig(cuda_graph_overhead_mb=-1.0)  # Negative value

        with pytest.raises(ValueError, match="cuda_graph_overhead_mb must be >= 0"):
            MemoryConfig(cuda_graph_overhead_mb=-512.0)  # Large negative value

    def test_valid_optimizer_memory_fraction_values(self):
        """Test valid optimizer memory fraction values."""
        MemoryConfig(optimizer_memory_fraction=0.0)  # No optimizer state
        MemoryConfig(optimizer_memory_fraction=1.0)  # Equal to model size
        MemoryConfig(optimizer_memory_fraction=2.0)  # Default (Adam-like)
        MemoryConfig(optimizer_memory_fraction=3.0)  # Heavy optimizer

    def test_invalid_optimizer_memory_fraction_values(self):
        """Test invalid optimizer memory fraction values raise ValueError."""
        with pytest.raises(ValueError, match="optimizer_memory_fraction must be >= 0"):
            MemoryConfig(optimizer_memory_fraction=-0.5)  # Negative value

        with pytest.raises(ValueError, match="optimizer_memory_fraction must be >= 0"):
            MemoryConfig(optimizer_memory_fraction=-2.0)  # Large negative value


class TestMemoryConfigProperties:
    """Tests for MemoryConfig property methods."""

    def test_effective_utilization_bound_calculation(self):
        """Test effective utilization bound calculation."""
        # Default config
        config = MemoryConfig(utilization_bound=0.85, safety_margin=0.05)
        expected = 0.85 * (1.0 - 0.05)  # 0.85 * 0.95 = 0.8075
        assert config.effective_utilization_bound == expected

        # High utilization, low safety margin
        config = MemoryConfig(utilization_bound=0.95, safety_margin=0.02)
        expected = 0.95 * (1.0 - 0.02)  # 0.95 * 0.98 = 0.931
        assert config.effective_utilization_bound == expected

        # Conservative config
        config = MemoryConfig(utilization_bound=0.70, safety_margin=0.15)
        expected = 0.70 * (1.0 - 0.15)  # 0.70 * 0.85 = 0.595
        assert config.effective_utilization_bound == expected

        # Zero safety margin
        config = MemoryConfig(utilization_bound=0.85, safety_margin=0.0)
        expected = 0.85 * (1.0 - 0.0)  # 0.85 * 1.0 = 0.85
        assert config.effective_utilization_bound == expected

    def test_quantization_dtype_bytes_all_formats(self):
        """Test quantization dtype bytes for all supported formats."""
        # Test all quantization formats
        test_cases = [
            ("fp16", 2.0),
            ("bf16", 2.0),
            ("fp8", 1.0),
            ("int8", 1.0),
            ("int4", 0.5),
            ("gptq", 0.5),
            ("awq", 0.5),
            ("bitsandbytes", 1.0),
        ]

        for format_name, expected_bytes in test_cases:
            config = MemoryConfig(quantization_format=format_name)
            assert config.quantization_dtype_bytes == expected_bytes, (
                f"Format {format_name} should have {expected_bytes} bytes per parameter"
            )

    def test_quantization_format_type_safety(self):
        """Test that only valid QuantizationFormat values are accepted."""
        # These should all work (all valid formats)
        valid_formats = [
            "fp16",
            "bf16",
            "fp8",
            "int8",
            "int4",
            "gptq",
            "awq",
            "bitsandbytes",
        ]

        for format_name in valid_formats:
            config = MemoryConfig(quantization_format=format_name)
            assert config.quantization_format == format_name

    def test_quantization_bytes_precision_categories(self):
        """Test quantization formats grouped by precision."""
        # 16-bit formats
        fp16_config = MemoryConfig(quantization_format="fp16")
        bf16_config = MemoryConfig(quantization_format="bf16")
        assert fp16_config.quantization_dtype_bytes == 2.0
        assert bf16_config.quantization_dtype_bytes == 2.0

        # 8-bit formats
        fp8_config = MemoryConfig(quantization_format="fp8")
        int8_config = MemoryConfig(quantization_format="int8")
        bnb_config = MemoryConfig(quantization_format="bitsandbytes")
        assert fp8_config.quantization_dtype_bytes == 1.0
        assert int8_config.quantization_dtype_bytes == 1.0
        assert bnb_config.quantization_dtype_bytes == 1.0

        # 4-bit formats
        int4_config = MemoryConfig(quantization_format="int4")
        gptq_config = MemoryConfig(quantization_format="gptq")
        awq_config = MemoryConfig(quantization_format="awq")
        assert int4_config.quantization_dtype_bytes == 0.5
        assert gptq_config.quantization_dtype_bytes == 0.5
        assert awq_config.quantization_dtype_bytes == 0.5


class TestQuantizationFormat:
    """Tests for QuantizationFormat type."""

    def test_all_supported_formats(self):
        """Test that all documented quantization formats are supported."""
        expected_formats = {
            "fp16",
            "bf16",
            "fp8",
            "int8",
            "int4",
            "gptq",
            "awq",
            "bitsandbytes",
        }

        # Verify each format can be used in MemoryConfig
        for format_name in expected_formats:
            config = MemoryConfig(quantization_format=format_name)
            assert config.quantization_format == format_name
            assert config.quantization_dtype_bytes > 0

    def test_format_byte_mapping_consistency(self):
        """Test that format-to-bytes mapping is consistent with expectations."""
        # Test format categorization by memory usage
        high_precision = ["fp16", "bf16"]  # 2 bytes
        medium_precision = ["fp8", "int8", "bitsandbytes"]  # 1 byte
        low_precision = ["int4", "gptq", "awq"]  # 0.5 bytes

        for format_name in high_precision:
            config = MemoryConfig(quantization_format=format_name)
            assert config.quantization_dtype_bytes == 2.0

        for format_name in medium_precision:
            config = MemoryConfig(quantization_format=format_name)
            assert config.quantization_dtype_bytes == 1.0

        for format_name in low_precision:
            config = MemoryConfig(quantization_format=format_name)
            assert config.quantization_dtype_bytes == 0.5


class TestMemoryConfigEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_boundary_utilization_values(self):
        """Test boundary values for utilization bound."""
        # Test minimum valid value (just above 0)
        config = MemoryConfig(utilization_bound=0.001)
        assert config.utilization_bound == 0.001

        # Test maximum valid value
        config = MemoryConfig(utilization_bound=1.0)
        assert config.utilization_bound == 1.0
        # Default safety margin is 0.05
        assert config.effective_utilization_bound == 1.0 * (1.0 - 0.05)

    def test_zero_overhead_configurations(self):
        """Test configurations with zero overhead values."""
        config = MemoryConfig(
            fragmentation_overhead=0.0,
            safety_margin=0.0,
            min_kv_cache_fraction=0.0,
            cuda_graph_overhead_mb=0.0,
            optimizer_memory_fraction=0.0,
        )

        assert config.fragmentation_overhead == 0.0
        assert config.safety_margin == 0.0
        assert config.min_kv_cache_fraction == 0.0
        assert config.cuda_graph_overhead_mb == 0.0
        assert config.optimizer_memory_fraction == 0.0
        assert config.effective_utilization_bound == config.utilization_bound

    def test_maximum_overhead_configurations(self):
        """Test configurations with maximum overhead values."""
        config = MemoryConfig(
            fragmentation_overhead=0.99,
            safety_margin=0.99,
            min_kv_cache_fraction=1.0,
            cuda_graph_overhead_mb=10000.0,
            optimizer_memory_fraction=10.0,
        )

        assert config.fragmentation_overhead == 0.99
        assert config.safety_margin == 0.99
        assert config.min_kv_cache_fraction == 1.0
        assert config.cuda_graph_overhead_mb == 10000.0
        assert config.optimizer_memory_fraction == 10.0
        # Effective utilization should be very low
        assert config.effective_utilization_bound == 0.85 * (1.0 - 0.99)

    def test_multiple_validation_errors(self):
        """Test that the first validation error is raised when multiple errors exist."""
        # This should raise an error for utilization_bound, which is checked first
        error_msg = "utilization_bound must be in \\(0.0, 1.0\\]"
        with pytest.raises(ValueError, match=error_msg):
            MemoryConfig(
                utilization_bound=1.5,  # Invalid
                fragmentation_overhead=1.5,  # Also invalid
                safety_margin=1.5,  # Also invalid
            )

    def test_config_immutability_after_validation(self):
        """Test that config values remain consistent after creation."""
        config = MemoryConfig(
            utilization_bound=0.90,
            safety_margin=0.10,
            quantization_format="int4",
        )

        # Values should remain consistent
        original_effective = config.effective_utilization_bound
        original_bytes = config.quantization_dtype_bytes

        # Multiple accesses should return same values
        assert config.effective_utilization_bound == original_effective
        assert config.quantization_dtype_bytes == original_bytes
        assert config.effective_utilization_bound == 0.90 * 0.90  # 0.81
        assert config.quantization_dtype_bytes == 0.5
