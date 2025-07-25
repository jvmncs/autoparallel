# AutoParallel Simplified Testing Strategy

## Testing Philosophy

Focus on behavior testing rather than implementation details. Prioritize fast, maintainable tests that verify the core value proposition: "given a model and cluster, return valid parallelism configurations."

## Test Structure

### Unit Tests (Co-located)
Test individual functions and classes in isolation.

**memory_test.py**
```python
def test_estimate_memory_basic():
    """Test memory estimation for standard model."""

def test_estimate_memory_moe():
    """Test memory estimation for MoE model."""

def test_memory_breakdown_fits_in_gpu():
    """Test memory fitting logic."""

@given(st.integers(min_value=1, max_value=8))
def test_tensor_parallel_memory_scaling(tp_size):
    """Property test: TP reduces per-GPU memory."""
```

**constraints_test.py**
```python
def test_valid_tensor_parallel_sizes():
    """Test TP size constraints from attention heads."""

def test_valid_pipeline_parallel_sizes():
    """Test PP size constraints from layer count."""

def test_moe_expert_parallel_constraints():
    """Test EP constraints for MoE models."""
```

**grid_search_test.py**
```python
def test_find_valid_configs_returns_feasible():
    """Test that all returned configs fit in memory."""

def test_configs_ranked_by_preference():
    """Test that configs are properly ranked."""

def test_no_configs_when_insufficient_memory():
    """Test empty result for impossible constraints."""
```

**public_api_test.py**
```python
def test_analyze_llama_model():
    """Test analyze() with real Llama model config."""

def test_best_config_minimize_gpus():
    """Test objective-based selection."""

def test_error_handling():
    """Test error cases (invalid model, insufficient memory)."""
```

### Integration Tests (Separate Directory)
Test with real model configurations using meta-device loading.

**tests/integration/test_real_models.py**
Test against specific model families that users actually deploy:

#### Core Model Families for Validation

**Qwen3 Family (bf16)**
```python
@pytest.mark.parametrize("model_id,expected_arch", [
    # Dense Models
    ("Qwen/Qwen3-8B", {"type": "dense", "hidden_size": 4096, "num_layers": 36, "attention_heads": 32, "kv_heads": 8}),
    ("Qwen/Qwen3-14B", {"type": "dense", "hidden_size": 5120, "num_layers": 40, "attention_heads": 40, "kv_heads": 8}),
    ("Qwen/Qwen3-32B", {"type": "dense", "hidden_size": 5120, "num_layers": 64, "attention_heads": 64, "kv_heads": 8}),
    # MoE Models
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", {"type": "moe", "total_params": "235B", "active_params": "22B", "experts": 128, "active_experts": 8}),
])
def test_qwen3_family_constraints(model_id, expected_arch):
    """Test constraint analysis for Qwen3 family."""

@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",  # 480B/35B MoE, 160 experts, 8 active
])
def test_qwen3_coder_moe_constraints(model_id):
    """Test Qwen3-Coder MoE constraint analysis."""
```

**Llama-3.x Family (bf16)**
```python
@pytest.mark.parametrize("model_id,context_length", [
    ("meta-llama/Llama-3.1-8B-Instruct", 128000),   # 8B, 128K context
    ("meta-llama/Llama-3.1-70B-Instruct", 128000),  # 70B, 128K context
    ("meta-llama/Llama-3.2-1B-Instruct", 128000),   # 1B, 128K context
    ("meta-llama/Llama-3.2-3B-Instruct", 128000),   # 3B, 128K context
    ("meta-llama/Llama-3.3-70B-Instruct", 128000),  # 70B, 128K context
])
def test_llama3x_family_constraints(model_id, context_length):
    """Test constraint analysis for Llama-3.x family."""
```

**Llama-4 Family (bf16, MoE)**
```python
@pytest.mark.parametrize("model_id,moe_config", [
    ("meta-llama/Llama-4-Scout-17B-16E-Instruct", {"experts": 16, "active": 1, "context": 10000000}),
    ("meta-llama/Llama-4-Maverick-17B-128E-Instruct", {"experts": 128, "active": 1, "context": 1000000}),
])
def test_llama4_moe_multimodal_constraints(model_id, moe_config):
    """Test Llama-4 MoE constraint analysis with multimodal capabilities."""
```

**DeepSeek Family (fp8 finegrained blockwise)**
```python
@pytest.mark.parametrize("model_id,fp8_config", [
    ("deepseek-ai/DeepSeek-V3", {"precision": "fp8_e4m3", "experts": 256, "active": 8, "shared": 1}),
    ("deepseek-ai/DeepSeek-R1-0528", {"precision": "fp8_e4m3", "experts": 256, "active": 8, "shared": 1}),
])
def test_deepseek_fp8_moe_constraints(model_id, fp8_config):
    """Test DeepSeek family with native fp8 training."""
```

**Kimi-K2 Family (fp8 finegrained blockwise)**
```python
@pytest.mark.parametrize("model_id", [
    "moonshotai/Kimi-K2-Instruct",  # 1T/32B, 384 experts, 8 active, fp8
])
def test_kimi_k2_extreme_moe_constraints(model_id):
    """Test Kimi-K2 extreme MoE architecture."""
```

#### Precision-Specific Testing
```python
def test_bf16_models_memory_estimation():
    """Test memory estimation accuracy for bf16 models (Qwen3, Llama-3.x, Llama-4)."""

def test_fp8_models_memory_estimation():
    """Test memory estimation accuracy for fp8 models (DeepSeek, Kimi-K2)."""

def test_moe_expert_parallel_constraints():
    """Test expert parallelism constraints across all MoE families."""
```

### Performance Tests
Ensure analysis completes quickly.

**tests/performance/test_speed.py**
```python
def test_analysis_speed_under_1s():
    """Test that analysis completes in <1 second."""

def test_memory_estimation_speed():
    """Test memory estimation performance."""
```

## Testing Utilities

### Model Config Factory
```python
def create_test_model_config(
    model_type: str = "llama",
    size: str = "7B",
    moe_experts: Optional[int] = None,
    **overrides
) -> PretrainedConfig:
    """Create test model configs without hardcoded dictionaries."""

    base_configs = {
        "llama": {
            "7B": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32},
            "13B": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40},
            "70B": {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64}
        },
        "mixtral": {
            "8x7B": {"hidden_size": 4096, "num_hidden_layers": 32, "num_local_experts": 8}
        }
    }

    config_dict = base_configs[model_type][size].copy()
    if moe_experts:
        config_dict["num_local_experts"] = moe_experts
    config_dict.update(overrides)

    return MockConfig(config_dict)
```

### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    hidden_size=st.integers(min_value=1024, max_value=8192).filter(lambda x: x % 64 == 0),
    num_heads=st.integers(min_value=8, max_value=128),
    num_layers=st.integers(min_value=6, max_value=96)
)
def test_memory_estimation_properties(hidden_size, num_heads, num_layers):
    """Test properties that should always hold."""
    config = create_test_model_config(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers
    )

    memory = estimate_memory(config)

    # Properties that should always be true
    assert memory.total > 0
    assert memory.weights > 0
    assert memory.total >= memory.weights
    assert memory.activations >= 0
```

## Test Data Management

### Fixture Strategy
```python
@pytest.fixture
def small_cluster():
    return {"gpu_memory_gb": 24, "gpus_per_node": 4, "num_nodes": 1}

@pytest.fixture
def large_cluster():
    return {"gpu_memory_gb": 80, "gpus_per_node": 8, "num_nodes": 4}

@pytest.fixture
def llama_7b_config():
    return create_test_model_config("llama", "7B")
```

### Mock Strategy
Minimal mocking, focused on external dependencies:
```python
@patch('transformers.AutoConfig.from_pretrained')
def test_analyze_with_model_loading_error(mock_from_pretrained):
    """Test error handling when model loading fails."""
    mock_from_pretrained.side_effect = OSError("Model not found")

    with pytest.raises(ModelNotFoundError):
        analyze("invalid/model", {"gpu_memory_gb": 80})
```

## Performance Targets

- **Unit tests**: <1 second total runtime
- **Integration tests**: <30 seconds total runtime
- **Memory estimation**: <10ms per call
- **Full analysis**: <1 second for common models
- **Test coverage**: >90% line coverage

## Model Validation Matrix

### Complete Model Coverage for AutoParallel Validation

| Model Family | Model ID | Architecture | Precision | Experts | Context | Priority |
|--------------|----------|--------------|-----------|---------|---------|----------|
| **Qwen3 Dense** | Qwen/Qwen3-8B | Dense | bf16 | - | 40K | HIGH |
| **Qwen3 Dense** | Qwen/Qwen3-14B | Dense | bf16 | - | 40K | HIGH |
| **Qwen3 Dense** | Qwen/Qwen3-32B | Dense | bf16 | - | 40K | MEDIUM |
| **Qwen3 MoE** | Qwen/Qwen3-235B-A22B-Instruct-2507 | MoE | bf16 | 128/8 | 262K | HIGH |
| **Qwen3-Coder** | Qwen/Qwen3-Coder-480B-A35B-Instruct | MoE | bf16 | 160/8 | 262K | HIGH |
| **Llama-3.1** | meta-llama/Llama-3.1-8B-Instruct | Dense | bf16 | - | 128K | HIGH |
| **Llama-3.1** | meta-llama/Llama-3.1-70B-Instruct | Dense | bf16 | - | 128K | HIGH |
| **Llama-3.2** | meta-llama/Llama-3.2-3B-Instruct | Dense | bf16 | - | 8K | MEDIUM |
| **Llama-3.3** | meta-llama/Llama-3.3-70B-Instruct | Dense | bf16 | - | 128K | MEDIUM |
| **Llama-4** | meta-llama/Llama-4-Scout-17B-16E-Instruct | MoE | bf16 | 16/1 | 10M | HIGH |
| **Llama-4** | meta-llama/Llama-4-Maverick-17B-128E-Instruct | MoE | bf16 | 128/1 | 1M | HIGH |
| **DeepSeek-V3** | deepseek-ai/DeepSeek-V3 | MoE | fp8 | 256/8 | 128K | CRITICAL |
| **DeepSeek-R1** | deepseek-ai/DeepSeek-R1 | MoE | fp8 | 256/8 | 128K | CRITICAL |
| **Kimi-K2** | moonshotai/Kimi-K2-Instruct | MoE | fp8 | 384/8 | 128K | CRITICAL |

**Priority Levels:**
- **CRITICAL**: Must validate before release (fp8 models, extreme MoE)
- **HIGH**: Core validation models (representative of each family)
- **MEDIUM**: Extended validation for completeness

### Validation Test Scenarios

**Memory Estimation Accuracy (Target: <20% error)**
```python
def test_memory_estimation_accuracy_by_precision():
    """Test estimation accuracy grouped by precision format."""
    bf16_models = ["Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
    fp8_models = ["deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct"]

    # Test bf16 vs fp8 memory calculation differences

def test_moe_vs_dense_memory_scaling():
    """Compare MoE vs Dense memory scaling patterns."""
    dense_models = ["Qwen/Qwen3-32B", "meta-llama/Llama-3.1-70B-Instruct"]
    moe_models = ["Qwen/Qwen3-235B-A22B-Instruct-2507", "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
```

**Constraint Detection Accuracy**
```python
def test_tensor_parallel_constraints_by_family():
    """Test TP constraints across different attention head configurations."""

def test_expert_parallel_constraints_by_expert_count():
    """Test EP constraints for different expert counts (16, 128, 160, 256, 384)."""

def test_multimodal_constraints():
    """Test constraints for multimodal models (Llama-4 family)."""
```

**Context Length Handling**
```python
def test_extreme_context_lengths():
    """Test models with extreme context: Llama-4-Scout (10M), Llama-4-Maverick (1M)."""

def test_standard_vs_long_context_memory():
    """Compare memory usage between standard (8K) and long context (128K+) models."""
```

## Simplified Test Categories

1. **Core Logic Tests**: Memory estimation, constraints, search
2. **API Tests**: Public interface, error handling
3. **Integration Tests**: Real models with meta-device (14 models above)
4. **Property Tests**: Mathematical relationships
5. **Performance Tests**: Speed requirements

## Removed Complexity

- No mock object hierarchies
- No framework-specific test infrastructure
- No exhaustive model family testing in CI
- No complex validation framework testing
- No testing of unimplemented features

The simplified strategy focuses on testing the core behavior efficiently while maintaining confidence in the system's correctness.
