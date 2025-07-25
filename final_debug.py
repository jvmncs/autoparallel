#!/usr/bin/env python3

from autoparallel.memory import estimate_memory

# Exact same config
config_70b = {
    "vocab_size": 32000,
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "intermediate_size": 28672,
}

print("Testing with exact same config:")

# Call 1: From scratch
memory1 = estimate_memory(
    model_config=config_70b,
    sequence_length=2048,
    batch_size=1,
    quantization_bytes=2,
)
print(f"Call 1 - Weights: {memory1.weights / (1024**3):.1f}GB")

# Call 2: Same parameters
memory2 = estimate_memory(
    model_config=config_70b.copy(),  # Fresh copy
    sequence_length=2048,
    batch_size=1,
    quantization_bytes=2,
)
print(f"Call 2 - Weights: {memory2.weights / (1024**3):.1f}GB")

# Let's try a different config to see if it works
config_7b = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 11008,
}

memory3 = estimate_memory(
    model_config=config_7b,
    sequence_length=2048,
    batch_size=1,
    quantization_bytes=2,
)
print(f"7B model - Weights: {memory3.weights / (1024**3):.1f}GB")

# Direct parameter calculation for comparison
from autoparallel.memory import _estimate_param_count

param_count_70b = _estimate_param_count(
    vocab_size=32000,
    hidden_size=8192,
    num_layers=80,
    intermediate_size=28672,
    num_experts=0,
)
print(f"Direct 70B params: {param_count_70b:,} = {param_count_70b * 2 / (1024**3):.1f}GB")

param_count_7b = _estimate_param_count(
    vocab_size=32000,
    hidden_size=4096,
    num_layers=32,
    intermediate_size=11008,
    num_experts=0,
)
print(f"Direct 7B params: {param_count_7b:,} = {param_count_7b * 2 / (1024**3):.1f}GB")
