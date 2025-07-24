"""Model-specific support utilities for autoparallel constraints.

This module provides architecture-specific constraint calculation logic, model family
detection, and constraint mapping for various transformer architectures including
dense transformers, MoE models, multimodal models, and long context models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import transformers
    from transformers import PretrainedConfig
except ImportError:
    # Handle case where transformers is not available during testing
    transformers = None  # type: ignore
    PretrainedConfig = object  # type: ignore


class ModelArchitectureType(Enum):
    """Model architecture types for constraint specialization."""

    DENSE_TRANSFORMER = "dense_transformer"
    MOE_TRANSFORMER = "moe_transformer"
    MULTIMODAL = "multimodal"
    LONG_CONTEXT = "long_context"
    STATE_SPACE = "state_space"
    UNKNOWN = "unknown"


class QuantizationType(Enum):
    """Quantization format types."""

    FULL_PRECISION = "full_precision"  # fp32/bf16/fp16
    GPTQ = "gptq"
    AWQ = "awq"
    BITSANDBYTES = "bitsandbytes"
    FP8 = "fp8"
    INT8 = "int8"
    GGUF = "gguf"
    UNKNOWN = "unknown"


@dataclass
class ModelArchitectureInfo:
    """Comprehensive model architecture information."""

    # Basic architecture parameters
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_layers: int
    vocab_size: int
    intermediate_size: int

    # Architecture type classification
    architecture_type: ModelArchitectureType
    model_type: str  # From config.model_type

    # MoE-specific parameters
    num_experts: int = 0
    num_experts_per_token: int = 0
    expert_capacity_factor: float = 1.0

    # Multimodal parameters
    vision_config: dict[str, Any] | None = None
    audio_config: dict[str, Any] | None = None

    # Long context parameters
    rope_scaling: dict[str, Any] | None = None
    max_position_embeddings: int = 2048

    # Quantization information
    quantization_type: QuantizationType = QuantizationType.FULL_PRECISION
    quantization_config: dict[str, Any] | None = None

    # Additional architectural features
    has_tied_embeddings: bool = False
    supports_flash_attention: bool = False
    supports_sliding_window: bool = False
    sliding_window_size: int | None = None

    # Precision information
    torch_dtype: str = "float16"
    bits_per_parameter: float = 16.0


@dataclass
class ArchitectureConstraints:
    """Architecture-specific parallelization constraints."""

    # Tensor parallelism constraints
    max_tensor_parallel: int
    tensor_parallel_divisors: set[int]

    # Expert parallelism constraints (for MoE)
    max_expert_parallel: int
    expert_parallel_divisors: set[int]

    # Pipeline parallelism constraints
    max_pipeline_parallel: int
    min_layers_per_stage: int

    # Additional constraints
    requires_tied_embeddings: bool
    supports_grouped_query_attention: bool
    vocab_divisibility_requirement: int

    # Architecture-specific optimizations
    preferred_attention_head_divisors: set[int]
    supports_sequence_parallel: bool = False
    requires_expert_load_balancing: bool = False

    def get_valid_tensor_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid tensor parallel sizes up to max_gpus."""
        valid_sizes = []
        for size in sorted(self.tensor_parallel_divisors):
            if size <= min(max_gpus, self.max_tensor_parallel):
                valid_sizes.append(size)
        return valid_sizes

    def get_valid_expert_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid expert parallel sizes up to max_gpus."""
        if self.max_expert_parallel == 0:
            return [1]  # Not an MoE model

        valid_sizes = []
        for size in sorted(self.expert_parallel_divisors):
            if size <= min(max_gpus, self.max_expert_parallel):
                valid_sizes.append(size)
        return valid_sizes

    def get_valid_pipeline_parallel_sizes(self, max_stages: int) -> list[int]:
        """Get valid pipeline parallel sizes up to max_stages."""
        max_pp = min(max_stages, self.max_pipeline_parallel)
        valid_sizes = []

        for size in range(1, max_pp + 1):
            layers_per_stage = self.max_pipeline_parallel / size
            if layers_per_stage >= self.min_layers_per_stage:
                valid_sizes.append(size)

        return valid_sizes


def detect_model_architecture(config: PretrainedConfig) -> ModelArchitectureInfo:
    """Detect model architecture type and extract relevant parameters.

    Args:
        config: Transformers model configuration

    Returns:
        ModelArchitectureInfo with detected architecture details
    """
    if transformers is None:
        raise ImportError(
            "transformers library is required for model architecture detection"
        )

    # Extract basic parameters
    model_type = getattr(config, "model_type", "unknown")
    hidden_size = getattr(config, "hidden_size", 0)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    num_layers = getattr(config, "num_hidden_layers", 0)
    vocab_size = getattr(config, "vocab_size", 50257)
    intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)

    # Detect architecture type
    architecture_type = _classify_architecture_type(config)

    # Extract MoE parameters
    moe_info = _extract_moe_parameters(config)

    # Extract multimodal parameters
    multimodal_info = _extract_multimodal_parameters(config)

    # Extract long context parameters
    long_context_info = _extract_long_context_parameters(config)

    # Detect quantization
    quantization_info = _detect_quantization(config)

    # Extract additional features
    additional_features = _extract_additional_features(config)

    return ModelArchitectureInfo(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        architecture_type=architecture_type,
        model_type=model_type,
        num_experts=moe_info["num_experts"],
        num_experts_per_token=moe_info["num_experts_per_token"],
        expert_capacity_factor=moe_info["expert_capacity_factor"],
        vision_config=multimodal_info.get("vision_config"),
        audio_config=multimodal_info.get("audio_config"),
        rope_scaling=long_context_info.get("rope_scaling"),
        max_position_embeddings=long_context_info.get("max_position_embeddings", 2048),
        quantization_type=quantization_info["type"],
        quantization_config=quantization_info["config"],
        has_tied_embeddings=additional_features["has_tied_embeddings"],
        supports_flash_attention=additional_features["supports_flash_attention"],
        supports_sliding_window=additional_features["supports_sliding_window"],
        sliding_window_size=additional_features["sliding_window_size"],
        torch_dtype=str(getattr(config, "torch_dtype", "float16")),
        bits_per_parameter=quantization_info["bits_per_parameter"],
    )


def calculate_architecture_constraints(
    arch_info: ModelArchitectureInfo,
    min_layers_per_stage: int = 2,
    max_tensor_parallel: int = 64,
    min_experts_per_device: int = 1,
) -> ArchitectureConstraints:
    """Calculate architecture-specific parallelization constraints.

    Args:
        arch_info: Model architecture information
        min_layers_per_stage: Minimum layers per pipeline stage
        max_tensor_parallel: Maximum tensor parallel size
        min_experts_per_device: Minimum experts per device for MoE

    Returns:
        ArchitectureConstraints with valid parallel configurations
    """
    # Tensor parallelism constraints
    tp_constraints = _calculate_tensor_parallel_constraints(
        arch_info, max_tensor_parallel
    )

    # Expert parallelism constraints
    ep_constraints = _calculate_expert_parallel_constraints(
        arch_info, min_experts_per_device
    )

    # Pipeline parallelism constraints
    pp_constraints = _calculate_pipeline_parallel_constraints(
        arch_info, min_layers_per_stage
    )

    # Additional constraint analysis
    vocab_divisibility = _determine_vocab_divisibility_requirement(arch_info.vocab_size)
    preferred_head_divisors = _get_preferred_attention_head_divisors(
        arch_info.num_attention_heads
    )

    return ArchitectureConstraints(
        max_tensor_parallel=tp_constraints["max_size"],
        tensor_parallel_divisors=tp_constraints["valid_sizes"],
        max_expert_parallel=ep_constraints["max_size"],
        expert_parallel_divisors=ep_constraints["valid_sizes"],
        max_pipeline_parallel=pp_constraints["max_size"],
        min_layers_per_stage=pp_constraints["min_layers_per_stage"],
        requires_tied_embeddings=arch_info.has_tied_embeddings,
        supports_grouped_query_attention=(
            arch_info.num_key_value_heads != arch_info.num_attention_heads
        ),
        vocab_divisibility_requirement=vocab_divisibility,
        preferred_attention_head_divisors=preferred_head_divisors,
        supports_sequence_parallel=_supports_sequence_parallel(arch_info),
        requires_expert_load_balancing=(
            arch_info.architecture_type == ModelArchitectureType.MOE_TRANSFORMER
        ),
    )


def get_model_family_constraints(model_name: str) -> dict[str, Any]:
    """Get model family specific constraints and optimizations.

    Args:
        model_name: Hugging Face model identifier

    Returns:
        Dictionary of model family specific constraints
    """
    model_name_lower = model_name.lower()

    # Llama family
    if any(x in model_name_lower for x in ["llama", "alpaca", "vicuna", "code-llama"]):
        return {
            "family": "llama",
            "supports_flash_attention": True,
            "preferred_tp_sizes": [1, 2, 4, 8],
            "max_context_length": 32768 if "code-llama" in model_name_lower else 4096,
            "rope_scaling_support": True,
        }

    # Mixtral family (MoE)
    elif "mixtral" in model_name_lower:
        return {
            "family": "mixtral",
            "supports_flash_attention": True,
            "preferred_tp_sizes": [1, 2, 4, 8],
            "preferred_ep_sizes": [1, 2, 4, 8],
            "expert_routing": "switch",
            "load_balancing_required": True,
        }

    # GPT family
    elif any(x in model_name_lower for x in ["gpt", "openai"]):
        return {
            "family": "gpt",
            "supports_flash_attention": False,
            "preferred_tp_sizes": [1, 2, 4, 8, 16],
            "max_context_length": 2048,
        }

    # T5 family
    elif "t5" in model_name_lower:
        return {
            "family": "t5",
            "encoder_decoder": True,
            "supports_flash_attention": False,
            "preferred_tp_sizes": [1, 2, 4, 8],
        }

    # Falcon family
    elif "falcon" in model_name_lower:
        return {
            "family": "falcon",
            "supports_flash_attention": True,
            "multiquery_attention": True,
            "preferred_tp_sizes": [1, 2, 4, 8],
        }

    # MPT family
    elif "mpt" in model_name_lower:
        return {
            "family": "mpt",
            "supports_flash_attention": True,
            "alibi_attention": True,
            "preferred_tp_sizes": [1, 2, 4, 8],
        }

    # DeepSeek MoE
    elif "deepseek" in model_name_lower and "moe" in model_name_lower:
        return {
            "family": "deepseek",
            "supports_flash_attention": True,
            "preferred_tp_sizes": [1, 2, 4, 8],
            "preferred_ep_sizes": [1, 2, 4, 8, 16, 64],
            "shared_expert_support": True,
        }

    # Multimodal models
    elif any(x in model_name_lower for x in ["llava", "instructblip", "flamingo"]):
        return {
            "family": "multimodal",
            "vision_encoder": True,
            "supports_flash_attention": True,
            "preferred_tp_sizes": [1, 2, 4, 8],
            "vision_tp_constraints": True,
        }

    # Default constraints
    else:
        return {
            "family": "unknown",
            "supports_flash_attention": False,
            "preferred_tp_sizes": [1, 2, 4, 8],
        }


def get_quantization_memory_multiplier(quantization_type: QuantizationType) -> float:
    """Get memory multiplier for different quantization formats.

    Args:
        quantization_type: Type of quantization

    Returns:
        Memory multiplier relative to full precision
    """
    multipliers = {
        QuantizationType.FULL_PRECISION: 1.0,
        QuantizationType.GPTQ: 0.25,  # 4-bit
        QuantizationType.AWQ: 0.25,  # 4-bit
        QuantizationType.BITSANDBYTES: 0.5,  # 8-bit typical
        QuantizationType.FP8: 0.5,
        QuantizationType.INT8: 0.5,
        QuantizationType.GGUF: 0.3,  # Variable, estimate
        QuantizationType.UNKNOWN: 1.0,
    }

    return multipliers.get(quantization_type, 1.0)


def _classify_architecture_type(config: PretrainedConfig) -> ModelArchitectureType:
    """Classify model architecture type from config."""
    model_type = getattr(config, "model_type", "").lower()

    # Check for MoE models
    if _is_moe_model(config):
        return ModelArchitectureType.MOE_TRANSFORMER

    # Check for multimodal models
    if _is_multimodal_model(config):
        return ModelArchitectureType.MULTIMODAL

    # Check for long context models
    if _is_long_context_model(config):
        return ModelArchitectureType.LONG_CONTEXT

    # Check for state space models
    if model_type in ["mamba", "state_space"]:
        return ModelArchitectureType.STATE_SPACE

    # Default to dense transformer
    if model_type in ["llama", "gpt2", "gpt_neox", "falcon", "mpt", "bloom", "opt"]:
        return ModelArchitectureType.DENSE_TRANSFORMER

    return ModelArchitectureType.UNKNOWN


def _is_moe_model(config: PretrainedConfig) -> bool:
    """Check if model is a Mixture of Experts model."""
    # Check common MoE indicators
    moe_indicators = [
        "num_local_experts",
        "num_experts",
        "expert_capacity_factor",
        "router_aux_loss_coef",
        "experts_per_token",
    ]

    return any(
        hasattr(config, attr) and getattr(config, attr, 0) > 0
        for attr in moe_indicators
    )


def _is_multimodal_model(config: PretrainedConfig) -> bool:
    """Check if model is multimodal."""
    multimodal_indicators = [
        "vision_config",
        "audio_config",
        "image_processor",
        "vision_tower",
    ]

    return any(hasattr(config, attr) for attr in multimodal_indicators)


def _is_long_context_model(config: PretrainedConfig) -> bool:
    """Check if model is optimized for long context."""
    max_pos = getattr(config, "max_position_embeddings", 2048)
    rope_scaling = getattr(config, "rope_scaling", None)

    return max_pos > 8192 or rope_scaling is not None


def _extract_moe_parameters(config: PretrainedConfig) -> dict[str, Any]:
    """Extract MoE-specific parameters."""
    return {
        "num_experts": getattr(
            config, "num_local_experts", getattr(config, "num_experts", 0)
        ),
        "num_experts_per_token": getattr(
            config, "num_experts_per_tok", getattr(config, "top_k", 0)
        ),
        "expert_capacity_factor": getattr(config, "expert_capacity_factor", 1.0),
    }


def _extract_multimodal_parameters(config: PretrainedConfig) -> dict[str, Any]:
    """Extract multimodal-specific parameters."""
    return {
        "vision_config": getattr(config, "vision_config", None),
        "audio_config": getattr(config, "audio_config", None),
    }


def _extract_long_context_parameters(config: PretrainedConfig) -> dict[str, Any]:
    """Extract long context-specific parameters."""
    return {
        "rope_scaling": getattr(config, "rope_scaling", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
    }


def _detect_quantization(config: PretrainedConfig) -> dict[str, Any]:
    """Detect quantization format and configuration."""
    # Check for quantization config
    quant_config = getattr(config, "quantization_config", None)

    if quant_config is not None:
        quant_method = getattr(quant_config, "quant_method", "").lower()

        if "gptq" in quant_method:
            bits = getattr(quant_config, "bits", 4)
            return {
                "type": QuantizationType.GPTQ,
                "config": quant_config,
                "bits_per_parameter": float(bits),
            }
        elif "awq" in quant_method:
            bits = getattr(quant_config, "w_bit", 4)
            return {
                "type": QuantizationType.AWQ,
                "config": quant_config,
                "bits_per_parameter": float(bits),
            }
        elif "bitsandbytes" in quant_method:
            load_in_8bit = getattr(quant_config, "load_in_8bit", False)
            load_in_4bit = getattr(quant_config, "load_in_4bit", False)
            bits = 4 if load_in_4bit else (8 if load_in_8bit else 16)
            return {
                "type": QuantizationType.BITSANDBYTES,
                "config": quant_config,
                "bits_per_parameter": float(bits),
            }

    # Check torch_dtype for precision
    torch_dtype = getattr(config, "torch_dtype", "float16")
    if "float32" in str(torch_dtype):
        bits = 32.0
    elif "bfloat16" in str(torch_dtype) or "float16" in str(torch_dtype):
        bits = 16.0
    else:
        bits = 16.0

    return {
        "type": QuantizationType.FULL_PRECISION,
        "config": None,
        "bits_per_parameter": bits,
    }


def _extract_additional_features(config: PretrainedConfig) -> dict[str, Any]:
    """Extract additional architectural features."""
    return {
        "has_tied_embeddings": getattr(config, "tie_word_embeddings", False),
        "supports_flash_attention": _supports_flash_attention(config),
        "supports_sliding_window": hasattr(config, "sliding_window"),
        "sliding_window_size": getattr(config, "sliding_window", None),
    }


def _supports_flash_attention(config: PretrainedConfig) -> bool:
    """Check if model architecture supports FlashAttention."""
    model_type = getattr(config, "model_type", "").lower()

    # Models known to support FlashAttention
    flash_attention_models = [
        "llama",
        "falcon",
        "mpt",
        "mixtral",
        "qwen",
        "baichuan",
        "internlm",
    ]

    return any(model in model_type for model in flash_attention_models)


def _calculate_tensor_parallel_constraints(
    arch_info: ModelArchitectureInfo, max_tensor_parallel: int
) -> dict[str, Any]:
    """Calculate tensor parallelism constraints."""
    # Key constraint: attention heads must be divisible by TP size
    attention_head_constraint = arch_info.num_attention_heads

    # For GQA models, KV heads are the limiting factor
    kv_head_constraint = arch_info.num_key_value_heads

    # Get divisors for various dimensions
    hidden_divisors = _get_efficient_divisors(
        arch_info.hidden_size, max_tensor_parallel
    )
    intermediate_divisors = _get_efficient_divisors(
        arch_info.intermediate_size, max_tensor_parallel
    )
    vocab_divisors = _get_efficient_divisors(arch_info.vocab_size, max_tensor_parallel)

    # Find intersection of all constraints
    valid_tp_sizes = set(
        range(1, min(attention_head_constraint, kv_head_constraint) + 1)
    )
    valid_tp_sizes &= set(hidden_divisors)
    valid_tp_sizes &= set(intermediate_divisors)
    valid_tp_sizes &= set(vocab_divisors)

    # Apply practical maximum
    practical_max = min(
        max_tensor_parallel, max(valid_tp_sizes) if valid_tp_sizes else 1
    )
    valid_tp_sizes = {size for size in valid_tp_sizes if size <= practical_max}

    return {
        "max_size": practical_max,
        "valid_sizes": valid_tp_sizes,
    }


def _calculate_expert_parallel_constraints(
    arch_info: ModelArchitectureInfo, min_experts_per_device: int
) -> dict[str, Any]:
    """Calculate expert parallelism constraints for MoE models."""
    if arch_info.num_experts == 0:
        return {"max_size": 0, "valid_sizes": {1}}

    # Expert parallelism: experts must be distributable across devices
    expert_divisors = _get_divisors(arch_info.num_experts)

    # Practical constraint: minimum experts per device
    max_ep_size = arch_info.num_experts // min_experts_per_device

    valid_ep_sizes = {size for size in expert_divisors if size <= max_ep_size}

    return {
        "max_size": max_ep_size,
        "valid_sizes": valid_ep_sizes,
    }


def _calculate_pipeline_parallel_constraints(
    arch_info: ModelArchitectureInfo, min_layers_per_stage: int
) -> dict[str, Any]:
    """Calculate pipeline parallelism constraints."""
    # Pipeline parallelism: distribute layers across stages
    max_pp_size = arch_info.num_layers // min_layers_per_stage

    return {
        "max_size": max_pp_size,
        "min_layers_per_stage": min_layers_per_stage,
    }


def _determine_vocab_divisibility_requirement(vocab_size: int) -> int:
    """Determine vocabulary divisibility requirement for efficient sharding."""
    if vocab_size >= 100000:
        return 8  # Large vocabularies
    elif vocab_size >= 50000:
        return 4  # Medium vocabularies
    else:
        return 2  # Small vocabularies


def _get_preferred_attention_head_divisors(num_heads: int) -> set[int]:
    """Get preferred divisors for attention heads."""
    divisors = _get_divisors(num_heads)

    # Prefer powers of 2 and small divisors
    preferred = set()
    for d in divisors:
        if d <= 64 and _is_power_of_2_or_small_factors(d):
            preferred.add(d)

    return preferred


def _supports_sequence_parallel(arch_info: ModelArchitectureInfo) -> bool:
    """Check if architecture supports sequence parallelism."""
    # Sequence parallelism is supported by most modern transformer architectures
    supported_types = [
        ModelArchitectureType.DENSE_TRANSFORMER,
        ModelArchitectureType.MOE_TRANSFORMER,
        ModelArchitectureType.LONG_CONTEXT,
    ]

    return arch_info.architecture_type in supported_types


def _get_divisors(n: int, max_divisor: int | None = None) -> list[int]:
    """Get all divisors of n up to max_divisor."""
    if max_divisor is None:
        max_divisor = n

    divisors = []
    for i in range(1, min(int(n**0.5) + 1, max_divisor + 1)):
        if n % i == 0:
            divisors.append(i)
            if i != n // i and n // i <= max_divisor:
                divisors.append(n // i)

    return sorted(divisors)


def _get_efficient_divisors(n: int, max_divisor: int = 64) -> list[int]:
    """Get divisors that are efficient for parallel computation."""
    all_divisors = _get_divisors(n, max_divisor)

    # Prefer powers of 2 and numbers with small prime factors
    efficient_divisors = []
    for d in all_divisors:
        if d <= max_divisor and _is_power_of_2_or_small_factors(d):
            efficient_divisors.append(d)

    return efficient_divisors


def _is_power_of_2_or_small_factors(n: int) -> bool:
    """Check if number is power of 2 or has only small prime factors."""
    if n & (n - 1) == 0:  # Power of 2
        return True

    # Check if all prime factors are small (≤ 7)
    temp = n
    for prime in [2, 3, 5, 7]:
        while temp % prime == 0:
            temp //= prime

    return temp == 1
