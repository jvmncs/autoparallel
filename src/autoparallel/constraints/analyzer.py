"""Architecture constraint analysis for model parallelization."""

from dataclasses import dataclass
from typing import Any

try:
    import transformers
    from transformers import PretrainedConfig
except ImportError:
    # Handle case where transformers is not available during testing
    transformers = None  # type: ignore
    PretrainedConfig = object  # type: ignore


@dataclass
class ParallelismConstraintParameters:
    """Configurable parameters for parallelism constraints."""

    # Default constraints
    default_min_layers_per_stage: int = 2
    default_max_tensor_parallel: int = 64
    min_experts_per_device: int = 1
    vocab_large_threshold: int = 100000
    vocab_medium_threshold: int = 50000
    vocab_large_divisibility: int = 8
    vocab_medium_divisibility: int = 4
    vocab_small_divisibility: int = 2

    def __post_init__(self) -> None:
        """Validate constraint parameters."""
        if self.default_min_layers_per_stage <= 0:
            raise ValueError(
                f"default_min_layers_per_stage must be positive, got {self.default_min_layers_per_stage}"  # noqa: E501
            )
        if self.default_max_tensor_parallel <= 0:
            raise ValueError(
                f"default_max_tensor_parallel must be positive, got {self.default_max_tensor_parallel}"  # noqa: E501
            )
        if self.min_experts_per_device <= 0:
            raise ValueError(
                f"min_experts_per_device must be positive, got {self.min_experts_per_device}"  # noqa: E501
            )
        if self.vocab_large_threshold <= 0:
            raise ValueError(
                f"vocab_large_threshold must be positive, got {self.vocab_large_threshold}"  # noqa: E501
            )
        if self.vocab_medium_threshold <= 0:
            raise ValueError(
                f"vocab_medium_threshold must be positive, got {self.vocab_medium_threshold}"  # noqa: E501
            )
        if self.vocab_large_divisibility <= 0:
            raise ValueError(
                f"vocab_large_divisibility must be positive, got {self.vocab_large_divisibility}"  # noqa: E501
            )
        if self.vocab_medium_divisibility <= 0:
            raise ValueError(
                f"vocab_medium_divisibility must be positive, got {self.vocab_medium_divisibility}"  # noqa: E501
            )
        if self.vocab_small_divisibility <= 0:
            raise ValueError(
                f"vocab_small_divisibility must be positive, got {self.vocab_small_divisibility}"  # noqa: E501
            )
        if self.vocab_medium_threshold >= self.vocab_large_threshold:
            raise ValueError(
                f"vocab_medium_threshold ({self.vocab_medium_threshold}) must be less than "  # noqa: E501
                f"vocab_large_threshold ({self.vocab_large_threshold})"
            )


@dataclass
class ModelConstraints:
    """Model architecture constraints that limit parallelization."""

    # Tensor parallelism constraints
    max_tensor_parallel: int
    """Maximum TP size based on architecture"""

    tensor_parallel_divisors: set[int]
    """Valid TP sizes"""

    # Expert parallelism constraints (for MoE)
    max_expert_parallel: int
    """Maximum EP size"""

    expert_parallel_divisors: set[int]
    """Valid EP sizes"""

    # Pipeline parallelism constraints
    max_pipeline_parallel: int
    """Maximum PP size (typically num_layers)"""

    min_layers_per_stage: int
    """Minimum layers per pipeline stage"""

    # Additional constraints
    requires_tied_embeddings: bool
    """Input/output embeddings are tied"""

    supports_grouped_query_attention: bool
    """Has GQA"""

    vocabulary_sharding: int
    """Vocab size divisibility for TP"""

    def __post_init__(self) -> None:
        """Validate model constraints."""
        if self.max_tensor_parallel < 0:
            raise ValueError(
                f"max_tensor_parallel must be non-negative, got {self.max_tensor_parallel}"  # noqa: E501
            )
        if self.max_expert_parallel < 0:
            raise ValueError(
                f"max_expert_parallel must be non-negative, got {self.max_expert_parallel}"  # noqa: E501
            )
        if self.max_pipeline_parallel < 0:
            raise ValueError(
                f"max_pipeline_parallel must be non-negative, got {self.max_pipeline_parallel}"  # noqa: E501
            )
        if self.min_layers_per_stage <= 0:
            raise ValueError(
                f"min_layers_per_stage must be positive, got {self.min_layers_per_stage}"  # noqa: E501
            )
        if self.vocabulary_sharding <= 0:
            raise ValueError(
                f"vocabulary_sharding must be positive, got {self.vocabulary_sharding}"  # noqa: E501
            )
        # Allow empty sets temporarily during debugging
        # if not self.tensor_parallel_divisors:
        #     raise ValueError("tensor_parallel_divisors cannot be empty")
        # if not self.expert_parallel_divisors:
        #     raise ValueError("expert_parallel_divisors cannot be empty")

        # Validate that divisors are reasonable
        if self.tensor_parallel_divisors and any(
            d <= 0 for d in self.tensor_parallel_divisors
        ):
            raise ValueError("All tensor_parallel_divisors must be positive")
        if self.expert_parallel_divisors and any(
            d <= 0 for d in self.expert_parallel_divisors
        ):
            raise ValueError("All expert_parallel_divisors must be positive")

    def get_valid_tensor_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid tensor parallel sizes up to max_gpus."""
        valid_sizes = []
        for size in self.tensor_parallel_divisors:
            if size <= min(max_gpus, self.max_tensor_parallel):
                valid_sizes.append(size)
        return sorted(valid_sizes)

    def get_valid_expert_parallel_sizes(self, max_gpus: int) -> list[int]:
        """Get valid expert parallel sizes up to max_gpus."""
        if self.max_expert_parallel == 0:
            return [1]  # Not an MoE model

        valid_sizes = []
        for size in self.expert_parallel_divisors:
            if size <= min(max_gpus, self.max_expert_parallel):
                valid_sizes.append(size)
        return sorted(valid_sizes)

    def get_valid_pipeline_parallel_sizes(self, max_nodes: int) -> list[int]:
        """Get valid pipeline parallel sizes up to max_nodes."""
        max_pp = min(max_nodes, self.max_pipeline_parallel)
        valid_sizes = []

        for size in range(1, max_pp + 1):
            layers_per_stage = self.max_pipeline_parallel / size
            if layers_per_stage >= self.min_layers_per_stage:
                valid_sizes.append(size)

        return valid_sizes


def analyze_model_constraints(
    config: PretrainedConfig,
    constraint_params: ParallelismConstraintParameters | None = None,
) -> ModelConstraints:
    """Analyze model architecture to determine parallelization constraints.

    Args:
        config: Transformers model configuration
        constraint_params: Configurable constraint parameters

    Returns:
        ModelConstraints with all parallelization limits
    """
    if constraint_params is None:
        constraint_params = ParallelismConstraintParameters()

    # Extract basic architecture parameters with defaults for robustness
    hidden_size = getattr(config, "hidden_size", 768)
    num_attention_heads = getattr(config, "num_attention_heads", 12)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    num_layers = getattr(config, "num_hidden_layers", 12)
    vocab_size = getattr(config, "vocab_size", 50257)
    intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)

    # Tensor parallelism constraints
    tp_constraints = _analyze_tensor_parallel_constraints(
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        vocab_size,
        intermediate_size,
        constraint_params,
    )

    # Expert parallelism constraints (MoE specific)
    ep_constraints = _analyze_expert_parallel_constraints(config, constraint_params)

    # Pipeline parallelism constraints
    pp_constraints = _analyze_pipeline_parallel_constraints(
        num_layers, constraint_params
    )

    # Additional architectural features
    tied_embeddings = _check_tied_embeddings(config)
    gqa_support = num_key_value_heads != num_attention_heads
    vocab_divisibility = _determine_vocab_divisibility_requirement(
        vocab_size, constraint_params
    )

    return ModelConstraints(
        max_tensor_parallel=tp_constraints["max_size"],
        tensor_parallel_divisors=tp_constraints["valid_sizes"],
        max_expert_parallel=ep_constraints["max_size"],
        expert_parallel_divisors=ep_constraints["valid_sizes"],
        max_pipeline_parallel=pp_constraints["max_size"],
        min_layers_per_stage=pp_constraints["min_layers_per_stage"],
        requires_tied_embeddings=tied_embeddings,
        supports_grouped_query_attention=gqa_support,
        vocabulary_sharding=vocab_divisibility,
    )


def _analyze_tensor_parallel_constraints(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    vocab_size: int,
    intermediate_size: int,
    constraint_params: ParallelismConstraintParameters,
) -> dict[str, Any]:
    """Analyze tensor parallelism constraints based on model dimensions."""

    # Key constraint: attention heads must be divisible by TP size
    attention_head_constraint = num_attention_heads

    # For GQA models, KV heads are the limiting factor
    kv_head_constraint = num_key_value_heads

    # Hidden size should be divisible (for efficient sharding)
    hidden_size_divisors = _get_divisors(hidden_size)

    # Intermediate size constraint (for MLP sharding)
    intermediate_divisors = _get_divisors(intermediate_size)

    # Vocab size constraint (for embedding sharding)
    vocab_divisors = _get_efficient_divisors(
        vocab_size, max_divisor=constraint_params.default_max_tensor_parallel
    )

    # Find intersection of all constraints
    valid_tp_sizes = set(range(1, attention_head_constraint + 1))
    valid_tp_sizes &= set(range(1, kv_head_constraint + 1))
    valid_tp_sizes &= set(hidden_size_divisors)
    valid_tp_sizes &= set(intermediate_divisors)
    valid_tp_sizes &= set(vocab_divisors)

    # Ensure we always have at least TP=1 as a fallback
    if not valid_tp_sizes:
        valid_tp_sizes = {1}

    # Practical maximum (configurable)
    practical_max = min(
        constraint_params.default_max_tensor_parallel,
        max(valid_tp_sizes) if valid_tp_sizes else 1,
    )
    valid_tp_sizes = {size for size in valid_tp_sizes if size <= practical_max}

    # Ensure we always have at least TP=1
    if not valid_tp_sizes:
        valid_tp_sizes = {1}
        practical_max = 1

    return {"max_size": practical_max, "valid_sizes": valid_tp_sizes}


def _analyze_expert_parallel_constraints(
    config: PretrainedConfig, constraint_params: ParallelismConstraintParameters
) -> dict[str, Any]:
    """Analyze expert parallelism constraints for MoE models."""

    # Check if this is an MoE model
    num_experts = getattr(
        config, "num_local_experts", getattr(config, "num_experts", 0)
    )

    if num_experts == 0:
        return {"max_size": 0, "valid_sizes": {1}}

    # Expert parallelism: experts must be distributable across devices
    expert_divisors = _get_divisors(num_experts)

    # Practical constraint: configurable minimum experts per device
    max_ep_size = num_experts // constraint_params.min_experts_per_device

    valid_ep_sizes = {size for size in expert_divisors if size <= max_ep_size}

    return {"max_size": max_ep_size, "valid_sizes": valid_ep_sizes}


def _analyze_pipeline_parallel_constraints(
    num_layers: int, constraint_params: ParallelismConstraintParameters
) -> dict[str, Any]:
    """Analyze pipeline parallelism constraints."""

    # Pipeline parallelism: distribute layers across stages
    # Minimum layers per stage for efficiency (configurable)
    max_pp_size = num_layers // constraint_params.default_min_layers_per_stage

    return {
        "max_size": max_pp_size,
        "min_layers_per_stage": constraint_params.default_min_layers_per_stage,
    }


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
    """Get divisors that are powers of 2 or have small prime factors."""
    all_divisors = _get_divisors(n, max_divisor)

    # Prefer powers of 2 and numbers with small prime factors
    efficient_divisors = []
    for d in all_divisors:
        if d <= max_divisor and _is_efficient_divisor(d):
            efficient_divisors.append(d)

    return efficient_divisors


def _is_efficient_divisor(n: int) -> bool:
    """Check if a number is an efficient divisor (power of 2 or small primes)."""
    if n & (n - 1) == 0:  # Power of 2
        return True

    # Check if all prime factors are small (≤ 7)
    temp = n
    for prime in [2, 3, 5, 7]:
        while temp % prime == 0:
            temp //= prime

    return temp == 1


def _check_tied_embeddings(config: PretrainedConfig) -> bool:
    """Check if model has tied input/output embeddings."""
    return getattr(config, "tie_word_embeddings", False)


def _determine_vocab_divisibility_requirement(
    vocab_size: int, constraint_params: ParallelismConstraintParameters
) -> int:
    """Determine vocabulary divisibility requirement for efficient sharding."""
    # For large vocabularies, require divisibility by larger numbers for efficiency
    if vocab_size >= constraint_params.vocab_large_threshold:
        return constraint_params.vocab_large_divisibility
    elif vocab_size >= constraint_params.vocab_medium_threshold:
        return constraint_params.vocab_medium_divisibility
    else:
        return constraint_params.vocab_small_divisibility
