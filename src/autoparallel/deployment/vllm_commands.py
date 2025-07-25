"""vLLM deployment command generation with integrated optimization."""

import logging
from dataclasses import dataclass
from typing import Any

from autoparallel.config.optimizer import (
    HardwareProfile,
    ParallelismConfiguration,
    WorkloadProfile,
)
from autoparallel.frameworks.vllm_memory import WorkloadProfile as VLLMWorkloadProfile
from autoparallel.frameworks.vllm_optimizer import GPUArchitecture, VLLMOptimizer

logger = logging.getLogger(__name__)


@dataclass
class VLLMDeploymentOptions:
    """Options for vLLM deployment command generation."""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Runtime mode
    enable_offline_mode: bool = False
    serve_api: bool = True

    # Performance tuning
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = True
    disable_cuda_graphs: bool = False

    # Custom arguments
    custom_args: dict[str, Any] | None = None

    # Environment variables
    environment_vars: dict[str, str] | None = None


@dataclass
class VLLMDeploymentResult:
    """Result of vLLM deployment command generation."""

    command: str
    environment_vars: dict[str, str]
    configuration_summary: dict[str, Any]
    optimization_insights: list[str]
    estimated_performance: dict[str, float] | None = None


class VLLMCommandGenerator:
    """Generate optimized vLLM deployment commands."""

    def __init__(
        self,
        model_name: str,
        hardware_profile: HardwareProfile,
        gpu_architecture: GPUArchitecture = GPUArchitecture.GENERIC,
    ):
        """Initialize vLLM command generator.

        Args:
            model_name: Model name for deployment
            hardware_profile: Cluster hardware specifications
            gpu_architecture: GPU architecture for specific optimizations
        """
        self.model_name = model_name
        self.hardware_profile = hardware_profile
        self.gpu_architecture = gpu_architecture

        # Initialize vLLM optimizer
        self.optimizer = VLLMOptimizer(
            model_name=model_name,
            gpu_memory_capacity_gb=hardware_profile.gpu_memory_gb,
            gpu_architecture=gpu_architecture,
        )

    def generate_serving_command(
        self,
        workload: WorkloadProfile,
        parallelism_config: ParallelismConfiguration,
        options: VLLMDeploymentOptions | None = None,
    ) -> VLLMDeploymentResult:
        """Generate vLLM serving API command.

        Args:
            workload: Expected workload characteristics
            parallelism_config: Parallelism configuration
            options: Deployment options

        Returns:
            VLLMDeploymentResult with command and metadata
        """
        if options is None:
            options = VLLMDeploymentOptions()

        # Convert workload to vLLM format
        vllm_workload = self._convert_workload(workload)

        # Optimize configuration for workload
        optimization_result = self.optimizer.optimize_for_workload(
            vllm_workload,
            optimization_objective="auto",
        )

        if not optimization_result.is_successful:
            raise ValueError(
                f"Failed to find optimal configuration for {self.model_name}"
            )

        # Generate base command parts
        cmd_parts = self._build_base_command(options)

        # Add core configuration
        cmd_parts.extend(self._build_core_config(optimization_result.optimal_config))

        # Add parallelism configuration
        cmd_parts.extend(self._build_parallelism_config(parallelism_config))

        # Add optimization-specific parameters
        cmd_parts.extend(
            self._build_optimization_config(
                optimization_result.optimal_config, vllm_workload, options
            )
        )

        # Add custom arguments
        if options.custom_args:
            cmd_parts.extend(self._build_custom_args(options.custom_args))

        # Build environment variables
        env_vars = self._build_environment_vars(parallelism_config, options)

        # Format final command
        command = self._format_command(cmd_parts)

        # Generate configuration summary
        config_summary = self._build_config_summary(
            optimization_result, parallelism_config, options
        )

        return VLLMDeploymentResult(
            command=command,
            environment_vars=env_vars,
            configuration_summary=config_summary,
            optimization_insights=optimization_result.recommendations,
            estimated_performance=self._extract_performance_estimates(
                optimization_result
            ),
        )

    def generate_offline_command(
        self,
        input_file: str,
        output_file: str,
        parallelism_config: ParallelismConfiguration,
        batch_size: int = 1000,
        options: VLLMDeploymentOptions | None = None,
    ) -> VLLMDeploymentResult:
        """Generate vLLM offline inference command.

        Args:
            input_file: Path to input file with prompts
            output_file: Path to output file for results
            parallelism_config: Parallelism configuration
            batch_size: Batch size for processing
            options: Deployment options

        Returns:
            VLLMDeploymentResult with command and metadata
        """
        if options is None:
            options = VLLMDeploymentOptions(enable_offline_mode=True, serve_api=False)

        # Create throughput-focused workload for offline processing
        offline_workload = VLLMWorkloadProfile(
            requests_per_second=10.0,  # Low rate for batch processing
            batch_size_distribution={batch_size: 1.0},
            sequence_length_distribution={1024: 1.0},  # Assume 1024 tokens
            target_metric="throughput",
            latency_budget_ms=1000.0,  # Relaxed latency for throughput
        )

        # Optimize for throughput
        optimization_result = self.optimizer.optimize_for_workload(
            offline_workload,
            optimization_objective="throughput",
        )

        if not optimization_result.is_successful:
            raise ValueError(
                f"Failed to find optimal configuration for offline {self.model_name}"
            )

        # Build offline-specific command
        cmd_parts = [
            "python -m vllm.entrypoints.offline_inference",
            f"--model {self.model_name}",
            f"--input {input_file}",
            f"--output {output_file}",
            f"--batch-size {batch_size}",
        ]

        # Add core configuration
        cmd_parts.extend(self._build_core_config(optimization_result.optimal_config))

        # Add parallelism configuration
        cmd_parts.extend(self._build_parallelism_config(parallelism_config))

        # Add optimization-specific parameters for offline
        cmd_parts.extend(
            self._build_optimization_config(
                optimization_result.optimal_config, offline_workload, options
            )
        )

        # Build environment variables
        env_vars = self._build_environment_vars(parallelism_config, options)

        # Format final command
        command = self._format_command(cmd_parts)

        # Generate configuration summary
        config_summary = self._build_config_summary(
            optimization_result, parallelism_config, options
        )

        return VLLMDeploymentResult(
            command=command,
            environment_vars=env_vars,
            configuration_summary=config_summary,
            optimization_insights=optimization_result.recommendations,
            estimated_performance=self._extract_performance_estimates(
                optimization_result
            ),
        )

    def _convert_workload(self, workload: WorkloadProfile) -> VLLMWorkloadProfile:
        """Convert general workload to vLLM workload format."""
        # Determine target metric based on workload type and latency budget
        if workload.workload_type.value in ("interactive", "chatbot"):
            target_metric = "latency"
        elif workload.workload_type.value == "inference":
            # For inference, use latency budget to decide
            target_metric = (
                "latency" if workload.latency_budget_ms < 100.0 else "throughput"
            )
        else:
            target_metric = "throughput"

        # Create batch size distribution based on workload batch size
        batch_size_distribution = {workload.batch_size: 1.0}

        # Create sequence length distribution based on workload sequence length
        sequence_length_distribution = {workload.sequence_length: 1.0}

        return VLLMWorkloadProfile(
            requests_per_second=workload.requests_per_second,
            batch_size_distribution=batch_size_distribution,
            sequence_length_distribution=sequence_length_distribution,
            target_metric=target_metric,
            latency_budget_ms=workload.latency_budget_ms,
        )

    def _build_base_command(self, options: VLLMDeploymentOptions) -> list[str]:
        """Build base command parts."""
        if options.enable_offline_mode:
            return ["python -m vllm.entrypoints.offline_inference"]
        else:
            cmd_parts = [
                "python -m vllm.entrypoints.openai.api_server",
                f"--model {self.model_name}",
                f"--host {options.host}",
                f"--port {options.port}",
            ]
            return cmd_parts

    def _build_core_config(self, config) -> list[str]:
        """Build core configuration parameters."""
        cmd_parts = [
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
            f"--max-model-len {config.max_model_len}",
            f"--max-num-seqs {config.calculate_effective_batch_size()}",
        ]

        if config.kv_cache_dtype != "auto":
            cmd_parts.append(f"--kv-cache-dtype {config.kv_cache_dtype}")

        return cmd_parts

    def _build_parallelism_config(
        self, parallelism_config: ParallelismConfiguration
    ) -> list[str]:
        """Build parallelism configuration parameters."""
        cmd_parts = []

        if parallelism_config.tensor_parallel_size > 1:
            cmd_parts.append(
                f"--tensor-parallel-size {parallelism_config.tensor_parallel_size}"
            )

        if parallelism_config.pipeline_parallel_size > 1:
            cmd_parts.append(
                f"--pipeline-parallel-size {parallelism_config.pipeline_parallel_size}"
            )

        return cmd_parts

    def _build_optimization_config(
        self, config, workload: VLLMWorkloadProfile, options: VLLMDeploymentOptions
    ) -> list[str]:
        """Build optimization-specific configuration parameters."""
        cmd_parts = []

        # CUDA graphs configuration
        if options.disable_cuda_graphs or not config.cudagraph_capture_sizes:
            cmd_parts.append("--enforce-eager")

        # Architecture-specific optimizations
        if self.gpu_architecture == GPUArchitecture.H100:
            if config.kv_cache_dtype == "fp8_e4m3":
                cmd_parts.append("--quantization fp8")
            if options.enable_chunked_prefill:
                cmd_parts.append("--enable-chunked-prefill")

        elif self.gpu_architecture == GPUArchitecture.A100:
            # A100-specific optimizations
            pass

        # Workload-specific optimizations
        if workload.target_metric == "latency":
            cmd_parts.append("--max-num-batched-tokens 8192")
        elif workload.target_metric == "throughput":
            max_batched_tokens = min(
                config.calculate_effective_batch_size() * config.max_model_len, 32768
            )
            cmd_parts.append(f"--max-num-batched-tokens {max_batched_tokens}")

        # Optional features
        if options.enable_prefix_caching:
            cmd_parts.append("--enable-prefix-caching")

        return cmd_parts

    def _build_custom_args(self, custom_args: dict[str, Any]) -> list[str]:
        """Build custom argument parameters."""
        cmd_parts = []

        for key, value in custom_args.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key} {value}")

        return cmd_parts

    def _build_environment_vars(
        self,
        parallelism_config: ParallelismConfiguration,
        options: VLLMDeploymentOptions,
    ) -> dict[str, str]:
        """Build environment variables."""
        env_vars = {}

        # Set CUDA visible devices for tensor parallelism
        total_gpus = parallelism_config.total_gpus_needed
        if total_gpus > 1:
            gpu_ids = ",".join(str(i) for i in range(total_gpus))
            env_vars["CUDA_VISIBLE_DEVICES"] = gpu_ids

        # Add custom environment variables
        if options.environment_vars:
            env_vars.update(options.environment_vars)

        return env_vars

    def _format_command(self, cmd_parts: list[str]) -> str:
        """Format command with proper line breaks."""
        return " \\\n    ".join(cmd_parts)

    def _build_config_summary(
        self,
        optimization_result,
        parallelism_config: ParallelismConfiguration,
        options: VLLMDeploymentOptions,
    ) -> dict[str, Any]:
        """Build configuration summary."""
        config = optimization_result.optimal_config

        return {
            "model_name": self.model_name,
            "gpu_architecture": self.gpu_architecture.value,
            "parallelism": {
                "tensor_parallel": parallelism_config.tensor_parallel_size,
                "pipeline_parallel": parallelism_config.pipeline_parallel_size,
                "total_gpus": parallelism_config.total_gpus_needed,
            },
            "memory_config": {
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_model_len": config.max_model_len,
                "kv_cache_dtype": config.kv_cache_dtype,
            },
            "performance_config": {
                "max_num_seqs": config.calculate_effective_batch_size(),
                "cuda_graphs_enabled": bool(config.cudagraph_capture_sizes)
                and not options.disable_cuda_graphs,
                "effective_batch_size": config.calculate_effective_batch_size(),
            },
            "deployment_mode": "offline" if options.enable_offline_mode else "serving",
        }

    def _extract_performance_estimates(
        self, optimization_result
    ) -> dict[str, float] | None:
        """Extract performance estimates from optimization result."""
        if not optimization_result.predictions:
            return None

        return {
            "throughput_tokens_per_second": optimization_result.predictions.get(
                "throughput", 0.0
            ),
            "latency_ms": optimization_result.predictions.get("latency", 0.0),
            "memory_efficiency": optimization_result.memory_efficiency,
            "performance_score": optimization_result.performance_score,
        }


def generate_vllm_deployment_command(
    model_name: str,
    hardware_profile: HardwareProfile,
    workload: WorkloadProfile,
    parallelism_config: ParallelismConfiguration,
    gpu_architecture: GPUArchitecture = GPUArchitecture.GENERIC,
    deployment_options: VLLMDeploymentOptions | None = None,
) -> VLLMDeploymentResult:
    """Generate optimized vLLM deployment command.

    This is the main entry point for vLLM deployment command generation
    that replaces the placeholder implementations in the API files.

    Args:
        model_name: Model name for deployment
        hardware_profile: Cluster hardware specifications
        workload: Expected workload characteristics
        parallelism_config: Parallelism configuration
        gpu_architecture: GPU architecture for optimizations
        deployment_options: Optional deployment configuration

    Returns:
        VLLMDeploymentResult with optimized command and metadata
    """
    generator = VLLMCommandGenerator(
        model_name=model_name,
        hardware_profile=hardware_profile,
        gpu_architecture=gpu_architecture,
    )

    return generator.generate_serving_command(
        workload=workload,
        parallelism_config=parallelism_config,
        options=deployment_options,
    )


def generate_vllm_offline_command(
    model_name: str,
    hardware_profile: HardwareProfile,
    parallelism_config: ParallelismConfiguration,
    input_file: str,
    output_file: str,
    batch_size: int = 1000,
    gpu_architecture: GPUArchitecture = GPUArchitecture.GENERIC,
    deployment_options: VLLMDeploymentOptions | None = None,
) -> VLLMDeploymentResult:
    """Generate optimized vLLM offline inference command.

    Args:
        model_name: Model name for deployment
        hardware_profile: Cluster hardware specifications
        parallelism_config: Parallelism configuration
        input_file: Path to input file with prompts
        output_file: Path to output file for results
        batch_size: Batch size for processing
        gpu_architecture: GPU architecture for optimizations
        deployment_options: Optional deployment configuration

    Returns:
        VLLMDeploymentResult with optimized command and metadata
    """
    generator = VLLMCommandGenerator(
        model_name=model_name,
        hardware_profile=hardware_profile,
        gpu_architecture=gpu_architecture,
    )

    return generator.generate_offline_command(
        input_file=input_file,
        output_file=output_file,
        parallelism_config=parallelism_config,
        batch_size=batch_size,
        options=deployment_options,
    )
