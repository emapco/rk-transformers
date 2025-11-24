# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field, fields
from typing import Any

from rktransformers.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OPSET,
    OptimizationLevelType,
    PlatformType,
    QuantizedAlgorithmType,
    QuantizedDtypeType,
    QuantizedMethodType,
    SupportedTaskType,
)


@dataclass
class QuantizationConfig:
    """
    Configuration for RKNN quantization.

    Quantization reduces model size and improves inference speed by converting weights
    and activations to lower precision (e.g., int8 instead of float32).

    Args:
        do_quantization: Enable quantization during build. Requires a calibration dataset.
        dataset_name: HuggingFace dataset name for quantization calibration (e.g., 'wikitext').
            Auto-detected: Not auto-detected, must be provided if do_quantization=True.
        dataset_subset: Subset name for the dataset (e.g., 'ax' for 'nyu-mll/glue').
        dataset_size: Number of samples to use from the dataset for calibration.
            Recommendation: 100-500 samples is usually sufficient.
        dataset_split: Dataset splits to use (e.g., ["train", "validation"]).
            Auto-detected: Uses ["train", "validation", "test"] if not specified.
        dataset_columns: List of dataset columns to use for calibration (e.g., ["question", "context"]).
            If not specified, falls back to auto-detection.
        quantized_dtype: Quantization data type.
            Options: "w8a8" (8-bit weights and activations), "w16a16i", "w16a16i_dfp".
            Recommendation: "w8a8" for best performance, "w16a16" for better accuracy.
        quantized_algorithm: Quantization calibration algorithm.
            Options: "normal", "mmse", "kl_divergence", "gdq".
            Recommendation: "normal" is fastest, "kl_divergence" may provide better accuracy.
        quantized_method: Quantization granularity.
            Options:
                "channel" (per-channel)
                "layer" (per-layer)
                "group{SIZE}" (group quantization) where SIZE is multiple of 32 between 32 and 256.
            Recommendation: "channel" provides better accuracy.
        quantized_hybrid_level: Hybrid quantization level (0-3).
            Higher values keep more layers in float for better accuracy but larger size.
        quant_img_RGB2BGR: Convert RGB to BGR during quantization (for image models only).
        auto_hybrid_cos_thresh: Cosine distance threshold for automatic hybrid quantization.
            Default: 0.98. Used when auto_hybrid is enabled in build.
        auto_hybrid_euc_thresh: Euclidean distance threshold for automatic hybrid quantization.
            Default: None. Used when auto_hybrid is enabled in build.
    """  # noqa: E501

    do_quantization: bool = False
    dataset_name: str | None = None
    dataset_subset: str | None = None
    dataset_size: int = 128
    dataset_split: list[str] | None = None
    dataset_columns: list[str] | None = None
    quantized_dtype: QuantizedDtypeType = "w8a8"
    quantized_algorithm: QuantizedAlgorithmType = "normal"
    quantized_method: QuantizedMethodType = "channel"
    quantized_hybrid_level: int = 0
    quant_img_RGB2BGR: bool = False
    auto_hybrid_cos_thresh: float = 0.98
    auto_hybrid_euc_thresh: float | None = None

    # Validation
    def __post_init__(self):
        if self.do_quantization and not self.dataset_name:
            # It's possible to provide a custom dataset file path instead of a HF dataset name,
            # but for this CLI we primarily support HF datasets or pre-processed files.
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "do_quantization": self.do_quantization,
            "dataset_name": self.dataset_name,
            "dataset_subset": self.dataset_subset,
            "dataset_size": self.dataset_size,
            "dataset_split": self.dataset_split,
            "dataset_columns": self.dataset_columns,
            "quantized_dtype": self.quantized_dtype,
            "quantized_algorithm": self.quantized_algorithm,
            "quantized_method": self.quantized_method,
            "quantized_hybrid_level": self.quantized_hybrid_level,
            "quant_img_RGB2BGR": self.quant_img_RGB2BGR,
            "auto_hybrid_cos_thresh": self.auto_hybrid_cos_thresh,
            "auto_hybrid_euc_thresh": self.auto_hybrid_euc_thresh,
        }


@dataclass
class OptimizationConfig:
    """
    Configuration for RKNN graph optimization.

    These optimizations transform the model graph for better performance on NPU.

    Args:
        optimization_level: Graph optimization level (0-3).
            0: No optimization
            1: Basic optimization
            2: Moderate optimization
            3: Aggressive optimization (recommended)
            Recommendation: Use 3 for best performance.
        enable_flash_attention: Enable Flash Attention optimization for transformer models.
            Significantly improves attention layer performance.
            Recommendation: Enable for transformer/BERT models.
        remove_weight: Remove weights from the model (for weight-sharing scenarios).
        compress_weight: Compress model weights to reduce model size.
        remove_reshape: Remove redundant reshape operations.
        sparse_infer: Enable sparse inference optimization.
        model_pruning: Enable model pruning to remove redundant connections.
    """

    optimization_level: OptimizationLevelType = 0
    enable_flash_attention: bool = False
    remove_weight: bool = False
    compress_weight: bool = False
    remove_reshape: bool = False
    sparse_infer: bool = False
    model_pruning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "optimization_level": self.optimization_level,
            "enable_flash_attention": self.enable_flash_attention,
            "remove_weight": self.remove_weight,
            "compress_weight": self.compress_weight,
            "remove_reshape": self.remove_reshape,
            "sparse_infer": self.sparse_infer,
            "model_pruning": self.model_pruning,
        }


@dataclass
class RKNNConfig:
    """
    This configures the conversion of ONNX models to RKNN format for Rockchip NPU.

    Args:
        target_platform: Target Rockchip platform.
            Options: "rk3566", "rk3568", "rk3588", "rk3576", "rk3562", "rk1106", "rk1103", "rv1126b".
            Auto-detected: Not auto-detected, defaults to "rk3588".
        quantization: Quantization configuration (see QuantizationConfig).
        optimization: Optimization configuration (see OptimizationConfig).

        # Model Input Configuration
        model_input_names: Names of model inputs (e.g., ["input_ids", "attention_mask", "token_type_ids"]).
            Auto-detected: Optimum automatically determines required inputs during ONNX export based on the model's architecture.
            - BERT models that use segment embeddings: Will include token_type_ids automatically
            - RoBERTa, sentence transformers: Will exclude token_type_ids automatically
            Note: This parameter is primarily used for RKNN conversion.
                  The ONNX export process inspects the exported model to determine actual inputs.
        type_vocab_size: Token type vocabulary size (informational, from model's config.json).
            Auto-detected: Read from model's config.json.

        # Model Dimensions
        batch_size: Batch size for input shapes during ONNX export and RKNN conversion (e.g., [batch_size, max_seq_length]).
            This controls the shape of inputs, not which inputs are included.
            Example: batch_size=1 creates inputs shaped [1, 128], batch_size=4 creates [4, 128].
        max_seq_length: Maximum sequence length for input shapes during ONNX export and RKNN conversion.
            Auto-detected: Read from model's config.json (max_position_embeddings).
            Falls back to 256 if not found in config.
            Note: large sequence length causes the RKNN export to segmentation fault.

        # RKNN-specific Parameters
        float_dtype: Floating point data type for non-quantized operations.
            Options: "float16", "float32".
            Recommendation: "float16" for better performance.
        mean_values: Mean values for input normalization (for image models).
        std_values: Standard deviation values for input normalization (for image models).
        custom_string: Custom configuration string passed to RKNN toolkit.
        inputs_yuv_fmt: YUV format for inputs (for image models).
        single_core_mode: Run model on single NPU core instead of multi-core.
            Only applicable for rk3588. Reduces model size.
        dynamic_input: Dynamic input shapes configuration. Experimental feature with sparse support on Rockchip NPUs.
            Format: [[[batch_size,seq_length],[batch_size,seq_length],...],[[batch_size,seq_length],[batch_size,seq_length],...],...].
            e.g.: [[[1,128],[1,128],...],[[1,256],[1,256],...],...].
        op_target: Specify the target device for specific operations.
            Useful for offloading operations to the CPU which are not supported by the NPU.
            Format: {'op_id':'cpu', 'op_id3':'cpu'}. Default is None.

            ```python
            unsupported_used = [(node.op_type, node.name) for node in model.graph.node if node.op_type in unsupported]
            op_target = {n:"cpu" for _,n in unsupported_used}
            ```

        # Export Settings
        model_id_or_path: Path to input ONNX model file or Hugging Face model ID.
        output_path: Path for output RKNN model file or directory.
            Optional. Defaults to the model's parent directory (for local files) or current directory (for Hub models).
        push_to_hub: Upload the exported model to HuggingFace Hub.
        hub_model_id: HuggingFace Hub repository ID (required if push_to_hub=True).
            Should include username/namespace (e.g., "username/model-name").
            If no namespace is provided (e.g., "model-name"), the username will be auto-detected from the token via whoami() API.
        hub_token: HuggingFace Hub authentication token.
        hub_private_repo: Create a private repository on HuggingFace Hub.

        # Optimum Export Settings
        opset: ONNX opset version (default: None, uses Optimum default or 18).
        task: Task type for export (default: "auto").
            Auto-detected: Uses model config.json `architectures` to determine task.
            - *ForSequenceClassification -> sequence-classification
            - *ForMaskedLM -> fill-mask
            - Fallback: feature-extraction (e.g. BertModel)
    """  # noqa: E501

    target_platform: PlatformType = "rk3588"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Model input configuration
    model_input_names: list[str] | None = None
    type_vocab_size: int | None = None

    # Model dimensions
    batch_size: int = DEFAULT_BATCH_SIZE
    max_seq_length: int | None = DEFAULT_MAX_SEQ_LENGTH

    # RKNN-specific parameters
    float_dtype: str = "float16"
    mean_values: list[list[float]] | None = None
    std_values: list[list[float]] | None = None
    custom_string: str | None = None
    inputs_yuv_fmt: list[str] | None = None
    single_core_mode: bool = False
    dynamic_input: list[list[list[int]]] | None = None
    op_target: dict[str, str] | None = None

    # Export settings
    model_id_or_path: str | None = None
    output_path: str | None = None
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None
    hub_private_repo: bool = False
    hub_create_pr: bool = False

    # Optimum export settings
    opset: int | None = DEFAULT_OPSET
    task: SupportedTaskType = "auto"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert config to dictionary for RKNN.config()
        """
        config_dict = {
            "target_platform": self.target_platform,
            "mean_values": self.mean_values,
            "std_values": self.std_values,
            # batch_size and max_seq_length are not direct RKNN.config args but used in build/load
            "quantized_dtype": self.quantization.quantized_dtype,
            "quantized_algorithm": self.quantization.quantized_algorithm,
            "quantized_method": self.quantization.quantized_method,
            "quantized_hybrid_level": self.quantization.quantized_hybrid_level,
            "quant_img_RGB2BGR": self.quantization.quant_img_RGB2BGR,
            "float_dtype": self.float_dtype,
            "optimization_level": self.optimization.optimization_level,
            "custom_string": self.custom_string,
            "remove_weight": self.optimization.remove_weight,
            "compress_weight": self.optimization.compress_weight,
            "inputs_yuv_fmt": self.inputs_yuv_fmt,
            "single_core_mode": self.single_core_mode,
            "dynamic_input": self.dynamic_input,
            "model_pruning": self.optimization.model_pruning,
            "remove_reshape": self.optimization.remove_reshape,
            "sparse_infer": self.optimization.sparse_infer,
            "enable_flash_attention": self.optimization.enable_flash_attention,
            "op_target": self.op_target,
            "auto_hybrid_cos_thresh": self.quantization.auto_hybrid_cos_thresh,
            "auto_hybrid_euc_thresh": self.quantization.auto_hybrid_euc_thresh,
        }
        # Filter out None values to let RKNN use defaults
        return {k: v for k, v in config_dict.items() if v is not None}

    def to_export_dict(self) -> dict[str, Any]:
        """
        Convert complete config to dictionary for export/persistence in rknn.json.
        This includes ALL configuration parameters for reproducibility.
        """
        export_dict = {
            # Model configuration
            "model_input_names": self.model_input_names,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "float_dtype": self.float_dtype,
            # Platform configuration
            "target_platform": self.target_platform,
            "single_core_mode": self.single_core_mode,
            # Other RKNN parameters
            "mean_values": self.mean_values,
            "std_values": self.std_values,
            "custom_string": self.custom_string,
            "inputs_yuv_fmt": self.inputs_yuv_fmt,
            "dynamic_input": self.dynamic_input,
            # Optimum export settings
            "opset": self.opset,
            "task": self.task,
            # Nested configurations
            "quantization": self.quantization.to_dict(),
            "optimization": self.optimization.to_dict(),
        }
        return {k: v for k, v in export_dict.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RKNNConfig":
        """Load configuration from a dictionary."""
        # Create copies of nested configs to avoid modifying input
        data = data.copy()

        if "quantization" in data and isinstance(data["quantization"], dict):
            data["quantization"] = QuantizationConfig(**data["quantization"])

        if "optimization" in data and isinstance(data["optimization"], dict):
            data["optimization"] = OptimizationConfig(**data["optimization"])

        # Filter arguments to only those accepted by __init__
        valid_fields = {f.name for f in fields(cls)}
        init_args = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**init_args)
