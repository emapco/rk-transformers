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

from argparse import ArgumentParser
from pathlib import Path

from rktransformers.commands.base import BaseRKNNCLICommand, CommandInfo
from rktransformers.configuration import OptimizationConfig, QuantizationConfig, RKNNConfig
from rktransformers.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OPSET,
    OPTIMIZATION_LEVEL_CHOICES,
    PLATFORM_CHOICES,
    QUANTIZED_ALGORITHM_CHOICES,
    QUANTIZED_DTYPE_CHOICES,
    QUANTIZED_METHOD_CHOICES,
    SUPPORTED_OPSETS,
    SUPPORTED_TASK_CHOICES,
)
from rktransformers.exporters.rknn.convert import export_rknn


def parse_args_rknn(parser: ArgumentParser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Path to ONNX model file or Hugging Face model ID."
    )
    required_group.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Path indicating the directory or file where to store the generated RKNN model. "
        "Defaults to the parent directory of the model file or the Hugging Face model directory.",
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for input shapes. Example: batch_size=1 → [1, seq_len], batch_size=4 → [4, seq_len]. "
        f"Default: {DEFAULT_BATCH_SIZE}",
    )
    optional_group.add_argument(
        "-msl",
        "--max-seq-length",
        type=int,
        default=None,
        help="Max sequence length for input shapes. Example: max_seq_length=128 → inputs shaped [batch, 128]. "
        f"Auto-detected from model config if not specified and uses {DEFAULT_MAX_SEQ_LENGTH} as a fallback. "
        "Note: large sequence length (4096) causes the RKNN export to segmentation fault.",
    )
    optional_group.add_argument(
        "--task-kwargs",
        type=str,
        default=None,
        help="Task-specific keyword arguments for ONNX export as comma-separated key=value pairs. "
        "Example: For multiple-choice tasks, use 'num_choices=4'. "
        "Multiple arguments can be specified like 'num_choices=4,other_param=value'.",
    )
    optional_group.add_argument(
        "--model-inputs",
        type=str,
        default=None,
        help="Comma-separated list of model input names (e.g., 'input_ids,attention_mask'). "
        "Auto-detected based on model's type_vocab_size from config.json. "
        "If type_vocab_size > 1: uses input_ids, attention_mask, token_type_ids. "
        "Otherwise: uses input_ids, attention_mask.",
    )
    optional_group.add_argument(
        "--platform",
        type=str,
        default="rk3588",
        choices=PLATFORM_CHOICES,
        help="Target platform. Default: rk3588",
    )

    optimization_group = parser.add_argument_group("Optimization arguments")
    optimization_group.add_argument(
        "-o",
        "--optimization-level",
        type=int,
        choices=OPTIMIZATION_LEVEL_CHOICES,
        default=0,
        help="RKNN Optimization level. Default: 0",
    )
    optimization_group.add_argument(
        "-fa",
        "--flash-attention",
        action="store_true",
        help="Enable Flash Attention optimization",
    )
    optimization_group.add_argument(
        "--compress-weight",
        action="store_true",
        help="Compress model weights to reduce RKNN model size",
    )
    optimization_group.add_argument(
        "--single-core-mode",
        action="store_true",
        help="Enable single NPU core mode (only applicable for rk3588). Reduces model size.",
    )

    optimization_group.add_argument(
        "--enable-custom-kernels",
        action="store_true",
        help="Enable custom kernels (e.g., CumSum) for operations not supported by RKNN.",
    )

    quantization_group = parser.add_argument_group("Quantization arguments")
    quantization_group.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Enable quantization. Otherwise, the model will be exported as float16.",
    )
    quantization_group.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default="w8a8",
        choices=QUANTIZED_DTYPE_CHOICES,
        help="Quantization data type. Options: w8a8 (8-bit weights/activations, best performance, most compatible), "
        "w8a16 (8-bit weights, 16-bit activations), w16a16i (16-bit integer), "
        "w16a16i_dfp (16-bit dynamic fixed point), w4a16 (4-bit weights, 16-bit activations)",
    )
    quantization_group.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="normal",
        choices=QUANTIZED_ALGORITHM_CHOICES,
        help="Quantization algorithm. Options: normal (default), mmse (Min Mean Square Error), "
        "kl_divergence (Kullback-Leibler divergence), gdq (Gradient-based Dynamic Quantization)",
    )
    quantization_group.add_argument(
        "-qm",
        "--quantized-method",
        type=str,
        default="channel",
        choices=QUANTIZED_METHOD_CHOICES,
        help="Quantization method. Options: layer (per-layer), channel (per-channel, better accuracy). "
        "Note: group quantization (e.g., group32) can be specified but is not validated by choices.",
    )
    quantization_group.add_argument(
        "--auto-hybrid-cos-thresh",
        type=float,
        default=0.98,
        help="Cosine distance threshold for automatic hybrid quantization (default: 0.98). "
        "Used when auto_hybrid is enabled during build.",
    )
    quantization_group.add_argument(
        "--auto-hybrid-euc-thresh",
        type=float,
        default=None,
        help="Euclidean distance threshold for automatic hybrid quantization (default: None). "
        "Used when auto_hybrid is enabled during build.",
    )

    dataset_group = parser.add_argument_group("Dataset arguments")
    dataset_group.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name for quantization (e.g. 'sentence-transformers/natural-questions')",
    )
    dataset_group.add_argument(
        "-dsb",
        "--dataset-subset",
        type=str,
        default=None,
        help="Subset name for the dataset (e.g. 'ax' for 'nyu-mll/glue').",
    )
    dataset_group.add_argument(
        "-dsz",
        "--dataset-size",
        type=int,
        default=128,
        help="Number of samples to use for quantization",
    )
    dataset_group.add_argument(
        "-dsp",
        "--dataset-split",
        type=str,
        default=None,
        help="Comma-separated list of dataset splits to use (e.g., 'train,validation'). "
        "Auto-detected: tries ['train', 'validation', 'test'] if not specified.",
    )
    dataset_group.add_argument(
        "-dc",
        "--dataset-columns",
        type=str,
        default=None,
        help="Comma-separated list of dataset columns to use for calibration (e.g., 'question,context'). "
        "If not specified, falls back to auto-detection. "
        "Currently only text-based datasets are supported and the first column is used.",
    )

    optimum_group = parser.add_argument_group("Optimum arguments (export Hugging Face models to ONNX)")
    optimum_group.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        choices=SUPPORTED_OPSETS,
        help="ONNX opset version. Recommended: 18+ for modern transformers. "
        "Minimum: 14 (required for SDPA). Maximum: 19. (maximum supported by RKNN). "
        f"Default: {DEFAULT_OPSET}.",
    )
    optimum_group.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=SUPPORTED_TASK_CHOICES,
        help="ONNX task type for export. Default: auto. "
        "Auto-detection uses `optimum` to determine the task "
        "(e.g., sequence-classification, fill-mask). Falls back to feature-extraction if undetermined. "
        "'auto' can be used to export models supported by `optimum` and not rk-transformers runtime functionality, "
        "in which case, the user is responsible for developing inference code using "
        "rknn-toolkit-lite2 library or subclassing `rktransformers.RKRTModel`.",
    )

    hub_group = parser.add_argument_group("Hugging Face Hub arguments")
    hub_group.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the exported model to the Hugging Face Hub.",
    )
    hub_group.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="The repository ID to push to on the Hugging Face Hub. "
        "Should include username/namespace (e.g., 'username/model-name'). "
        "If no namespace is provided (e.g., 'model-name'), the username will be "
        "auto-detected from your token via the whoami() API.",
    )
    hub_group.add_argument(
        "--token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    hub_group.add_argument(
        "--private-repo",
        action="store_true",
        help="Indicates whether the repository created should be private.",
    )
    hub_group.add_argument(
        "--create-pr",
        action="store_true",
        help="Whether to create a Pull Request instead of pushing directly to the main branch.",
    )


class RKNNExportCommand(BaseRKNNCLICommand):
    COMMAND = CommandInfo(name="export", help="Export ONNX to RKNN")

    @staticmethod
    def parse_args(parser: ArgumentParser):
        return parse_args_rknn(parser)

    def run(self):
        # Parse model inputs if provided
        model_input_names = None

        if self.args.model_inputs:
            model_input_names = [name.strip() for name in self.args.model_inputs.split(",")]

        # Parse task kwargs if provided (format: key1=value1,key2=value2)
        task_kwargs = None
        if self.args.task_kwargs:
            task_kwargs = {}
            for pair in self.args.task_kwargs.split(","):
                if "=" not in pair:
                    print(f"Warning: Ignoring invalid task kwarg '{pair}' (expected format: key=value)")
                    continue
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Try to convert to int, otherwise keep as string
                try:
                    task_kwargs[key] = int(value)
                except ValueError:
                    task_kwargs[key] = value

        # Parse dataset parameters if provided
        dataset_split = None
        if self.args.dataset_split:
            dataset_split = [split.strip() for split in self.args.dataset_split.split(",")]

        # Parse dataset columns if provided
        dataset_columns = None
        if self.args.dataset_columns:
            dataset_columns = [col.strip() for col in self.args.dataset_columns.split(",")]

        # Create configuration from args
        quantization_config = QuantizationConfig(
            do_quantization=self.args.quantize,
            dataset_name=self.args.dataset,
            dataset_subset=self.args.dataset_subset,
            dataset_size=self.args.dataset_size,
            dataset_split=dataset_split,
            dataset_columns=dataset_columns,
            quantized_dtype=self.args.dtype,
            quantized_algorithm=self.args.algorithm,
            quantized_method=self.args.quantized_method,
            auto_hybrid_cos_thresh=self.args.auto_hybrid_cos_thresh,
            auto_hybrid_euc_thresh=self.args.auto_hybrid_euc_thresh,
        )

        optimization_config = OptimizationConfig(
            optimization_level=self.args.optimization_level,
            enable_flash_attention=self.args.flash_attention,
            compress_weight=self.args.compress_weight,
        )

        config = RKNNConfig(
            target_platform=self.args.platform,
            quantization=quantization_config,
            optimization=optimization_config,
            model_input_names=model_input_names,
            model_name_or_path=self.args.model,
            output_path=str(self.args.output) if self.args.output else None,
            batch_size=self.args.batch_size,
            max_seq_length=self.args.max_seq_length,
            task_kwargs=task_kwargs,
            push_to_hub=self.args.push_to_hub,
            hub_model_id=self.args.model_id,
            hub_token=self.args.token,
            hub_private_repo=self.args.private_repo,
            hub_create_pr=self.args.create_pr,
            single_core_mode=self.args.single_core_mode,
            opset=self.args.opset,
            task=self.args.task,
        )

        try:
            export_rknn(config)
        except Exception as e:
            print(f"Export failed: {e}")
