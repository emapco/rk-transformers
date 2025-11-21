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

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantizationConfig:
    """
    Configuration for RKNN quantization.
    """

    do_quantization: bool = False
    dataset_name: str | None = None
    dataset_size: int = 100
    quantized_dtype: str = "w8a8"
    quantized_algorithm: str = "normal"
    quantized_method: str = "channel"
    quantized_hybrid_level: int = 0
    quant_img_RGB2BGR: bool = False

    # Validation
    def __post_init__(self):
        if self.do_quantization and not self.dataset_name:
            # It's possible to provide a custom dataset file path instead of a HF dataset name,
            # but for this CLI we primarily support HF datasets or pre-processed files.
            pass


@dataclass
class OptimizationConfig:
    """
    Configuration for RKNN optimization.
    """

    optimization_level: int = 3
    enable_flash_attention: bool = False
    remove_weight: bool = False
    compress_weight: bool = False
    remove_reshape: bool = False
    sparse_infer: bool = False
    model_pruning: bool = False


@dataclass
class RKNNConfig:
    """
    Main configuration for RKNN export.
    """

    target_platform: str = "rk3588"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Other RKNN config parameters
    mean_values: list[list[float]] | None = None
    std_values: list[list[float]] | None = None
    batch_size: int = 1
    max_seq_length: int = 256
    float_dtype: str = "float16"
    custom_string: str | None = None
    inputs_yuv_fmt: list[str] | None = None
    single_core_mode: bool = False
    dynamic_input: list[list[list[int]]] | None = None

    # Export specific
    model_path: str | None = None
    output_path: str | None = None
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None
    hub_private_repo: bool = False

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
        }
        # Filter out None values to let RKNN use defaults
        return {k: v for k, v in config_dict.items() if v is not None}
