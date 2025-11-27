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

from typing import Literal, get_args

RKNN_WEIGHTS_NAME = "model.rknn"
RKNN_FILE_PATTERN = r".*\.rknn$"

DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 1
DEFAULT_OPSET = 18
AUTO_DETECT_TEXT_FIELDS = (
    "text",
    "sentence",
    "content",
    "query",
    "question",
    "context",
    "prompt",
)
IGNORE_MODEL_REPO_FILES = [
    "*.onnx",
    "*.bin",
    "*.safetensors",
    "*.pt",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
    "*.tflite",
    "*.ot",
    ".gitattributes",
    "openvino/**",
    "onnx/**",
    ".locks/**",
    "models--*/**",
]
ALLOW_MODEL_REPO_FILES = [
    "*.json",
    "*.txt",
    "*.md",
    "*.py",
    "*.yaml",
    "*.yml",
    "*.rknn",
]

PlatformType = Literal[
    "simulator",
    "rk3588",
    "rk3576",
    "rk3568",
    "rk3566",
    "rk3562",
    "rk2118",
    "rv1126b",
    "rv1106b",
    "rv1106",
    "rv1103b",
    "rv1103",
]
PLATFORM_CHOICES = get_args(PlatformType)
CoreMaskType = Literal[  # for rk3588/rk3576 devices or other device with multiple NPU cores
    "auto",  # automatic mode - select idle core inside the NPU
    "0",  # NPU0 core
    "1",  # NPU1 core
    "2",  # NPU2 core
    "0_1",  # NPU0 and NPU1 core simultaneously
    "0_1_2",  # NPU0, NPU1 and NPU2 core simultaneously - only supported by rk3588
    "all",  # all npu cores
]
CORE_MASK_CHOICES = get_args(CoreMaskType)
# Quantization data types
# w8a8: 8-bit weights and activations (default, best performance)
# w8a16: 8-bit weights, 16-bit activations
# w16a16i: 16-bit weights and activations (integer)
# w16a16i_dfp: 16-bit weights and activations (dynamic fixed point)
# w4a16: 4-bit weights, 16-bit activations
QuantizedDtypeType = Literal["w8a8", "w8a16", "w16a16i", "w16a16i_dfp", "w4a16"]
QUANTIZED_DTYPE_CHOICES = get_args(QuantizedDtypeType)
# Quantization algorithms
# normal: Default quantization algorithm
# mmse: Min Mean Square Error
# kl_divergence: Kullback-Leibler divergence
# gdq: Gradient-based Dynamic Quantization
QuantizedAlgorithmType = Literal["normal", "mmse", "kl_divergence", "gdq"]
QUANTIZED_ALGORITHM_CHOICES = get_args(QuantizedAlgorithmType)
# Quantization methods
# layer: Per-layer quantization
# channel: Per-channel quantization (better accuracy)
# group{SIZE}: Group quantization where SIZE is multiple of 32 between 32 and 256 (e.g., group32, group64)
QuantizedMethodType = Literal[
    "layer", "channel", "group32", "group64", "group96", "group128", "group160", "group192", "group224", "group256"
]
QUANTIZED_METHOD_CHOICES = get_args(QuantizedMethodType)
OptimizationLevelType = Literal[0, 1, 2, 3]
OPTIMIZATION_LEVEL_CHOICES = get_args(OptimizationLevelType)
# Supported tasks by export and modeling
SupportedTaskType = Literal[
    "auto",  # detect task from model config.json architectures[0]
    "feature-extraction",
    "fill-mask",
    "sequence-classification",
]
SUPPORTED_TASK_CHOICES = get_args(SupportedTaskType)
OpsetType = Literal[14, 15, 16, 17, 18, 19]  # ONNX opset 14 to 19 - sdpa added in 14 and rknn supports up to 19
SUPPORTED_OPSETS = get_args(OpsetType)

# Mapping from task to RKRTModel class name
TASK_TO_RK_MODEL_CLASS = {
    "fill-mask": "RKRTModelForMaskedLM",
    "sequence-classification": "RKRTModelForSequenceClassification",
    "feature-extraction": "RKRTModelForFeatureExtraction",
}
