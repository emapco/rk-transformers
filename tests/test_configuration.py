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

"""Tests for RKNN configuration dataclasses."""

import json
from pathlib import Path

from rktransformers.configuration import OptimizationConfig, QuantizationConfig, RKNNConfig


class TestRKNNConfig:
    def test_defaults(self) -> None:
        config = RKNNConfig()
        assert config.target_platform == "rk3588"
        assert isinstance(config.quantization, QuantizationConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert config.quantization.quantized_dtype == "w8a8"
        assert config.optimization.optimization_level == 0

    def test_from_dict(self) -> None:
        data = {
            "target_platform": "rk3588",
            "quantization": {
                "do_quantization": True,
                "dataset_name": "wikitext",
                "quantized_dtype": "w8a8",
            },
            "optimization": {
                "optimization_level": 3,
                "enable_flash_attention": True,
            },
            "model_input_names": ["input_ids", "attention_mask"],
            "batch_size": 4,
        }
        config = RKNNConfig.from_dict(data)

        assert config.target_platform == "rk3588"
        assert config.quantization.do_quantization is True
        assert config.quantization.dataset_name == "wikitext"
        assert config.optimization.optimization_level == 3
        assert config.optimization.enable_flash_attention is True
        assert config.model_input_names == ["input_ids", "attention_mask"]
        assert config.batch_size == 4

    def test_from_dict_extra_fields(self) -> None:
        # Test that extra fields are ignored
        data = {
            "target_platform": "rk3588",
            "extra_field": "should_be_ignored",
        }
        config = RKNNConfig.from_dict(data)
        assert config.target_platform == "rk3588"
        assert not hasattr(config, "extra_field")

    def test_to_export_dict_roundtrip(self) -> None:
        original_config = RKNNConfig(
            target_platform="rk3588",
            model_input_names=["input_ids"],
            quantization=QuantizationConfig(do_quantization=True),
            optimization=OptimizationConfig(optimization_level=2),
        )

        export_dict = original_config.to_export_dict()
        loaded_config = RKNNConfig.from_dict(export_dict)

        assert loaded_config.target_platform == original_config.target_platform
        assert loaded_config.model_input_names == original_config.model_input_names
        assert loaded_config.quantization.do_quantization == original_config.quantization.do_quantization
        assert loaded_config.optimization.optimization_level == original_config.optimization.optimization_level

    def test_load_from_file(self) -> None:
        rknn_json_path = Path(__file__).parent / "data/random_bert/rknn.json"
        with open(rknn_json_path) as f:
            full_config = json.load(f)

        # Simulate selecting the config for "model_b1_s32_o3.rknn"
        target_key = "rknn/model_b1_s32_o3.rknn"
        config_dict = full_config[target_key]

        config = RKNNConfig.from_dict(config_dict)

        assert config.target_platform == "rk3588"
        assert config.model_input_names == ["input_ids", "attention_mask", "token_type_ids"]
        assert config.optimization.optimization_level == 3
        assert config.quantization.do_quantization is False
