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

"""Tests for RKNN CLI and configuration."""

from argparse import ArgumentParser
from typing import cast

import pytest

from rktransformers.commands.export.rknn import parse_args_rknn
from rktransformers.configuration import RKNNConfig
from rktransformers.constants import PlatformType


class TestRKNNConfig:
    """Tests for RKNNConfig class."""

    def test_config_defaults(self) -> None:
        """Test that RKNNConfig has correct default values."""
        config = RKNNConfig()

        assert config.target_platform == "rk3588"
        assert not config.quantization.do_quantization
        assert config.optimization.optimization_level == 0

    def test_to_dict(self) -> None:
        """Test converting RKNNConfig to dictionary."""
        config = RKNNConfig(target_platform="rk3566")
        config.quantization.do_quantization = True
        config.quantization.quantized_dtype = "w16a16i"

        d = config.to_dict()

        assert d["target_platform"] == "rk3566"
        assert d["quantized_dtype"] == "w16a16i"
        assert d["optimization_level"] == 0
        assert config.batch_size == 1
        assert config.max_seq_length == 512
        assert not config.push_to_hub
        assert config.hub_model_id is None

    @pytest.mark.parametrize(
        "platform,expected",
        [
            ("rk3588", "rk3588"),
            ("rk3566", "rk3566"),
            ("rk3568", "rk3568"),
        ],
    )
    def test_different_platforms(self, platform: str, expected: str) -> None:
        """Test RKNNConfig with different target platforms."""
        config = RKNNConfig(target_platform=cast(PlatformType, platform))
        assert config.target_platform == expected

    def test_quantization_config(self) -> None:
        """Test quantization configuration settings."""
        config = RKNNConfig()
        config.quantization.do_quantization = True
        config.quantization.dataset_name = "test_dataset"
        config.quantization.quantized_dtype = "w16a16i_dfp"

        assert config.quantization.do_quantization
        assert config.quantization.dataset_name == "test_dataset"
        assert config.quantization.quantized_dtype == "w16a16i_dfp"


class TestRKNNCommand:
    """Tests for RKNN CLI argument parsing."""

    def test_parse_args_basic(self) -> None:
        """Test basic CLI argument parsing."""
        parser = ArgumentParser()
        parse_args_rknn(parser)

        args = parser.parse_args(
            [
                "--model",
                "model.onnx",
                "output.rknn",
                "--platform",
                "rk3588",
            ]
        )

        assert args.model == "model.onnx"
        assert str(args.output) == "output.rknn"
        assert args.platform == "rk3588"
        assert not args.quantize

    def test_parse_args_with_quantization(self) -> None:
        """Test CLI argument parsing with quantization options."""
        parser = ArgumentParser()
        parse_args_rknn(parser)

        args = parser.parse_args(
            [
                "--model",
                "model.onnx",
                "output.rknn",
                "--platform",
                "rk3588",
                "--quantize",
                "--dataset",
                "test_dataset",
            ]
        )

        assert args.model == "model.onnx"
        assert args.platform == "rk3588"
        assert args.quantize
        assert args.dataset == "test_dataset"

    def test_parse_args_with_hub_upload(self) -> None:
        """Test CLI argument parsing with Hugging Face Hub options."""
        parser = ArgumentParser()
        parse_args_rknn(parser)

        args = parser.parse_args(
            [
                "--model",
                "model.onnx",
                "output.rknn",
                "--platform",
                "rk3588",
                "--push-to-hub",
                "--model-id",
                "test/model",
            ]
        )

        assert args.push_to_hub
        assert args.model_id == "test/model"

    def test_parse_args_all_options(self) -> None:
        """Test CLI argument parsing with all options."""
        parser = ArgumentParser()
        parse_args_rknn(parser)

        args = parser.parse_args(
            [
                "--model",
                "model.onnx",
                "output.rknn",
                "--platform",
                "rk3588",
                "--quantize",
                "--dataset",
                "test_dataset",
                "--batch-size",
                "4",
                "--max-seq-length",
                "512",
                "--push-to-hub",
                "--model-id",
                "test/model",
            ]
        )

        assert args.model == "model.onnx"
        assert str(args.output) == "output.rknn"
        assert args.platform == "rk3588"
        assert args.quantize
        assert args.dataset == "test_dataset"
        assert args.batch_size == 4
        assert args.max_seq_length == 512
        assert args.push_to_hub
        assert args.model_id == "test/model"

    @pytest.mark.parametrize(
        "batch_size,max_seq_length",
        [
            (1, 128),
            (2, 256),
            (4, 512),
            (8, 1024),
        ],
    )
    def test_parse_args_batch_and_sequence(self, batch_size: int, max_seq_length: int) -> None:
        """Test CLI argument parsing with different batch sizes and sequence lengths."""
        parser = ArgumentParser()
        parse_args_rknn(parser)

        args = parser.parse_args(
            [
                "--model",
                "model.onnx",
                "output.rknn",
                "--batch-size",
                str(batch_size),
                "--max-seq-length",
                str(max_seq_length),
            ]
        )

        assert args.batch_size == batch_size
        assert args.max_seq_length == max_seq_length
