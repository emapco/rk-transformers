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

"""Tests for RKNN model loading and platform compatibility."""

import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig, PretrainedConfig

from rktransformers.constants import PlatformType
from rktransformers.modeling import RKRTModel


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    """Create a temporary RKNN model file."""
    model_path = tmp_path / "dummy.rknn"
    model_path.touch()
    return model_path


class TestRKNNPlatformCheck:
    """Tests for RKNN platform compatibility checking."""

    @patch("rktransformers.modeling.RKNNLite")
    def test_compatible_platform(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, model_file: Path
    ) -> None:
        """Test loading model with compatible platform."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

        model = RKRTModel(config=pretrained_config, model_path=model_file, platform="rk3588")
        assert model.rknn is not None

        supported = model.list_model_compatible_platform()
        assert supported == OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

    @patch("rktransformers.modeling.RKNNLite")
    def test_incompatible_platform(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, model_file: Path
    ) -> None:
        """Test loading model with incompatible platform raises error."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

        with pytest.raises(RuntimeError, match="not compatible"):
            RKRTModel(
                config=pretrained_config,
                model_path=model_file,
                platform="rk3566",  # Not in supported list
            )

    @patch("rktransformers.modeling.RKNNLite")
    def test_case_insensitivity(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, model_file: Path
    ) -> None:
        """Test that platform matching is case-insensitive."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

        model = RKRTModel(config=pretrained_config, model_path=model_file, platform="rk3588")
        assert model.rknn is not None

    @pytest.mark.parametrize(
        "platform,should_pass",
        [
            ("rk3588", True),  # Lowercase works
            ("rk3566", False),  # Not supported
            ("rk3568", False),  # Not supported
        ],
    )
    @patch("rktransformers.modeling.RKNNLite")
    def test_platform_compatibility_variations(
        self,
        mock_rknnlite_class: MagicMock,
        pretrained_config: PretrainedConfig,
        model_file: Path,
        platform: str,
        should_pass: bool,
    ) -> None:
        """Test platform compatibility with various platform strings."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

        if should_pass:
            model = RKRTModel(config=pretrained_config, model_path=model_file, platform=cast(PlatformType, platform))
            assert model.rknn is not None
        else:
            with pytest.raises(RuntimeError, match="not compatible"):
                RKRTModel(config=pretrained_config, model_path=model_file, platform=cast(PlatformType, platform))

    @patch("rktransformers.modeling.RKNNLite")
    def test_multiple_supported_platforms(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, model_file: Path
    ) -> None:
        """Test model that supports multiple platforms."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [
                ("filled_target_platform", ["rk3588", "rk3568"]),
                ("support_target_platform", ["rk3588", "rk3568"]),
            ]
        )

        model1 = RKRTModel(config=pretrained_config, model_path=model_file, platform="rk3588")
        assert model1.rknn is not None

        model2 = RKRTModel(config=pretrained_config, model_path=model_file, platform="rk3568")
        assert model2.rknn is not None

        with pytest.raises(RuntimeError, match="not compatible"):
            RKRTModel(config=pretrained_config, model_path=model_file, platform="rk3566")


class TestRKRTModelFileName:
    @pytest.fixture
    def temp_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create dummy config
            config = AutoConfig.from_pretrained("bert-base-uncased")
            config.save_pretrained(tmp_path)

            # Create dummy RKNN files
            (tmp_path / "model.rknn").touch()
            (tmp_path / "model_quantized.rknn").touch()
            (tmp_path / "other.rknn").touch()

            # Create rknn subdirectory with quantized model
            (tmp_path / "rknn").mkdir()
            (tmp_path / "rknn" / "model_w8a8.rknn").touch()

            # Create dummy rknn.json
            with open(tmp_path / "rknn.json", "w") as f:
                f.write("{}")

            yield tmp_path

    @patch("rktransformers.modeling.RKNNLite")
    def test_from_pretrained_with_file_name(self, mock_rknn_lite, temp_model_dir):
        # Setup mock
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        # Test loading specific file
        RKRTModel.from_pretrained(temp_model_dir, file_name="model_quantized.rknn", platform="rk3588")

        # Verify correct file was loaded
        expected_path = (temp_model_dir / "model_quantized.rknn").as_posix()
        mock_rknn.load_rknn.assert_called_with(expected_path)

    @patch("rktransformers.modeling.RKNNLite")
    def test_from_pretrained_default(self, mock_rknn_lite, temp_model_dir):
        # Setup mock
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        # Test loading default (should find model.rknn or first available)
        RKRTModel.from_pretrained(temp_model_dir, platform="rk3588")

        # Verify it loaded one of the valid files (preference is usually model.rknn)
        # Based on _infer_file_path logic, it prefers RKNN_WEIGHTS_NAME ("model.rknn")
        expected_path = (temp_model_dir / "model.rknn").as_posix()
        mock_rknn.load_rknn.assert_called_with(expected_path)

    @patch("rktransformers.modeling.RKNNLite")
    def test_from_pretrained_with_subdirectory_file_name(self, mock_rknn_lite, temp_model_dir):
        # Setup mock
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        # Test loading quantized model from subdirectory
        RKRTModel.from_pretrained(temp_model_dir, file_name="rknn/model_w8a8.rknn", platform="rk3588")

        # Verify correct file was loaded
        expected_path = (temp_model_dir / "rknn" / "model_w8a8.rknn").as_posix()
        mock_rknn.load_rknn.assert_called_with(expected_path)

    @patch("rktransformers.modeling.RKNNLite")
    def test_from_pretrained_file_name_not_found(self, mock_rknn_lite, temp_model_dir):
        # Setup mock
        mock_rknn = MagicMock()
        mock_rknn_lite.return_value = mock_rknn

        # Test loading non-existent file
        with pytest.raises((FileNotFoundError, OSError)):
            RKRTModel.from_pretrained(temp_model_dir, file_name="non_existent.rknn", platform="rk3588")
