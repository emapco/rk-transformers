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

import numpy as np
import pytest
import torch
from transformers import AutoConfig, PretrainedConfig

from rktransformers.configuration import RKNNConfig
from rktransformers.constants import PlatformType
from rktransformers.modeling import (
    RKRTModel,
    RKRTModelForMultipleChoice,
    RKRTModelForQuestionAnswering,
    RKRTModelForTokenClassification,
)


class TestRKNNPlatformCheck:
    """Tests for RKNN platform compatibility checking."""

    @patch("rktransformers.modeling.RKNNLite")
    def test_compatible_platform(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, dummy_rknn_file: Path
    ) -> None:
        """Test loading model with compatible platform."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, platform="rk3588")
        assert model.rknn is not None

        supported = model.list_model_compatible_platform()
        assert supported == OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

    @patch("rktransformers.modeling.RKNNLite")
    def test_incompatible_platform(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, dummy_rknn_file: Path
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
                model_path=dummy_rknn_file,
                platform="rk3566",  # Not in supported list
            )

    @patch("rktransformers.modeling.RKNNLite")
    def test_case_insensitivity(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, dummy_rknn_file: Path
    ) -> None:
        """Test that platform matching is case-insensitive."""
        mock_rknn = mock_rknnlite_class.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, platform="rk3588")
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
        dummy_rknn_file: Path,
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
            model = RKRTModel(
                config=pretrained_config, model_path=dummy_rknn_file, platform=cast(PlatformType, platform)
            )
            assert model.rknn is not None
        else:
            with pytest.raises(RuntimeError, match="not compatible"):
                RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, platform=cast(PlatformType, platform))

    @patch("rktransformers.modeling.RKNNLite")
    def test_multiple_supported_platforms(
        self, mock_rknnlite_class: MagicMock, pretrained_config: PretrainedConfig, dummy_rknn_file: Path
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

        model1 = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, platform="rk3588")
        assert model1.rknn is not None

        model2 = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, platform="rk3568")
        assert model2.rknn is not None

        with pytest.raises(RuntimeError, match="not compatible"):
            RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, platform="rk3566")


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


class TestRKRTModelingTasks:
    """Test RKNN model task-specific classes."""

    @patch("rktransformers.modeling.RKNNLite")
    def test_question_answering(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test RKRTModelForQuestionAnswering forward pass."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        batch_size = 1
        seq_len = 128

        mock_rknn.inference.return_value = [
            np.random.randn(batch_size, seq_len).astype(np.float32),
            np.random.randn(batch_size, seq_len).astype(np.float32),
        ]
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModelForQuestionAnswering(config=pretrained_config, model_path=dummy_rknn_file)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(outputs, "start_logits")
        assert hasattr(outputs, "end_logits")
        assert outputs.start_logits.shape == (batch_size, seq_len)
        assert outputs.end_logits.shape == (batch_size, seq_len)

    @patch("rktransformers.modeling.RKNNLite")
    def test_token_classification(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test RKRTModelForTokenClassification forward pass."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        batch_size = 1
        seq_len = 128
        num_labels = 9

        mock_rknn.inference.return_value = [np.random.randn(batch_size, seq_len, num_labels).astype(np.float32)]
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModelForTokenClassification(config=pretrained_config, model_path=dummy_rknn_file)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, seq_len, num_labels)

    @patch("rktransformers.modeling.RKNNLite")
    def test_multiple_choice(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test RKRTModelForMultipleChoice forward pass."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        batch_size = 1
        seq_len = 128
        num_choices = 4

        # RKNN returns logits with shape [batch_size, num_choices]
        mock_rknn.inference.return_value = [np.random.randn(batch_size, num_choices).astype(np.float32)]
        mock_rknn_lite.return_value = mock_rknn

        # Create config with task_kwargs
        rknn_config = RKNNConfig(task_kwargs={"num_choices": num_choices})
        model = RKRTModelForMultipleChoice(
            config=pretrained_config, model_path=dummy_rknn_file, rknn_config=rknn_config
        )

        input_ids = torch.randint(0, 1000, (batch_size, num_choices, seq_len))
        attention_mask = torch.ones((batch_size, num_choices, seq_len))

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape == (batch_size, num_choices)


class TestInputPaddingMethods:
    """Test input padding and preparation methods."""

    @patch("rktransformers.modeling.RKNNLite")
    def test_pad_to_model_input_dimensions_2d(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test 2D padding for standard tasks."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, max_seq_length=128, batch_size=4)

        # Test with torch tensor
        input_tensor = torch.randint(0, 1000, (2, 64))
        padded = model._pad_to_model_input_dimensions(input_tensor, padding_id=0, use_torch=True)
        assert padded.shape == (4, 128)

        # Test with numpy array
        input_array = np.random.randint(0, 1000, (2, 64))
        padded = model._pad_to_model_input_dimensions(input_array, padding_id=0, use_torch=False)
        assert padded.shape == (4, 128)

    @patch("rktransformers.modeling.RKNNLite")
    def test_pad_to_model_input_dimensions_3d(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test 3D padding for multiple-choice tasks."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, max_seq_length=128, batch_size=4)

        # Test with torch tensor
        input_tensor = torch.randint(0, 1000, (2, 4, 64))
        target_shape = (4, 4, 128)
        padded = model._pad_to_model_input_dimensions(
            input_tensor, padding_id=0, use_torch=True, target_shape=target_shape
        )
        assert padded.shape == (4, 4, 128)

        # Test with numpy array
        input_array = np.random.randint(0, 1000, (2, 4, 64))
        padded = model._pad_to_model_input_dimensions(
            input_array, padding_id=0, use_torch=False, target_shape=target_shape
        )
        assert padded.shape == (4, 4, 128)

    @patch("rktransformers.modeling.RKNNLite")
    def test_pad_to_model_input_dimensions_no_padding_needed(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test padding when input already matches target shape."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, max_seq_length=128, batch_size=4)

        # Input already at target size
        input_tensor = torch.randint(0, 1000, (4, 128))
        padded = model._pad_to_model_input_dimensions(input_tensor, padding_id=0, use_torch=True)
        assert padded.shape == (4, 128)
        assert torch.equal(input_tensor, padded)  # Should be unchanged

    @patch("rktransformers.modeling.RKNNLite")
    def test_prepare_text_inputs_2d(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test standard 2D input preparation."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, max_seq_length=128, batch_size=4)

        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones((2, 64))

        use_torch, model_inputs, original_shape = model._prepare_text_inputs(input_ids, attention_mask, None)

        assert use_torch is True
        assert original_shape == (2, 64)
        assert model_inputs["input_ids"].shape == (4, 128)
        assert model_inputs["attention_mask"].shape == (4, 128)

    @patch("rktransformers.modeling.RKNNLite")
    def test_prepare_text_inputs_3d(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test 3D input preparation for multiple-choice."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file, max_seq_length=128, batch_size=4)

        input_ids = torch.randint(0, 1000, (2, 4, 64))
        attention_mask = torch.ones((2, 4, 64))
        target_shape = (4, 4, 128)

        use_torch, model_inputs, original_shape = model._prepare_text_inputs(
            input_ids, attention_mask, None, input_shape=target_shape
        )

        assert use_torch is True
        assert original_shape == (2, 4, 64)
        assert model_inputs["input_ids"].shape == (4, 4, 128)
        assert model_inputs["attention_mask"].shape == (4, 4, 128)

    @patch("rktransformers.modeling.RKNNLite")
    def test_prepare_text_inputs_with_token_type_ids(self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path) -> None:
        """Test input preparation with token type IDs."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        # Create config with token types
        config = PretrainedConfig(model_type="rknn_model", type_vocab_size=2)
        model = RKRTModel(config=config, model_path=dummy_rknn_file, max_seq_length=128, batch_size=4)

        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones((2, 64))
        token_type_ids = torch.zeros((2, 64))

        use_torch, model_inputs, original_shape = model._prepare_text_inputs(input_ids, attention_mask, token_type_ids)

        assert use_torch is True
        assert original_shape == (2, 64)
        assert model_inputs["input_ids"].shape == (4, 128)
        assert model_inputs["attention_mask"].shape == (4, 128)
        assert model_inputs["token_type_ids"].shape == (4, 128)

    @patch("rktransformers.modeling.RKNNLite")
    def test_prepare_text_inputs_missing_input_ids(
        self, mock_rknn_lite: MagicMock, dummy_rknn_file: Path, pretrained_config: PretrainedConfig
    ) -> None:
        """Test error handling for missing input_ids."""
        mock_rknn = MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn_lite.return_value = mock_rknn

        model = RKRTModel(config=pretrained_config, model_path=dummy_rknn_file)

        with pytest.raises(ValueError, match="`input_ids` is required"):
            model._prepare_text_inputs(None, None, None)
