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

"""Tests for RKNN model loading utilities."""

from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from rktransformers.load import load_rknn_model


class TestLoadRKNNModel:
    """Tests for load_rknn_model function."""

    @patch("rktransformers.modeling.RKRTModelForFeatureExtraction.from_pretrained")
    def test_load_feature_extraction(
        self, mock_from_pretrained: MagicMock, pretrained_config: PretrainedConfig
    ) -> None:
        """Test loading model for feature extraction task."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_rknn_model("test-model", pretrained_config, "feature-extraction")

        mock_from_pretrained.assert_called_once_with("test-model", config=pretrained_config)
        assert model == mock_from_pretrained.return_value

    @patch("rktransformers.modeling.RKRTModelForMaskedLM.from_pretrained")
    def test_load_masked_lm(self, mock_from_pretrained: MagicMock, pretrained_config: PretrainedConfig) -> None:
        """Test loading model for masked language modeling task."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_rknn_model("test-model", pretrained_config, "fill-mask")

        mock_from_pretrained.assert_called_once_with("test-model", config=pretrained_config)
        assert model == mock_from_pretrained.return_value

    @patch("rktransformers.modeling.RKRTModelForSequenceClassification.from_pretrained")
    def test_load_sequence_classification(
        self, mock_from_pretrained: MagicMock, pretrained_config: PretrainedConfig
    ) -> None:
        """Test loading model for sequence classification task."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_rknn_model("test-model", pretrained_config, "sequence-classification")

        mock_from_pretrained.assert_called_once_with("test-model", config=pretrained_config)
        assert model == mock_from_pretrained.return_value

    def test_unsupported_task(self, pretrained_config: PretrainedConfig) -> None:
        """Test that unsupported task raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported task"):
            load_rknn_model("test-model", pretrained_config, "unsupported-task")

    @pytest.mark.parametrize(
        "task,expected_model_class",
        [
            ("feature-extraction", "RKRTModelForFeatureExtraction"),
            ("fill-mask", "RKRTModelForMaskedLM"),
            ("sequence-classification", "RKRTModelForSequenceClassification"),
        ],
    )
    def test_task_to_model_mapping(
        self, task: str, expected_model_class: str, pretrained_config: PretrainedConfig
    ) -> None:
        """Test that different tasks load the correct model class."""
        with patch(f"rktransformers.modeling.{expected_model_class}.from_pretrained") as mock_from_pretrained:
            mock_model = MagicMock()
            mock_from_pretrained.return_value = mock_model

            model = load_rknn_model("test-model", pretrained_config, task)

            mock_from_pretrained.assert_called_once_with("test-model", config=pretrained_config)
            assert model == mock_model

    @pytest.mark.parametrize(
        "invalid_task",
        [
            "invalid-task",
            "text-generation",
            "translation",
            "",
            "none",
        ],
    )
    def test_invalid_tasks(self, invalid_task: str, pretrained_config: PretrainedConfig) -> None:
        """Test that invalid tasks raise ValueError."""
        with pytest.raises(ValueError):
            load_rknn_model("test-model", pretrained_config, invalid_task)
