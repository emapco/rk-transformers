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

from unittest.mock import MagicMock, patch

from rktransformers.exporters.rknn.convert import detect_task, load_model_config


class TestAutoTaskDetection:
    """Tests for auto task detection functionality."""

    def test_detect_task_sequence_classification(self) -> None:
        config = {"architectures": ["BertForSequenceClassification"]}
        task = detect_task(config)
        assert task == "sequence-classification"

    def test_detect_task_masked_lm(self) -> None:
        config = {"architectures": ["RobertaForMaskedLM"]}
        task = detect_task(config)
        assert task == "fill-mask"

    def test_detect_task_feature_extraction_explicit(self) -> None:
        config = {"architectures": ["BertModel"]}
        task = detect_task(config)
        assert task == "feature-extraction"

    def test_detect_task_fallback_unknown_arch(self) -> None:
        config = {"architectures": ["UnknownArchitecture"]}
        task = detect_task(config)
        assert task == "feature-extraction"

    def test_detect_task_fallback_no_arch(self) -> None:
        config = {}
        task = detect_task(config)
        assert task == "feature-extraction"

    def test_detect_task_fallback_empty_arch_list(self) -> None:
        config = {"architectures": []}
        task = detect_task(config)
        assert task == "feature-extraction"


class TestLoadModelConfig:
    """Tests for loading model configuration."""

    @patch("rktransformers.exporters.rknn.utils.AutoConfig")
    def test_load_model_config_auto_config(self, mock_auto_config: MagicMock) -> None:
        mock_config_instance = MagicMock()
        mock_config_instance.to_dict.return_value = {"architectures": ["BertForSequenceClassification"]}
        mock_auto_config.from_pretrained.return_value = mock_config_instance

        config = load_model_config("dummy-model-id")
        assert config == {"architectures": ["BertForSequenceClassification"]}
        mock_auto_config.from_pretrained.assert_called_with("dummy-model-id", trust_remote_code=True)

    @patch("rktransformers.exporters.rknn.utils.AutoConfig")
    def test_load_model_config_failure(self, mock_auto_config: MagicMock) -> None:
        mock_auto_config.from_pretrained.side_effect = Exception("Failed to load")

        # Should return empty dict on failure
        config = load_model_config("non-existent-model")
        assert config == {}
