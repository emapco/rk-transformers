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

import sys
import unittest
from unittest.mock import MagicMock, patch

from transformers import PretrainedConfig

from rkruntime.load import load_rknn_model

# Mock rknnlite.api before importing rkruntime.modeling
mock_rknnlite = MagicMock()
mock_rknnlite.__spec__ = MagicMock()
sys.modules["rknnlite"] = mock_rknnlite

mock_rknnlite_api = MagicMock()
mock_rknnlite_api.__spec__ = MagicMock()
sys.modules["rknnlite.api"] = mock_rknnlite_api


class TestLoadRKNNModel(unittest.TestCase):
    @patch("rkruntime.modeling.RKRTModelForFeatureExtraction.from_pretrained")
    def test_load_feature_extraction(self, mock_from_pretrained):
        config = PretrainedConfig()
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_rknn_model("test-model", config, "feature-extraction")

        mock_from_pretrained.assert_called_once_with("test-model", config=config)
        self.assertEqual(model, mock_from_pretrained.return_value)

    @patch("rkruntime.modeling.RKRTModelForMaskedLM.from_pretrained")
    def test_load_masked_lm(self, mock_from_pretrained):
        config = PretrainedConfig()
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_rknn_model("test-model", config, "fill-mask")

        mock_from_pretrained.assert_called_once_with("test-model", config=config)
        self.assertEqual(model, mock_from_pretrained.return_value)

    @patch("rkruntime.modeling.RKRTModelForSequenceClassification.from_pretrained")
    def test_load_sequence_classification(self, mock_from_pretrained):
        config = PretrainedConfig()
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_rknn_model("test-model", config, "sequence-classification")

        mock_from_pretrained.assert_called_once_with("test-model", config=config)
        self.assertEqual(model, mock_from_pretrained.return_value)

    def test_unsupported_task(self):
        config = PretrainedConfig()
        with self.assertRaises(ValueError):
            load_rknn_model("test-model", config, "unsupported-task")


if __name__ == "__main__":
    unittest.main()
