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

"""Tests for RKNN backend integration with CrossEncoder."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sentence_transformers import CrossEncoder

from rktransformers import patch_cross_encoder
from rktransformers.utils.env_utils import is_rockchip_platform
from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
    is_rknn_toolkit_lite_available,
)


@pytest.fixture(scope="module", autouse=True)
def setup_module() -> None:
    """Apply the CrossEncoder patch before running tests."""
    patch_cross_encoder()


@pytest.mark.requires_rknpu
class TestCrossEncoderPatch:
    """Tests for RKNN backend integration with CrossEncoder."""

    def test_load_rknn_cross_encoder(self) -> None:
        """Test loading CrossEncoder with backend='rknn'."""
        with (
            patch("rktransformers.load.load_rknn_model") as mock_load_rknn,
            patch("rktransformers.load.cached_file") as mock_cached_file,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_cls,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
            patch("sentence_transformers.cross_encoder.CrossEncoder.load_file_path") as mock_load_file_path,
        ):
            mock_config = MagicMock()
            # Setting sentence_transformers to empty dict avoids "version" check
            mock_config.sentence_transformers = {}
            # Ensure no activation function is accidentally found
            mock_config.sbert_ce_default_activation_function = None
            # Also ensure architectures is list of strings or None
            mock_config.architectures = ["BertForSequenceClassification"]
            # Set max_position_embeddings to int
            mock_config.max_position_embeddings = 512
            mock_config_cls.return_value = mock_config

            mock_tokenizer = MagicMock()
            mock_tokenizer.model_max_length = 512
            mock_tokenizer_cls.return_value = mock_tokenizer

            # Mock load_file_path to return None (no README)
            mock_load_file_path.return_value = None

            # Mock load_rknn_model to avoid actual RKNN loading
            mock_rknn_model = MagicMock()
            mock_rknn_model.config = mock_config  # Link config
            mock_load_rknn.return_value = mock_rknn_model

            # Mock cached_file to return a dummy rknn.json
            mock_cached_file.return_value = None  # Simulate no rknn.json found via cached_file

            try:
                model = CrossEncoder(
                    "dummy-model",
                    backend="rknn",  # type: ignore
                    model_kwargs={"file_name": "model.rknn"},
                )

                # Verify load_rknn_model was called
                mock_load_rknn.assert_called_once()
                assert model.backend == "rknn"

            except Exception as e:
                pytest.fail(f"Failed to load mocked RKNN CrossEncoder: {e}")

    def test_predict_rknn_padding(self) -> None:
        """Test predict method enforces max_length padding."""

        class MockBatchEncoding(dict):
            def to(self, device):
                return self

        with (
            patch("rktransformers.load.load_rknn_model") as mock_load_rknn,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_cls,
            patch("sentence_transformers.cross_encoder.CrossEncoder.load_file_path") as mock_load_file_path,
        ):
            # Mock AutoConfig
            mock_config = MagicMock()
            mock_config.num_labels = 1
            # Ensure no activation function is accidentally found
            mock_config.sbert_ce_default_activation_function = None
            mock_config.sentence_transformers = {}
            mock_config_cls.return_value = mock_config

            # Mock load_file_path
            mock_load_file_path.return_value = None

            # Mock load_rknn_model
            mock_rknn_model = MagicMock()
            mock_rknn_model.config = mock_config  # Link config so it shares the deletions
            mock_rknn_model.config.max_position_embeddings = 128
            mock_rknn_model.device = "cpu"
            mock_rknn_model.return_value = MagicMock(logits=torch.tensor([[0.5]]))  # Mock forward pass output
            mock_load_rknn.return_value = mock_rknn_model

            # Mock AutoTokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.model_max_length = 128
            # Return MockBatchEncoding which has .to() method
            mock_tokenizer.return_value = MockBatchEncoding(
                {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
            )
            mock_tokenizer_cls.return_value = mock_tokenizer

            model = CrossEncoder(
                "dummy-model",
                backend="rknn",  # type: ignore
                model_kwargs={"file_name": "model.rknn"},
            )

            # Inject _rknn_config to trigger RKNN specific logic in predict
            model._rknn_config = {"batch_size": 1}  # type: ignore

            pairs = [["test", "pair"]]
            model.predict(pairs)

            # Verify tokenizer was called with padding="max_length"
            # CrossEncoder.predict calls tokenizer(batch, ...)
            # We need to check the call args of mock_tokenizer
            call_args = mock_tokenizer.call_args
            assert call_args is not None
            assert call_args.kwargs.get("padding") == "max_length"
            assert call_args.kwargs.get("max_length") == 128


pytestmark = pytest.mark.skipif(
    not is_rockchip_platform() or (not is_rknn_toolkit_lite_available() and not is_rknn_toolkit_available()),
    reason="Skipping RKNN tests on non-Rockchip platform or missing RKNN Toolkit Lite library.",
)


@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.requires_rknpu
class TestCrossEncoderPatchIntegration:
    """Integration tests for RKNN backend with CrossEncoder."""

    def test_inference(self):
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            "rk-transformers/bge-reranker-base",
            backend="rknn",
            model_kwargs={"platform": "rk3588", "core_mask": "auto"},
        )

        pairs = [["How old are you?", "What is your age?"], ["Hello world", "Hi there!"]]
        scores = model.predict(pairs)
        assert len(scores) == 2, f"Expected 2 scores, got {len(scores)}"
        assert all(isinstance(score, (int, float, np.floating)) for score in scores), "Scores should be numeric"
