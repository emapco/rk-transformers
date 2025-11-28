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

import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rktransformers import RKCrossEncoder
from rktransformers.utils.env_utils import is_rockchip_platform
from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
    is_rknn_toolkit_lite_available,
)


@pytest.mark.requires_rknpu
class TestRKCrossEncoder:
    """Tests for RKNN backend integration with CrossEncoder."""

    def test_rk_cross_encoder(self) -> None:
        """Test using RKCrossEncoder derived class."""
        with (
            patch("rktransformers.integrations.sentence_transformers.load_rknn_model") as mock_load_rknn,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_cls,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
        ):
            mock_config = MagicMock()
            mock_config.sentence_transformers = {}
            mock_config.sbert_ce_default_activation_function = None
            mock_config.architectures = ["BertForSequenceClassification"]
            mock_config.max_position_embeddings = 512
            mock_config_cls.return_value = mock_config

            mock_tokenizer = MagicMock()
            mock_tokenizer.model_max_length = 512
            mock_tokenizer_cls.return_value = mock_tokenizer

            mock_rknn_model = MagicMock()
            mock_rknn_model.config = mock_config
            mock_load_rknn.return_value = mock_rknn_model

            try:
                model = RKCrossEncoder(
                    "dummy-model",
                    model_kwargs={"file_name": "model.rknn"},
                )
                mock_load_rknn.assert_called_once()
                assert model.backend == "rknn"

            except Exception as e:
                pytest.fail(f"Failed to load mocked RKCrossEncoder: {e}")

    def test_predict_rknn_padding(self) -> None:
        """Test predict method enforces max_length padding."""

        class MockBatchEncoding(dict):
            def to(self, device):
                return self

        with (
            patch("rktransformers.integrations.sentence_transformers.load_rknn_model") as mock_load_rknn,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls,
            patch("transformers.AutoConfig.from_pretrained") as mock_config_cls,
        ):
            mock_config = MagicMock()
            mock_config.num_labels = 1
            # Ensure no activation function is accidentally found
            mock_config.sbert_ce_default_activation_function = None
            mock_config.sentence_transformers = {}
            mock_config_cls.return_value = mock_config

            mock_rknn_model = MagicMock()
            mock_rknn_model.config = mock_config  # Link config so it shares the deletions
            mock_rknn_model.config.max_position_embeddings = 128
            mock_rknn_model.device = "cpu"
            mock_rknn_model.return_value = MagicMock(logits=torch.tensor([[0.5]]))  # Mock forward pass output
            mock_load_rknn.return_value = mock_rknn_model

            mock_tokenizer = MagicMock()
            mock_tokenizer.model_max_length = 128
            # Return MockBatchEncoding which has .to() method
            mock_tokenizer.return_value = MockBatchEncoding(
                {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
            )
            mock_tokenizer_cls.return_value = mock_tokenizer

            model = RKCrossEncoder(
                "dummy-model",
                model_kwargs={"file_name": "model.rknn"},
            )

            # Inject _rknn_config to trigger RKNN specific logic in predict
            model._rknn_config = {"batch_size": 1}

            pairs = [["test", "pair"]]
            model.predict(pairs)

            call_args = mock_tokenizer.call_args
            assert call_args is not None
            assert call_args.kwargs.get("padding") == "max_length"
            assert call_args.kwargs.get("max_length") == 128


pytestmark = pytest.mark.skipif(
    not is_rockchip_platform() or (not is_rknn_toolkit_lite_available() and not is_rknn_toolkit_available()),
    reason="Skipping RKNN tests on non-Rockchip platform or missing RKNN Toolkit Lite library.",
)

INTEGRATION_CROSS_ENCODER_MODEL = "rk-transformers/ms-marco-MiniLM-L12-v2"


# RKNN backend gets overloaded when running all tests at once
@pytest.mark.flaky(reruns=1, reruns_delay=random.uniform(15, 30), only_rerun=["RuntimeError"])
@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.requires_rknpu
class TestRKCrossEncoderIntegration:
    """Integration tests for RKNN backend with CrossEncoder."""

    @pytest.mark.parametrize(
        "model_id,file_name,batch_size",
        [
            (INTEGRATION_CROSS_ENCODER_MODEL, None, 1),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s512.rknn", 1),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s256.rknn", 1),
            (INTEGRATION_CROSS_ENCODER_MODEL, None, 2),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s512.rknn", 2),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s256.rknn", 2),
            (INTEGRATION_CROSS_ENCODER_MODEL, None, 3),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s512.rknn", 3),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s256.rknn", 3),
            (INTEGRATION_CROSS_ENCODER_MODEL, None, 4),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s512.rknn", 4),
            (INTEGRATION_CROSS_ENCODER_MODEL, "model_b4_s256.rknn", 4),
        ],
    )
    def test_inference_with_file_names_and_batch_sizes(self, model_id: str, file_name: str | None, batch_size: int):
        """Test inference with RKNN backend on real models with different file names and batch sizes.

        Args:
            model_id: Hugging Face model identifier
            file_name: Specific RKNN file to load (or None for default)
            batch_size: Batch size to use for prediction
        """
        from rktransformers import RKCrossEncoder

        # Build model_kwargs with file_name if provided
        model_kwargs = {"platform": "rk3588", "core_mask": "auto"}
        if file_name:
            model_kwargs["file_name"] = file_name

        model = RKCrossEncoder(
            model_id,
            model_kwargs=model_kwargs,
        )

        pairs = [
            ["How old are you?", "What is your age?"],
            ["Hello world", "Hi there!"],
            ["How old are you?", "What is your age?"],
        ]
        scores = model.predict(pairs, batch_size=batch_size)
        assert len(scores) == len(pairs), f"Expected {len(pairs)} scores, got {len(scores)}"
        assert all(isinstance(score, (int, float, np.floating)) for score in scores), "Scores should be numeric"
