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

"""Tests for RKNN backend integration with SentenceTransformer."""

from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from rktransformers.load import patch_sentence_transformer
from rktransformers.utils.env_utils import is_rockchip_platform
from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
    is_rknn_toolkit_lite_available,
)

pytestmark = pytest.mark.skipif(
    not is_rockchip_platform() or (not is_rknn_toolkit_lite_available() and not is_rknn_toolkit_available()),
    reason="Skipping RKNN tests on non-Rockchip platform or missing RKNN Toolkit Lite library.",
)


@pytest.fixture(scope="module", autouse=True)
def setup_module() -> None:
    """Apply the SentenceTransformer patch before running tests."""
    patch_sentence_transformer()


@pytest.mark.requires_rknpu
class TestSentenceTransformersenceTransformerPatch:
    """Tests for RKNN backend integration with SentenceTransformers."""

    def test_load_rknn_model(self, random_bert_model_path: Path) -> None:
        """Test loading model with backend='rknn'."""
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )

        assert model.backend == "rknn"
        # Check if the underlying auto_model is an RKRTModel
        assert "RKRTModel" in model[0].auto_model.__class__.__name__
        # Check if max_seq_length was updated from rknn.json
        assert model.max_seq_length == 32

    def test_inference(self, random_bert_model_path: Path) -> None:
        """Test inference with the RKNN-loaded model."""
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )
        sentences = ["hello", "world"]
        embeddings = model.encode(sentences)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    @pytest.mark.parametrize(
        "sentences,batch_size",
        [
            (["hello"], 1),
            (["hello", "world"], 1),
            (["hello", "world"], 2),
            (["a", "b", "c", "d"], 2),
        ],
        ids=["single-b1", "two-b1", "two-b2", "four-b2"],
    )
    def test_inference_with_different_inputs(
        self, random_bert_model_path: Path, sentences: list[str], batch_size: int
    ) -> None:
        """Test inference with different input sizes and batch sizes.

        Args:
            random_bert_model_path: Path to the test model
            sentences: List of sentences to encode
            batch_size: Batch size for encoding
        """
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )
        embeddings = model.encode(sentences, batch_size=batch_size)

        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] > 0

    def test_model_attributes(self, random_bert_model_path: Path) -> None:
        """Test that model has expected attributes after loading."""
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )

        assert hasattr(model, "backend")
        assert hasattr(model, "max_seq_length")
        assert model.backend == "rknn"
        assert isinstance(model.max_seq_length, int)
        assert model.max_seq_length > 0

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_encode_with_rknn_model(self, random_bert_model_path: Path, batch_size: int) -> None:
        """Test that encode works correctly with RKNN backend.

        Args:
            random_bert_model_path: Path to the test model
            batch_size: Batch size for encoding
        """
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )
        sentences = ["test sentence one", "test sentence two", "test sentence three", "test sentence four"]
        embeddings = model.encode(sentences, batch_size=batch_size)

        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] > 0

    def test_encode_single_sentence(self, random_bert_model_path: Path) -> None:
        """Test encoding a single sentence with RKNN backend."""
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )
        sentence = "single test sentence"
        embedding = model.encode(sentence)

        assert embedding.ndim == 2  # [num_sentence, embedding_dim]
        assert embedding.shape[0] > 0

    def test_encode_empty_list(self, random_bert_model_path: Path) -> None:
        """Test encoding an empty list with RKNN backend."""
        model = SentenceTransformer(
            str(random_bert_model_path),
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": "rknn/model_b1_s32_o3.rknn"},
        )
        sentences = []
        embeddings = model.encode(sentences)

        assert embeddings.shape[0] == 0


integration_sentence_transformer_model_id = "rk-transformers/all-MiniLM-L6-v2"
integration_transformer_model_model_id = "rk-transformers/bert-base-uncased"


@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.requires_rknpu
class TestSentenceTransformerPatchIntegration:
    """Integration tests for RKNN backend with SentenceTransformers."""

    @pytest.mark.parametrize(
        "model_id,file_name,batch_size",
        [
            (integration_sentence_transformer_model_id, None, 1),
            (integration_sentence_transformer_model_id, "model_b4_s512.rknn", 1),
            (integration_sentence_transformer_model_id, None, 4),
            (integration_sentence_transformer_model_id, "model_b4_s512.rknn", 4),
            (integration_transformer_model_model_id, None, 1),
            (integration_transformer_model_model_id, "model_b4_s512.rknn", 1),
            (integration_transformer_model_model_id, None, 4),
            (integration_transformer_model_model_id, "model_b4_s512.rknn", 4),
        ],
    )
    def test_inference_with_file_names_and_batch_sizes(
        self, model_id: str, file_name: str | None, batch_size: int
    ) -> None:
        """Test inference with RKNN backend on real models with different batch sizes.

        Args:
            model_id: Hugging Face model identifier
            batch_size: Batch size to use for encoding
        """
        model = SentenceTransformer(
            model_id,
            backend="rknn",  # type: ignore
            model_kwargs={"file_name": file_name} if file_name else {},
        )
        sentences = ["This is a test.", "RKNN integration test.", "Another sentence.", "Final test."]
        embeddings = model.encode(sentences, batch_size=batch_size)

        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] > 0
