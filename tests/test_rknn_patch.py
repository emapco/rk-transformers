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

"""Tests for RKNN patches to SentenceTransformers."""

from pathlib import Path

import pytest

from rktransformers.load import patch_sentence_transformer
from rktransformers.utils.env_utils import is_rockchip_platform
from rktransformers.utils.import_utils import is_sentence_transformers_available

# Skip all tests in this module if not on Rockchip platform
pytestmark = pytest.mark.skipif(
    not is_rockchip_platform() and not is_sentence_transformers_available(),
    reason="Skipping RKNN tests on non-Rockchip platform",
)

from sentence_transformers import SentenceTransformer  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def setup_module() -> None:
    """Apply the SentenceTransformer patch before running tests."""
    patch_sentence_transformer()


@pytest.fixture
def model_path(test_data_dir: Path) -> Path:
    """Return path to the test BERT model."""
    return test_data_dir / "random_bert"


@pytest.mark.requires_rknn
class TestRKNNPatch:
    """Tests for RKNN backend integration with SentenceTransformers."""

    def test_load_rknn_model(self, model_path: Path) -> None:
        """Test loading model with backend='rknn'."""
        model = SentenceTransformer(
            str(model_path),
            similarity_fn_name="cosine",
            trust_remote_code=True,
            backend="rknn",  # type: ignore
        )

        assert model.backend == "rknn"
        # Check if the underlying auto_model is an RKRTModel
        assert "RKRTModel" in model[0].auto_model.__class__.__name__
        # Check if max_seq_length was updated from rknn.json
        assert model.max_seq_length == 32

    def test_inference(self, model_path: Path) -> None:
        """Test inference with the RKNN-loaded model."""
        model = SentenceTransformer(
            str(model_path),
            similarity_fn_name="cosine",
            trust_remote_code=True,
            backend="rknn",  # type: ignore
        )
        sentences = ["CCO", "CCN"]
        embeddings = model.encode(sentences, batch_size=1)

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
    )
    def test_inference_with_different_inputs(self, model_path: Path, sentences: list[str], batch_size: int) -> None:
        """Test inference with different input sizes and batch sizes."""
        model = SentenceTransformer(
            str(model_path),
            similarity_fn_name="cosine",
            trust_remote_code=True,
            backend="rknn",  # type: ignore
        )
        embeddings = model.encode(sentences, batch_size=batch_size)

        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] > 0

    def test_model_attributes(self, model_path: Path) -> None:
        """Test that model has expected attributes after loading."""
        model = SentenceTransformer(
            str(model_path),
            similarity_fn_name="cosine",
            trust_remote_code=True,
            backend="rknn",  # type: ignore
        )

        assert hasattr(model, "backend")
        assert hasattr(model, "max_seq_length")
        assert model.backend == "rknn"
        assert isinstance(model.max_seq_length, int)
        assert model.max_seq_length > 0
