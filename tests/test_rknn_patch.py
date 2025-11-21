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

import os

import pytest
from sentence_transformers import SentenceTransformer

from rkruntime.load import patch_sentence_transformer
from rkruntime.utils.env_utils import is_rockchip_platform

if not is_rockchip_platform():
    pytest.skip("Skipping RKNN tests on non-Rockchip platform", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # Apply the patch
    patch_sentence_transformer()


@pytest.fixture
def model_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data/random_bert")


def test_load_rknn_model(model_path):
    """Test loading the model with backend='rknn'"""
    try:
        model = SentenceTransformer(
            model_path,
            similarity_fn_name="cosine",
            trust_remote_code=True,
            backend="rknn",  # type: ignore
        )
    except Exception as e:
        pytest.fail(f"Failed to load RKNN model: {e}")

    assert model.backend == "rknn"
    # Check if the underlying auto_model is an RKRTModel
    assert "RKRTModel" in model[0].auto_model.__class__.__name__
    # Check if max_seq_length was updated from rknn.json
    assert model.max_seq_length == 32


def test_inference(model_path):
    """Test inference with the loaded model"""
    model = SentenceTransformer(
        model_path,
        similarity_fn_name="cosine",
        trust_remote_code=True,
        backend="rknn",  # type: ignore
    )
    sentences = ["CCO", "CCN"]
    embeddings = model.encode(sentences, batch_size=1)

    assert embeddings.shape[0] == 2
    # The dimension depends on the model, usually 384 or similar.
    # We can just check it's not empty.
    assert embeddings.shape[1] > 0
