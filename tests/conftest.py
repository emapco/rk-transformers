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

"""Shared pytest fixtures for rktransformers tests."""

import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


@pytest.fixture
def dummy_onnx_file(temp_dir: Path) -> Path:
    """Create a dummy ONNX model file."""
    model_path = temp_dir / "model.onnx"
    model_path.write_text("dummy onnx content")
    return model_path


@pytest.fixture
def dummy_rknn_file(temp_dir: Path) -> Path:
    """Create a dummy RKNN model file."""
    model_path = temp_dir / "model.rknn"
    model_path.touch()
    return model_path


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def pretrained_config() -> PretrainedConfig:
    """Create a basic pretrained config for testing."""
    return PretrainedConfig(model_type="rknn_model")


@pytest.fixture
def random_bert_model_path(test_data_dir: Path) -> Path:
    """Return path to the random BERT test model."""
    return test_data_dir / "random_bert"


@pytest.fixture
def mock_rknn_lite() -> Generator[MagicMock, None, None]:
    """Mock RKNNLite class and instance."""
    with patch("rktransformers.modeling_utils.RKNNLite") as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.load_rknn.return_value = 0
        mock_instance.init_runtime.return_value = 0
        mock_instance.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["rk3588"])]
        )
        yield mock_class
