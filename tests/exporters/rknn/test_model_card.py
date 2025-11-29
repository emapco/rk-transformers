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

import json
import os
from pathlib import Path
from typing import cast

import pytest

from rktransformers.configuration import OptimizationConfig, QuantizationConfig, RKNNConfig
from rktransformers.constants import TASK_TO_RK_MODEL_CLASS, SupportedTaskType
from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
)

pytestmark = pytest.mark.skipif(
    not is_rknn_toolkit_available(),
    reason="Skipping tests that require the `export` extra but it's not installed.",
)

from rktransformers.exporters.rknn.model_card import ModelCardGenerator  # noqa: E402


class TestModelCardGenerator:
    """Tests for ModelCardGenerator."""

    @pytest.fixture(autouse=True)
    def setup_method(self, temp_dir: Path):
        self.output_dir = str(temp_dir)
        self.config = RKNNConfig(
            model_name_or_path="test-model",
            output_path=os.path.join(self.output_dir, "model.rknn"),
            target_platform="rk3588",
        )
        self.generator = ModelCardGenerator()

    def test_generate_new_readme(self):
        output_file = self.generator.generate(self.config, self.output_dir)
        assert output_file == os.path.join(self.output_dir, "README.md")
        assert output_file is not None
        assert os.path.exists(output_file)
        with open(output_file) as f:
            content = f.read()
            assert "library_name: rk-transformers" in content
            assert "test-model" in content

    def test_generate_existing_non_rknn_readme(self):
        readme_path = os.path.join(self.output_dir, "README.md")
        original_content = "# Original README\n\nSome content."
        with open(readme_path, "w") as f:
            f.write(original_content)

        output_file = self.generator.generate(self.config, self.output_dir)
        assert output_file == os.path.join(self.output_dir, "README.md")
        assert output_file is not None
        with open(output_file) as f:
            content = f.read()
            # Check for header
            assert "# test-model (RKNN2)" in content
            # Check for details tag
            assert "<details><summary>Click to see the RKNN model details and usage examples</summary>" in content
            assert "</details>" in content
            # Check for original content
            assert original_content in content
            # Check that details are inside the details block (rough check)
            assert content.index("<details>") < content.index("## Model Details")
            assert content.index("</details>") > content.index("## Model Details")
            # Check that original content is after details
            assert content.index("</details>") < content.index("# Original README")

    def test_generate_existing_rknn_readme(self):
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("---\nlibrary_name: rk-transformers\n---\n# Old RKNN README")

        output_file = self.generator.generate(self.config, self.output_dir)
        assert output_file == os.path.join(self.output_dir, "README.md")
        assert output_file is not None
        with open(output_file) as f:
            content = f.read()
            assert "library_name: rk-transformers" in content
            # Should overwrite
            assert "Old RKNN README" not in content
            assert "# test-model (RKNN2)" in content

    def test_generate_multiple_models(self):
        # Create dummy model files
        with open(os.path.join(self.output_dir, "model.rknn"), "w") as f:
            f.write("dummy")

        # Create dummy sentence transformer config
        with open(os.path.join(self.output_dir, "config_sentence_transformers.json"), "w") as f:
            f.write("{}")

        rknn_dir = os.path.join(self.output_dir, "rknn")
        os.makedirs(rknn_dir)
        with open(os.path.join(rknn_dir, "model_quantized.rknn"), "w") as f:
            f.write("dummy")

        output_file = self.generator.generate(self.config, self.output_dir)
        assert output_file is not None

        with open(output_file) as f:
            content = f.read()
            # model.rknn is default, so it shouldn't be explicitly mentioned in usage as file_name
            assert 'file_name="model.rknn"' not in content
        # rknn/model_quantized.rknn should be explicitly mentioned in usage
        assert "rknn/model_quantized.rknn" in content

    def test_generate_available_models_table(self):
        """Test that the available models table is generated correctly."""
        # Create dummy model files
        with open(os.path.join(self.output_dir, "model.rknn"), "w") as f:
            f.write("dummy")

        rknn_dir = os.path.join(self.output_dir, "rknn")
        os.makedirs(rknn_dir)
        with open(os.path.join(rknn_dir, "model_w8a8.rknn"), "w") as f:
            f.write("dummy")

        # Create config.json with rknn key
        rknn_config = {
            "rknn": {
                "model.rknn": {
                    "optimization": {"optimization_level": 0},
                    "quantization": {"do_quantization": False},
                },
                "rknn/model_w8a8.rknn": {
                    "optimization": {"optimization_level": 2},
                    "quantization": {"do_quantization": True, "quantized_dtype": "w8a8"},
                },
            }
        }
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(rknn_config, f)

        output_file = self.generator.generate(self.config, self.output_dir)
        assert output_file is not None

        with open(output_file) as f:
            content = f.read()
            # Check for table header
            assert "| Model File | Optimization Level | Quantization | File Size |" in content
            # Check for table separator (HuggingFace hub formats with width-matched dashes)
            assert "|" in content and "---" in content  # Basic separator check

            # Check for model entries parts
            assert "[model.rknn](./model.rknn)" in content
            assert "| 0 | float16 | 5.0 B |" in content

            assert "[rknn/model_w8a8.rknn](./rknn/model_w8a8.rknn)" in content
            assert "| 2 | w8a8 | 5.0 B |" in content

    def test_generate_readme(self, temp_dir: Path) -> None:
        """Test generating a README file."""
        output_path = str(temp_dir / "model.rknn")
        # Create dummy model file
        with open(output_path, "w") as f:
            f.write("dummy")

        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path="test/model.onnx",
            output_path=output_path,
            optimization=OptimizationConfig(optimization_level=2),
        )

        generator = ModelCardGenerator()
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        assert os.path.exists(readme_path)
        assert os.path.basename(readme_path) == "README.md"

        with open(readme_path) as f:
            content = f.read()
            assert "model.onnx" in content
            assert "rk3588" in content
            # Check for table entry instead of individual fields
            # We don't know exact file size here (it depends on export), so just check parts
            assert "| [model.rknn](./model.rknn) | 2 | float16 |" in content

    def test_generate_readme_quantized(self, temp_dir: Path) -> None:
        """Test generating a README file for quantized model."""
        # Create a dummy local model file in a specific directory
        model_dir = temp_dir / "my_model"
        model_dir.mkdir()
        model_path = model_dir / "model.onnx"
        model_path.touch()

        output_path = str(temp_dir / "model_quant.rknn")
        # Create dummy model file
        with open(output_path, "w") as f:
            f.write("dummy")

        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path=str(model_path),
            output_path=output_path,
            quantization=QuantizationConfig(
                do_quantization=True,
                quantized_dtype="w8a8",
                dataset_name="sentence-transformers/natural-questions",
            ),
        )

        generator = ModelCardGenerator()
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        with open(readme_path) as f:
            content = f.read()
            # Check for table entry
            assert "| [model_quant.rknn](./model_quant.rknn) | 0 | w8a8 |" in content
            assert "datasets:" in content
            assert "- sentence-transformers/natural-questions" in content
            assert "# my_model (RKNN2)" in content  # Title check for local file in 'my_model' directory

    def test_generate_readme_st_conditional(self, temp_dir: Path) -> None:
        """Test that Sentence Transformers usage is shown only when config exists."""
        # Create a dummy local model directory with ST config
        model_dir = temp_dir / "st_model"
        model_dir.mkdir()
        (model_dir / "modules.json").touch()

        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path=str(model_dir),
            output_path=str(temp_dir / "st_model.rknn"),
        )

        generator = ModelCardGenerator()
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        with open(readme_path) as f:
            content = f.read()
            assert "Sentence Transformers" in content
            assert "RKSentenceTransformer" in content

    def test_generate_readme_no_st(self, temp_dir: Path) -> None:
        """Test that Sentence Transformers usage is HIDDEN when config is missing."""
        # Create a dummy local model directory WITHOUT ST config
        model_dir = temp_dir / "plain_model"
        model_dir.mkdir()

        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path=str(model_dir),
            output_path=str(temp_dir / "plain_model.rknn"),
        )

        generator = ModelCardGenerator()
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        with open(readme_path) as f:
            content = f.read()
            assert "Sentence Transformers" not in content
            assert "RKSentenceTransformer" not in content

    def test_generate_readme_dynamic_task(self, temp_dir: Path) -> None:
        """Test that the correct RKModel class is used based on the task."""
        for task, expected_class in TASK_TO_RK_MODEL_CLASS.items():
            config = RKNNConfig(
                target_platform="rk3588",
                model_name_or_path="test/model.onnx",
                output_path=str(temp_dir / f"model_{task}.rknn"),
                task=cast(SupportedTaskType, task),
            )

            generator = ModelCardGenerator()
            readme_path = generator.generate(config, str(temp_dir))
            assert readme_path is not None

            with open(readme_path) as f:
                content = f.read()
                assert expected_class in content
                # Check for new usage pattern
                assert f"model = {expected_class}.from_pretrained(" in content
                # Should use model name since it's local
                assert '"model"' in content or "model.onnx" in content
                assert f'file_name="model_{task}.rknn"' in content  # Should specify file_name since it's not model.rknn

    def test_generate_readme_unsupported_task(self, temp_dir: Path) -> None:
        """Test that unsupported tasks show the correct message."""
        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path="test/model.onnx",
            output_path=str(temp_dir / "model_unsupported.rknn"),
            task=cast(SupportedTaskType, "unknown-task"),
        )

        generator = ModelCardGenerator()
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        with open(readme_path) as f:
            content = f.read()
            assert "Unsupported RKNN Model" in content
            assert "pip install rknn-toolkit-lite2" in content
            assert "#### RK-Transformers API" not in content

    def test_generate_readme_architecture_detection(self, temp_dir: Path) -> None:
        """Test that RKModel class is detected from architecture when task is unknown."""
        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path="test/model.onnx",
            output_path=str(temp_dir / "model_arch.rknn"),
            task=cast(SupportedTaskType, "unknown-task"),
        )

        # Mock PretrainedConfig with specific architecture
        from transformers import PretrainedConfig

        pretrained_config = PretrainedConfig()
        pretrained_config.architectures = ["BertForTokenClassification"]

        generator = ModelCardGenerator(pretrained_config=pretrained_config)
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        with open(readme_path) as f:
            content = f.read()
            # Should detect RKModelForTokenClassification from BertForTokenClassification
            assert "RKModelForTokenClassification" in content
            assert "RK-Transformers API" in content
            assert "Unsupported RKNN Model" not in content

    def test_generate_readme_default_model_name(self, temp_dir: Path) -> None:
        """Test usage example when model name is default model.rknn"""
        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path="test/model.onnx",
            output_path=str(temp_dir / "model.rknn"),
        )
        generator = ModelCardGenerator()
        readme_path = generator.generate(config, str(temp_dir))
        assert readme_path is not None

        with open(readme_path) as f:
            content = f.read()
            # Should NOT have file_name argument for default model
            assert 'file_name="model.rknn"' not in content
            # Should use model name (onnx) since it's local
            assert '"model"' in content or "model.onnx" in content

    def test_generate_readme_removes_existing_card_data(self, temp_dir: Path) -> None:
        """Test that existing card data from an existing README is removed when merging.
        The following card data is removed:
            - model-index
            - datasets
            - eval_results
        """
        readme_path = str(temp_dir / "README.md")
        existing_content = """---
language: en
license: apache-2.0
datasets:
- sst2
model-index:
- name: test-model
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: sst2
      type: sst2
    metrics:
    - type: accuracy
      value: 0.95
      name: Accuracy
---
# Original Model

This is the original model README.
"""
        with open(readme_path, "w") as f:
            f.write(existing_content)

        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path="test/model.onnx",
            output_path=str(temp_dir / "model.rknn"),
        )

        generator = ModelCardGenerator()
        output_file = generator.generate(config, str(temp_dir))
        assert output_file is not None

        with open(output_file) as f:
            content = f.read()
            assert "model-index:" not in content
            assert "datasets:" not in content
            assert "eval_results" not in content
            assert "- sst2" not in content
            assert "# Original Model" in content
            assert "library_name: rk-transformers" in content
            assert "<details><summary>Click to see the RKNN model details" in content

    def test_generate_readme_replaces_datasets_with_quantization(self, temp_dir: Path) -> None:
        """Test that datasets are replaced when RKNN has quantization dataset."""
        # Create an existing README with original datasets
        readme_path = str(temp_dir / "README.md")
        existing_content = """---
language: en
license: apache-2.0
datasets:
- sst2
- glue
---
# Original Model

This is the original model README.
"""
        with open(readme_path, "w") as f:
            f.write(existing_content)

        config = RKNNConfig(
            target_platform="rk3588",
            model_name_or_path="test/model.onnx",
            output_path=str(temp_dir / "model.rknn"),
            quantization=QuantizationConfig(
                do_quantization=True,
                quantized_dtype="w8a8",
                dataset_name="sentence-transformers/natural-questions",
            ),
        )

        generator = ModelCardGenerator()
        output_file = generator.generate(config, str(temp_dir))
        assert output_file is not None

        with open(output_file) as f:
            content = f.read()
            # Verify RKNN quantization dataset replaced original datasets
            assert "datasets:" in content
            assert "- sentence-transformers/natural-questions" in content
            # Verify original datasets are NOT present
            assert "- sst2" not in content
            assert "- glue" not in content
            # Verify original content is still there (wrapped in details)
            assert "# Original Model" in content
            # Verify RKNN content was added
            assert "library_name: rk-transformers" in content
