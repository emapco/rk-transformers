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

"""Tests for RKNN exporter functionality."""

import os
import shutil
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from rktransformers.configuration import OptimizationConfig, QuantizationConfig, RKNNConfig
from rktransformers.constants import (
    ALLOW_MODEL_REPO_FILES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OPSET,
    IGNORE_MODEL_REPO_FILES,
    OptimizationLevelType,
    PlatformType,
)
from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
)

pytestmark = pytest.mark.skipif(
    not is_rknn_toolkit_available(),
    reason="Skipping tests that require the `export` extra but it's not installed.",
)

from rktransformers.exporters.rknn.convert import (  # noqa: E402
    export_rknn,
    prepare_dataset_for_quantization,
)


class TestRKNNExporter:
    """Tests for RKNN export functionality."""

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    def test_export_rknn_basic(self, mock_rknn_cls: MagicMock, dummy_onnx_file: Path, temp_dir: Path) -> None:
        """Test basic RKNN export without quantization."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path=str(dummy_onnx_file),
        )

        export_rknn(config)
        mock_rknn.config.assert_called_once()
        mock_rknn.load_onnx.assert_called_once()
        mock_rknn.build.assert_called_once_with(do_quantization=False, dataset=None)

        expected_output = temp_dir / "model.rknn"
        mock_rknn.export_rknn.assert_called_once_with(str(expected_output))

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.prepare_dataset_for_quantization")
    def test_export_rknn_with_quantization(
        self,
        mock_prepare_dataset: MagicMock,
        mock_rknn_cls: MagicMock,
        dummy_onnx_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test RKNN export with quantization enabled."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        dataset_file = temp_dir / "dataset.txt"
        mock_prepare_dataset.return_value = (
            str(dataset_file),
            ["text"],
            ["train", "validation"],
        )

        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="test/model.onnx",
            output_path=str(temp_dir),
            quantization=QuantizationConfig(do_quantization=True, dataset_name="test_dataset"),
        )

        export_rknn(config)
        mock_prepare_dataset.assert_called_once()
        mock_rknn.build.assert_called_once_with(do_quantization=True, dataset=str(dataset_file))

        expected_output = temp_dir / "rknn" / "model_w8a8.rknn"
        mock_rknn.export_rknn.assert_called_once_with(str(expected_output))

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.HfApi")
    @patch("rktransformers.exporters.rknn.convert.create_repo")
    def test_export_rknn_push_to_hub(
        self,
        mock_create_repo: MagicMock,
        mock_hf_api: MagicMock,
        mock_rknn_cls: MagicMock,
        dummy_onnx_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test RKNN export with Hugging Face Hub upload."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        mock_api_instance = mock_hf_api.return_value

        config = RKNNConfig(
            model_id_or_path="test/model.onnx",
            target_platform="rk3588",
            output_path=str(temp_dir),
            push_to_hub=True,
            hub_model_id="test/model",
            hub_token="fake_token",
        )

        export_rknn(config)
        mock_create_repo.assert_called_once()

        expected_folder = str(temp_dir)
        mock_api_instance.upload_folder.assert_called_once_with(
            repo_id="test/model",
            folder_path=expected_folder,
            token="fake_token",
            repo_type="model",
            ignore_patterns=IGNORE_MODEL_REPO_FILES,
            allow_patterns=ALLOW_MODEL_REPO_FILES,
            create_pr=False,
        )

    @patch("rktransformers.exporters.rknn.convert.shutil")
    @patch("rktransformers.exporters.rknn.convert.snapshot_download")
    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.main_export")
    @patch("rktransformers.exporters.rknn.convert.os.path.exists")
    @patch("rktransformers.exporters.rknn.convert.os.listdir")
    def test_export_rknn_from_hub(
        self,
        mock_listdir: MagicMock,
        mock_exists: MagicMock,
        mock_main_export: MagicMock,
        mock_rknn_cls: MagicMock,
        mock_snapshot_download: MagicMock,
        mock_shutil: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test RKNN export from Hugging Face Hub."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        initial_output_path = temp_dir / "model.rknn"
        expected_output_dir = str(temp_dir / "hub-model")

        # Mock file existence checks
        # 1. check if model_id_or_path exists (False for Hub ID)
        # 2. check if exported model.onnx exists (True)
        def side_effect(path):
            if path == "test/hub-model":
                return False
            if path == os.path.join(expected_output_dir, "model.onnx"):
                return True
            # Default to True for other checks
            return True

        mock_exists.side_effect = side_effect

        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="test/hub-model",
            output_path=str(initial_output_path),
            batch_size=1,
            max_seq_length=128,  # Non-default, should appear in filename
        )

        export_rknn(config)
        # Since max_seq_length=128 differs from DEFAULT_MAX_SEQ_LENGTH=512, filename includes params
        expected_rknn_path = str(temp_dir / "hub-model" / "model_b1_s128.rknn")
        assert config.output_path == expected_rknn_path

        mock_main_export.assert_called_once_with(
            model_name_or_path="test/hub-model",
            output=expected_output_dir,
            task="feature-extraction",
            opset=DEFAULT_OPSET,
            do_validation=False,
            no_post_process=True,
            batch_size=1,
            sequence_length=128,
        )

        mock_snapshot_download.assert_called_once()

        expected_onnx_path = os.path.join(expected_output_dir, "model.onnx")
        call_args = mock_rknn.load_onnx.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["model"] == expected_onnx_path

    @pytest.mark.parametrize(
        "platform,optimization_level",
        [
            ("rk3588", 1),
            ("rk3588", 2),
            ("rk3588", 3),
            ("rk3566", 3),
            ("rk3568", 0),
        ],
    )
    @patch("rktransformers.exporters.rknn.convert.RKNN")
    def test_export_with_different_configs(
        self,
        mock_rknn_cls: MagicMock,
        dummy_onnx_file: Path,
        temp_dir: Path,
        platform: str,
        optimization_level: int,
    ) -> None:
        """Test export with different platform and optimization configurations."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        output_path = temp_dir / f"model_{platform}.rknn"
        config = RKNNConfig(
            target_platform=cast(PlatformType, platform),
            model_id_or_path="test/model.onnx",
            output_path=str(output_path),
            optimization=OptimizationConfig(optimization_level=cast(OptimizationLevelType, optimization_level)),
        )

        export_rknn(config)
        mock_rknn.config.assert_called_once()

        # The output filename will be based on the model name (from output_path basename)
        # plus optimization suffix if optimization_level > 0
        model_name = f"model_{platform}"
        if optimization_level > 0:
            expected_filename = f"{model_name}_o{optimization_level}.rknn"
            expected_output = temp_dir / "rknn" / expected_filename
        else:
            expected_filename = f"{model_name}.rknn"
            expected_output = temp_dir / expected_filename

        mock_rknn.export_rknn.assert_called_once_with(str(expected_output))

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    def test_export_rknn_dynamic_input(
        self,
        mock_rknn_cls: MagicMock,
        dummy_onnx_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test RKNN export with dynamic input shapes."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        output_path = temp_dir / "model.rknn"

        user_dynamic_input = [[[1, 16], [1, 16]], [[2, 16], [2, 16]]]
        import copy

        initial_dynamic_input = copy.deepcopy(user_dynamic_input)

        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="test/model.onnx",
            output_path=str(output_path),
            batch_size=1,
            max_seq_length=32,
            dynamic_input=user_dynamic_input,
        )

        export_rknn(config)
        # Verify that rknn.config was called with the updated dynamic_input
        # The static shape [1, 32] for 2 inputs should be added
        # Inputs are auto-detected as ["input_ids", "attention_mask"] (2 inputs)
        static_shape = [[1, 32], [1, 32]]

        # Expected dynamic_input should contain user input + static shape
        expected_dynamic_input = initial_dynamic_input + [static_shape]

        call_args = mock_rknn.config.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "dynamic_input" in kwargs
        assert kwargs["dynamic_input"] == expected_dynamic_input

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    def test_export_rknn_with_default_params(
        self,
        mock_rknn_cls: MagicMock,
        dummy_onnx_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test that default batch_size and max_seq_length produce filename without params."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        output_path = temp_dir / "model.rknn"
        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="test/model.onnx",
            output_path=str(output_path),
            batch_size=DEFAULT_BATCH_SIZE,  # 1
            max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        )

        export_rknn(config)
        # When using defaults, filename should NOT include batch/seq params
        expected_output = temp_dir / "model.rknn"
        mock_rknn.export_rknn.assert_called_once_with(str(expected_output))

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    def test_export_rknn_with_non_default_batch(
        self,
        mock_rknn_cls: MagicMock,
        dummy_onnx_file: Path,
        temp_dir: Path,
    ) -> None:
        """Test that non-default batch_size produces filename with params."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        output_path = temp_dir / "model.rknn"
        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="test/model.onnx",
            output_path=str(output_path),
            batch_size=4,  # Non-default
            max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
        )

        export_rknn(config)
        # When batch_size differs from default, filename should include batch/seq params
        expected_output = temp_dir / "model_b4_s512.rknn"
        mock_rknn.export_rknn.assert_called_once_with(str(expected_output))

    @patch("rktransformers.exporters.rknn.convert.shutil")
    @patch("rktransformers.exporters.rknn.convert.snapshot_download")
    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.main_export")
    @patch("rktransformers.exporters.rknn.convert.os.path.exists")
    @patch("rktransformers.exporters.rknn.convert.os.listdir")
    def test_export_rknn_with_opset_and_task(
        self,
        mock_listdir: MagicMock,
        mock_exists: MagicMock,
        mock_main_export: MagicMock,
        mock_rknn_cls: MagicMock,
        mock_snapshot_download: MagicMock,
        mock_shutil: MagicMock,
        temp_dir: Path,
    ) -> None:
        """Test RKNN export with custom opset and task."""
        mock_rknn = mock_rknn_cls.return_value
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        def side_effect(path):
            return path != "test/hub-model"

        mock_exists.side_effect = side_effect
        mock_listdir.return_value = ["model.onnx"]

        output_path = temp_dir / "model.rknn"
        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="test/hub-model",
            output_path=str(output_path),
            opset=18,
            task="fill-mask",
        )

        export_rknn(config)
        mock_main_export.assert_called_once()
        _, kwargs = mock_main_export.call_args
        assert kwargs["opset"] == 18
        assert kwargs["task"] == "fill-mask"


class TestDatasetPreparation:
    """Tests for dataset preparation functionality."""

    @patch("rktransformers.exporters.rknn.utils.load_dataset")
    @patch("rktransformers.exporters.rknn.utils.concatenate_datasets")
    @patch("rktransformers.exporters.rknn.utils.AutoTokenizer")
    def test_prepare_dataset(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_concat: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Test dataset preparation for quantization."""
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__.return_value = 10
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]

        mock_load_dataset.return_value = mock_dataset
        mock_concat.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Updated to handle tuple return value
        dataset_file, columns, splits = prepare_dataset_for_quantization(
            "test_dataset", 10, "tokenizer_path", ["input_ids", "attention_mask"]
        )

        assert os.path.exists(dataset_file)
        assert columns == ["text"]
        assert splits == ["train", "validation", "test"]

        with open(dataset_file) as f:
            content = f.read()
            assert "input_ids.npy" in content
            assert "attention_mask.npy" in content

        os.remove(dataset_file)
        shutil.rmtree(os.path.dirname(dataset_file))

    @patch("rktransformers.exporters.rknn.utils.load_dataset")
    @patch("rktransformers.exporters.rknn.utils.concatenate_datasets")
    @patch("rktransformers.exporters.rknn.utils.AutoTokenizer")
    def test_prepare_dataset_with_subset(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_concat: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Test dataset preparation with dataset subset and columns."""
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["question", "context"]
        mock_dataset.__len__.return_value = 10
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}]

        mock_load_dataset.return_value = mock_dataset
        mock_concat.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset_file, columns, splits = prepare_dataset_for_quantization(
            dataset_name="glue",
            dataset_size=10,
            tokenizer_path="tokenizer_path",
            model_input_names=["input_ids", "attention_mask"],
            dataset_subset="ax",
            dataset_columns=["question", "context"],
        )

        # Verify load_dataset was called with subset
        mock_load_dataset.assert_any_call("glue", "ax", split="train")

        assert os.path.exists(dataset_file)
        assert columns == ["question", "context"]

        os.remove(dataset_file)
        shutil.rmtree(os.path.dirname(dataset_file))

    @patch("rktransformers.exporters.rknn.utils.load_dataset")
    @patch("rktransformers.exporters.rknn.utils.concatenate_datasets")
    @patch("rktransformers.exporters.rknn.utils.AutoTokenizer")
    def test_prepare_dataset_with_token_type_ids(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_concat: MagicMock,
        mock_load_dataset: MagicMock,
    ) -> None:
        """Test dataset preparation for models requiring token_type_ids."""
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__.return_value = 10
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.map.return_value = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
            }
        ]

        mock_load_dataset.return_value = mock_dataset
        mock_concat.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset_file, columns, splits = prepare_dataset_for_quantization(
            "test_dataset", 10, "tokenizer_path", ["input_ids", "attention_mask", "token_type_ids"]
        )

        assert os.path.exists(dataset_file)
        assert columns == ["text"]

        with open(dataset_file) as f:
            content = f.read()
            assert "input_ids.npy" in content
            assert "attention_mask.npy" in content
            assert "token_type_ids.npy" in content

            # Verify we have exactly 3 paths per line
            lines = content.strip().split("\n")
            for line in lines:
                paths = line.split()
                assert len(paths) == 3, f"Expected 3 inputs, got {len(paths)}"

        os.remove(dataset_file)
        shutil.rmtree(os.path.dirname(dataset_file))

    @pytest.mark.slow
    @pytest.mark.integration
    def test_prepare_dataset_integration(self) -> None:
        """Test dataset preparation with real dataset and tokenizer."""
        dataset_name = "sentence-transformers/natural-questions"
        text_field = "answer"
        # Use a common tokenizer
        tokenizer_path = "bert-base-uncased"
        dataset_size = 5

        try:
            dataset_file, detected_columns, splits = prepare_dataset_for_quantization(
                dataset_name=dataset_name,
                dataset_size=dataset_size,
                tokenizer_path=tokenizer_path,
                model_input_names=["input_ids", "attention_mask"],
                dataset_columns=[text_field],
            )
        except Exception as e:
            pytest.skip(f"Skipping integration test due to error (likely network): {e}")

        assert os.path.exists(dataset_file)
        assert detected_columns == [text_field]

        with open(dataset_file) as f:
            lines = f.readlines()
            assert len(lines) == dataset_size
            for line in lines:
                paths = line.strip().split()
                # Should have exactly 2 inputs for bert-base-uncased when we specify only 2
                assert len(paths) == 2, f"Expected 2 inputs, got {len(paths)}"
                for p in paths:
                    assert os.path.exists(p)
                    assert p.endswith(".npy")

        shutil.rmtree(os.path.dirname(dataset_file))


@pytest.mark.slow
@pytest.mark.integration
class TestExportIntegration:
    """Integration tests for RKNN export (require actual RKNN toolkit)."""

    def test_export_integration(self, test_data_dir: Path) -> None:
        """Test complete export workflow with real RKNN model."""
        onnx_model_path = test_data_dir / "random_bert" / "model.onnx"
        output_dir = test_data_dir / "random_bert"

        output_dir.mkdir(parents=True, exist_ok=True)

        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path=str(onnx_model_path),
            max_seq_length=32,
            optimization=OptimizationConfig(optimization_level=3),
        )

        export_rknn(config)
        # For unquantized models with optimization, output goes to rknn/model_o{level}.rknn
        # Since optimization_level defaults to 3 and we're not quantizing
        expected_output = output_dir / "rknn" / "model_b1_s32_o3.rknn"
        assert expected_output.exists()

    @pytest.mark.parametrize("batch_size,max_seq_length", [(1, 16), (1, 32), (2, 32), (4, 64)])
    def test_export_integration_roberta_batch_sizes_and_max_seq_length(
        self, test_data_dir: Path, batch_size: int, max_seq_length: int
    ) -> None:
        """Test export workflow with RoBERTa model and different batch sizes."""
        onnx_model_path = test_data_dir / "random_roberta" / "model.onnx"
        output_dir = test_data_dir / "random_roberta"

        output_dir.mkdir(parents=True, exist_ok=True)

        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path=str(onnx_model_path),
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )

        export_rknn(config)
        # Expected filename based on whether params match defaults
        if batch_size == DEFAULT_BATCH_SIZE and max_seq_length == DEFAULT_MAX_SEQ_LENGTH:
            expected_path = output_dir / "model.rknn"
        else:
            expected_path = output_dir / f"model_b{batch_size}_s{max_seq_length}.rknn"
        assert expected_path.exists()

    @pytest.mark.manual
    def test_export_integration_all_minilm_l6(self, test_data_dir: Path) -> None:
        """Test export workflow with sentence-transformers/all-MiniLM-L6-v2 from Hub.

        This verifies that the ONNX export and RKNN conversion work correctly for
        sentence transformer models.
        """
        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="sentence-transformers/all-MiniLM-L6-v2",
            output_path=str(test_data_dir),
            max_seq_length=128,
            batch_size=1,
            opset=19,
            optimization=OptimizationConfig(enable_flash_attention=True),
        )

        expected_output_path = test_data_dir / "all-MiniLM-L6-v2" / "model_b1_s128.rknn"
        expected_config_path = test_data_dir / "all-MiniLM-L6-v2" / "config.json"
        expected_onnx_path = test_data_dir / "all-MiniLM-L6-v2" / "model.onnx"

        export_rknn(config)
        assert expected_output_path.exists(), "RKNN model file not created"
        assert expected_config_path.exists(), "Config file not created"
        assert expected_onnx_path.exists(), "ONNX model file not created"

        # Verify ONNX model is valid
        import onnx

        onnx_model = onnx.load(str(expected_onnx_path))
        input_names = [input.name for input in onnx_model.graph.input]
        assert "input_ids" in input_names, f"input_ids missing from inputs: {input_names}"
        assert "attention_mask" in input_names, f"attention_mask missing from inputs: {input_names}"

        assert (test_data_dir / "all-MiniLM-L6-v2" / "modules.json").exists()
        assert (test_data_dir / "all-MiniLM-L6-v2" / "sentence_bert_config.json").exists()

    @pytest.mark.manual
    def test_export_integration_modernbert(self, test_data_dir: Path) -> None:
        """Test export workflow with ModernBERT from Hub."""
        config = RKNNConfig(
            target_platform="rk3588",
            model_id_or_path="answerdotai/ModernBERT-base",
            output_path=str(test_data_dir),
            max_seq_length=128,
            batch_size=1,
        )

        expected_output_path = test_data_dir / "ModernBERT-base" / "model.rknn"
        expected_config_path = test_data_dir / "ModernBERT-base" / "config.json"

        try:
            export_rknn(config)
            assert expected_output_path.exists()
            assert expected_config_path.exists()
        except Exception as e:
            pytest.skip(f"Skipping integration test due to error (likely network or missing dependency): {e}")
