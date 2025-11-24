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

import contextlib
import logging
import os
import shutil
from typing import cast

from huggingface_hub import HfApi, create_repo, snapshot_download
from optimum.exporters.onnx.__main__ import main_export
from rknn.api import RKNN

from rktransformers.configuration import RKNNConfig
from rktransformers.constants import (
    ALLOW_MODEL_REPO_FILES,
    DEFAULT_MAX_SEQ_LENGTH,
    IGNORE_MODEL_REPO_FILES,
    SupportedTaskType,
)
from rktransformers.exporters.rknn.model_card import ModelCardGenerator

from .utils import (
    clean_intermediate_onnx_files,
    detect_task,
    generate_rknn_output_path,
    get_onnx_input_names,
    load_model_config,
    prepare_dataset_for_quantization,
    resolve_hub_repo_id,
    store_rknn_json,
)

logger = logging.getLogger(__name__)


def export_rknn(config: RKNNConfig) -> None:
    """
    Export ONNX model or Hugging Face model to RKNN using the provided configuration.

    For Hugging Face models, this function:
    1. Exports to ONNX using Optimum (automatically detects required inputs like token_type_ids)
    2. Inspects the exported ONNX model to determine actual inputs
    3. Configures RKNN toolkit with optimization and quantization settings
    4. Loads and converts the ONNX model to RKNN format
    5. Optionally quantizes using calibration dataset
    6. Exports to RKNN format and saves complete configuration

    Args:
        config: RKNN configuration object

    Returns:
        None on success, raises RuntimeError on failure
    """
    if not config.model_id_or_path:
        raise ValueError("model_id_or_path is required in configuration")

    # Capture original model ID for model card
    base_model_id = config.model_id_or_path

    # Check if model_id_or_path is a local file
    # Consider it local if it has .onnx extension or exists on disk
    is_local_model = config.model_id_or_path.endswith(".onnx") or os.path.exists(config.model_id_or_path)

    if not is_local_model:
        logger.info(f"Model path '{config.model_id_or_path}' not found locally. Treating as Hugging Face Hub ID.")

        # Determine output directory for caching model files
        if config.output_path:
            # Check if output_path is a file or directory
            if config.output_path.endswith(".rknn"):
                # It's a file path, extract directory and filename
                base_output_dir = os.path.dirname(config.output_path)
                rknn_filename = os.path.basename(config.output_path)
            else:
                # It's a directory path
                base_output_dir = config.output_path
                rknn_filename = "model.rknn"
        else:
            base_output_dir = os.getcwd()
            rknn_filename = "model.rknn"

        # Extract model name from Hub ID (e.g. "answerdotai/ModernBERT-base" -> "ModernBERT-base")
        model_name = config.model_id_or_path.split("/")[-1]
        output_dir = os.path.join(base_output_dir, model_name)

        # Update config.output_path to be inside this new directory
        config.output_path = os.path.join(output_dir, rknn_filename)
        logger.info(f"Exporting model from Hub to directory: {output_dir}")

        try:
            # Download all model files (config, tokenizer, etc.) but exclude weights
            # This ensures we have files like modules.json, sentence_bert_config.json, etc.
            snapshot_download(
                repo_id=config.model_id_or_path,
                local_dir=output_dir,
                ignore_patterns=IGNORE_MODEL_REPO_FILES,
                allow_patterns=ALLOW_MODEL_REPO_FILES,
            )

            cache_dir = os.path.join(output_dir, ".cache")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

            # Export to ONNX using Optimum's main_export
            # Use batch_size and max_seq_length from config for input shapes
            # Resolve task if auto
            export_task = config.task
            if export_task == "auto":
                model_config = load_model_config(config.model_id_or_path)
                export_task = detect_task(model_config)
                # Update config with detected task so it's preserved
                config.task = cast(SupportedTaskType, export_task)

            main_export(
                model_name_or_path=config.model_id_or_path,
                output=output_dir,
                task=export_task,
                opset=config.opset,
                do_validation=False,
                no_post_process=True,
                batch_size=config.batch_size,
                sequence_length=config.max_seq_length or DEFAULT_MAX_SEQ_LENGTH,
            )

            # main_export creates model.onnx in the output directory
            exported_model_path = os.path.join(output_dir, "model.onnx")
            logger.info(f"Successfully exported to {exported_model_path}")
            # Update config to use the exported file
            config.model_id_or_path = exported_model_path

        except Exception as e:
            raise RuntimeError(f"Failed to export model from Hub: {e}") from e

    model_config = load_model_config(config.model_id_or_path)  # used for auto-detection

    # Auto-detect max_seq_length if not provided
    is_user_specified_seq_len = config.max_seq_length is not None
    if config.max_seq_length is None:
        config.max_seq_length = model_config.get("max_position_embeddings", DEFAULT_MAX_SEQ_LENGTH)
        logger.info(f"Auto-detected max_seq_length: {config.max_seq_length}")

    # Auto-detect type_vocab_size if not provided
    if config.type_vocab_size is None:
        config.type_vocab_size = model_config.get("type_vocab_size")
        if config.type_vocab_size:
            logger.info(f"Auto-detected type_vocab_size: {config.type_vocab_size}")

    rknn = RKNN(verbose=True)
    logger.info(f"Configuring RKNN for {config.target_platform}")
    rknn.config(**config.to_dict())

    logger.info(f"Loading ONNX model: {config.model_id_or_path}")
    sequence_length = config.max_seq_length
    batch_size = config.batch_size
    assert sequence_length is not None
    assert sequence_length >= 1, "max_seq_length must be at least 1"
    assert batch_size >= 1, "batch_size must be at least 1"

    if config.model_input_names:
        inputs = config.model_input_names
        logger.info(f"Using user-specified inputs: {inputs}")
    else:
        inputs = get_onnx_input_names(config.model_id_or_path)
        if inputs is None:
            # Fallback to heuristic based on type_vocab_size if ONNX inspection fails
            logger.warning("Failed to extract inputs from ONNX model, falling back to heuristic")
            if config.type_vocab_size and config.type_vocab_size > 1:
                inputs = ["input_ids", "attention_mask", "token_type_ids"]
                logger.info(f"Auto-detected inputs (with token_type_ids): {inputs}")
            else:
                inputs = ["input_ids", "attention_mask"]
                logger.info(f"Auto-detected inputs: {inputs}")

        config.model_input_names = inputs

    input_size_list: list[list[int]] = [[batch_size, sequence_length]] * len(inputs)
    if config.dynamic_input is not None and (input_size_list not in config.dynamic_input):
        config.dynamic_input.append(input_size_list)

    logger.info(f"Loading ONNX model into RKNN with inputs: {inputs} and sizes: {input_size_list}")
    ret = rknn.load_onnx(
        model=config.model_id_or_path,
        inputs=inputs,
        input_size_list=input_size_list,
    )
    if ret != 0:
        raise RuntimeError("Failed to load ONNX model!")

    dataset_file = None
    actual_columns = None
    actual_splits = None

    if config.quantization.do_quantization and config.quantization.dataset_name:
        model_dir = os.path.dirname(config.model_id_or_path)

        dataset_file, actual_columns, actual_splits = prepare_dataset_for_quantization(
            config.quantization.dataset_name,
            config.quantization.dataset_size,
            model_dir,
            inputs,
            sequence_length,
            config.quantization.dataset_split,
            config.quantization.dataset_subset,
            config.quantization.dataset_columns,
        )

        if actual_columns and not config.quantization.dataset_columns:
            config.quantization.dataset_columns = actual_columns
        if actual_splits and not config.quantization.dataset_split:
            config.quantization.dataset_split = actual_splits

    logger.info("Building RKNN model")
    ret = rknn.build(do_quantization=config.quantization.do_quantization, dataset=dataset_file)
    if ret != 0:
        clean_intermediate_onnx_files()
        raise RuntimeError("Failed to build RKNN model!")

    # Determine output path based on configuration
    if config.output_path:
        if config.output_path.endswith(".rknn"):
            model_dir = os.path.dirname(config.output_path)
            model_name = os.path.splitext(os.path.basename(config.output_path))[0]
        else:
            model_dir = config.output_path
            model_name = os.path.splitext(os.path.basename(config.model_id_or_path.rstrip(os.sep)))[0]
    else:
        if is_local_model:
            model_dir = os.path.dirname(os.path.abspath(config.model_id_or_path))
            model_name = os.path.splitext(os.path.basename(config.model_id_or_path))[0]
        else:
            model_dir = os.getcwd()
            model_name = "model"

    output_path, rknn_key = generate_rknn_output_path(
        config, model_dir, model_name, batch_size, sequence_length, is_user_specified_seq_len
    )
    config.output_path = output_path

    logger.info(f"Exporting RKNN model to {config.output_path}")
    ret = rknn.export_rknn(config.output_path)
    if ret != 0:
        clean_intermediate_onnx_files()
        raise RuntimeError("Failed to export RKNN model!")

    # Resolve hub_model_id early so model card uses the resolved ID
    if config.push_to_hub:
        if not config.hub_model_id:
            clean_intermediate_onnx_files()
            raise ValueError("hub_model_id is required when push_to_hub is True")
        config.hub_model_id = resolve_hub_repo_id(config.hub_model_id, config.hub_token)
    # rknn.json is required for model card generation
    if config.output_path:
        store_rknn_json(config, model_dir, rknn_key)
    generator = ModelCardGenerator()
    readme_path = generator.generate(config, model_dir, base_model_id)
    if readme_path:
        logger.info(f"Generated model card at {readme_path}")

    logger.info("Done!")
    rknn.release()

    if config.push_to_hub:
        assert config.hub_model_id is not None, "hub_model_id should be set here"
        logger.info(f"Pushing to Hub: {config.hub_model_id}")
        api = HfApi(token=config.hub_token)

        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=config.hub_model_id, token=config.hub_token, private=config.hub_private_repo, exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Failed to create/check repo: {e}")

        # Upload files
        try:
            # Upload the entire output directory, excluding original weights
            # We want to keep tokenizer, config, etc., plus the new RKNN model and rknn.json
            logger.info(f"Uploading directory {model_dir} to Hub...")

            api.upload_folder(
                repo_id=config.hub_model_id,
                folder_path=model_dir,
                token=config.hub_token,
                repo_type="model",
                ignore_patterns=IGNORE_MODEL_REPO_FILES,
                allow_patterns=ALLOW_MODEL_REPO_FILES,
                create_pr=config.hub_create_pr,
            )

            logger.info(f"Successfully pushed to Hub: {config.hub_model_id}")
        except Exception as e:
            logger.warning(f"Error uploading to Hub: {e}")
        finally:
            # Cleanup dataset file
            if dataset_file and os.path.exists(dataset_file):
                with contextlib.suppress(Exception):
                    os.remove(dataset_file)
                    # Also try to remove the temp directory
                    temp_dir = os.path.dirname(dataset_file)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

    if dataset_file and os.path.exists(dataset_file):
        with contextlib.suppress(Exception):
            os.remove(dataset_file)
            temp_dir = os.path.dirname(dataset_file)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    clean_intermediate_onnx_files()
