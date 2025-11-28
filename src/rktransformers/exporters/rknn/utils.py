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
import json
import logging
import os
import tempfile

import numpy as np
import onnx
from datasets import concatenate_datasets, load_dataset
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from rktransformers.configuration import RKNNConfig
from rktransformers.constants import (
    AUTO_DETECT_TEXT_FIELDS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH,
    TASK_TO_RK_MODEL_CLASS,
)

logger = logging.getLogger(__name__)


def download_sentence_transformer_modules_weights(
    repo_id: str,
    local_dir: str,
    token: str | None = None,
) -> bool:
    """
    Download sentence-transformers module subdirectories if modules.json exists.

    Parses modules.json and downloads module subdirectories containing weights
    (e.g., 1_Pooling, 2_Dense, 3_Dense). These subdirectories contain small
    safetensor weights that are used by sentence-transformers for additional
    model layers like pooling and dense transformations.

    Args:
        repo_id: Hugging Face repository ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        local_dir: Local directory where files should be downloaded
        token: Optional Hugging Face API token for authentication

    Returns:
        True if modules were downloaded, False otherwise (e.g., no modules.json found)
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    modules_json_path = os.path.join(local_dir, "modules.json")
    # If modules.json doesn't exist locally, try to download it first
    if not os.path.exists(modules_json_path):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="modules.json",
                local_dir=local_dir,
                token=token,
            )
        except Exception:
            # modules.json doesn't exist in the repository, not a sentence-transformers model
            return False

    try:
        with open(modules_json_path) as f:
            modules = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse modules.json: {e}")
        return False

    # Download each module subdirectory - these contain small safetensor weights
    downloaded_any = False
    for module in modules:
        module_path = module.get("path", "")

        # Skip the root transformer (empty path) as it's handled by the main download
        if not module_path:
            continue

        logger.info(f"Downloading sentence-transformers module: {module_path}")

        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=f"{module_path}/**",
                token=token,
            )
            downloaded_any = True
        except Exception as e:
            logger.warning(f"Failed to download module {module_path}: {e}")

    return downloaded_any


def prepare_dataset_for_quantization(
    dataset_name,
    dataset_size,
    tokenizer_path,
    model_input_names,
    sequence_length=DEFAULT_MAX_SEQ_LENGTH,
    dataset_split=None,
    dataset_subset=None,
    dataset_columns=None,
):
    """
    Prepare HuggingFace dataset for RKNN quantization.

    Args:
        dataset_name: Name of the HuggingFace dataset
        dataset_size: Number of samples to use
        tokenizer_path: Path to tokenizer
        model_input_names: List of model input names to save (e.g., ["input_ids", "attention_mask"])
        sequence_length: Maximum sequence length
        dataset_split: Optional list of splits to use (e.g., ["train", "validation"])
        dataset_subset: Optional subset name for the dataset
        dataset_columns: Optional list of columns to use for text data

    Returns:
        Path to the dataset file for RKNN quantization
    """
    logger.info(f"Loading HuggingFace dataset: {dataset_name} (subset: {dataset_subset})")

    # Load all splits and combine them
    dataset_splits = []
    splits_to_try = dataset_split if dataset_split else ["train", "validation", "test"]

    for split in splits_to_try:
        with contextlib.suppress(Exception):
            if dataset_subset:
                split_data = load_dataset(dataset_name, dataset_subset, split=split)
            else:
                split_data = load_dataset(dataset_name, split=split)
            dataset_splits.append(split_data)

    if not dataset_splits:
        # Try loading without split to get default
        try:
            split_data = load_dataset(dataset_name, dataset_subset) if dataset_subset else load_dataset(dataset_name)

            if isinstance(split_data, dict):
                dataset_splits.extend(split_data.values())
            else:
                dataset_splits.append(split_data)
        except Exception as e:
            raise ValueError(f"Failed to load dataset {dataset_name}: {e}") from e

    combined_dataset = concatenate_datasets(dataset_splits)

    # Identify text columns
    target_columns = []
    if dataset_columns:
        # User specified columns
        for col in dataset_columns:
            if col in combined_dataset.column_names:
                target_columns.append(col)
            else:
                raise ValueError(
                    f"Specified column '{col}' not found in dataset. Available fields: {combined_dataset.column_names}"
                )
    else:
        # Auto-detect text field
        for field in AUTO_DETECT_TEXT_FIELDS:
            if field in combined_dataset.column_names:
                target_columns.append(field)
                break

    if not target_columns:
        target_columns = [combined_dataset.column_names[0]]
        logger.warning(f"Using field '{target_columns[0]}' for text data")

    # Sample
    total_available = len(combined_dataset)
    num_samples = min(dataset_size, total_available)

    if num_samples < total_available:
        indices = [int(i * (total_available - 1) / (num_samples - 1)) for i in range(num_samples)]
        dataset = combined_dataset.select(indices)
    else:
        dataset = combined_dataset

    # Tokenize
    logger.info("Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as err:
        logger.warning(f"Could not load tokenizer from {tokenizer_path}.")
        raise RuntimeError(f"Could not load tokenizer from {tokenizer_path}.") from err

    def tokenize_function(examples):
        args = [examples[col] for col in target_columns]
        return tokenizer(
            args[0],
            max_length=sequence_length,
            padding="max_length",
            truncation=True,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Create dataset file for RKNN
    temp_dir = tempfile.mkdtemp()
    dataset_file_path = os.path.join(temp_dir, "dataset.txt")
    logger.info(f"Saving tokenized data to numpy files for inputs: {model_input_names}")

    with open(dataset_file_path, "w") as temp_file:
        for idx in range(len(tokenized_dataset)):
            sample = tokenized_dataset[idx]

            # Only save inputs specified in model_input_names
            input_paths = []
            for input_name in model_input_names:
                if input_name not in sample:
                    logger.warning(f"Input '{input_name}' not found in tokenized sample {idx}, skipping")
                    continue

                # Convert to numpy array and save
                input_data = np.array([sample[input_name]], dtype=np.int64)
                input_path = os.path.join(temp_dir, f"sample_{idx}_{input_name}.npy")
                np.save(input_path, input_data)
                input_paths.append(input_path)

            # Skip this sample if we couldn't save any inputs
            if not input_paths:
                logger.warning(f"No valid inputs found for sample {idx}, skipping")
                continue

            # Write all input paths on one line, space-separated
            line = " ".join(input_paths)
            temp_file.write(f"{line}\n")

    return dataset_file_path, target_columns, splits_to_try


def load_model_config(model_name_or_path: str) -> PretrainedConfig | None:
    """
    Load model configuration from config.json or Hugging Face Hub.

    Args:
        model_name_or_path: Path to the ONNX model file or Hub ID

    Returns:
        PretrainedConfig object or None if not found.
    """
    try:
        # Use AutoConfig to load configuration (handles both local and Hub)
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        return config
    except Exception as e:
        logger.warning(f"Failed to load config for {model_name_or_path}. Attemping to load local config.json: {e}")

        # Fallback for local directory if AutoConfig fails
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                try:
                    # Try to load as generic PretrainedConfig from dict
                    with open(config_path) as f:
                        config_dict = json.load(f)
                    return PretrainedConfig.from_dict(config_dict)
                except Exception as inner_e:
                    logger.warning(f"Failed to load local config.json: {inner_e}")

        return None


def generate_rknn_output_path(
    config: RKNNConfig,
    model_dir: str,
    model_name: str,
    batch_size: int,
    sequence_length: int,
    is_user_specified_seq_len: bool = True,
) -> tuple[str, str]:
    """
    Generate the output path and RKNN JSON key for the exported model.

    Args:
        config: RKNN configuration object.
        model_dir: Directory where the model should be saved.
        model_name: Base name of the model.
        batch_size: Batch size used for export.
        sequence_length: Sequence length used for export.
        is_user_specified_seq_len: Whether the sequence length was explicitly specified by the user.

    Returns:
        A tuple containing:
            - output_path: Full path to the output RKNN file.
            - rknn_key: Key for the rknn config in config.json (relative path).
    """
    # Determine if batch_size and max_seq_length are non-default
    # Only include them in filename if they differ from defaults
    use_default_batch = batch_size == DEFAULT_BATCH_SIZE
    # Use default seq len if it matches constant OR if it was auto-detected (not specified by user)
    use_default_seq = (sequence_length == DEFAULT_MAX_SEQ_LENGTH) or (not is_user_specified_seq_len)

    use_default_params = use_default_batch and use_default_seq

    # Build parameter suffix for batch and sequence length
    param_suffix = "" if use_default_params else f"_b{batch_size}_s{sequence_length}"

    # Build optimization and quantization suffix
    if config.quantization.do_quantization:
        # Quantized model: rknn/model_b{batch}_s{seq}_o{opt}_w8a8.rknn
        sub_dir = "rknn"
        if config.optimization.optimization_level > 0:
            opt_quant_suffix = f"_o{config.optimization.optimization_level}_{config.quantization.quantized_dtype}"
        else:
            opt_quant_suffix = f"_{config.quantization.quantized_dtype}"
    elif config.optimization.optimization_level > 0:
        # Optimized but not quantized: rknn/model_b{batch}_s{seq}_o{opt}.rknn
        sub_dir = "rknn"
        opt_quant_suffix = f"_o{config.optimization.optimization_level}"
    else:
        # Unoptimized, unquantized: model_b{batch}_s{seq}.rknn or model.rknn (if defaults)
        sub_dir = ""
        opt_quant_suffix = ""

    # Combine all suffixes: param_suffix + opt_quant_suffix
    suffix = param_suffix + opt_quant_suffix

    if sub_dir:
        output_dir = os.path.join(model_dir, sub_dir)
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name}{suffix}.rknn"
        output_path = os.path.join(output_dir, filename)
        # Relative path for the rknn config in config.json
        rknn_key = f"{sub_dir}/{filename}"
    else:
        output_dir = model_dir
        # Ensure output directory exists if it's not empty
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name}{suffix}.rknn"
        output_path = os.path.join(output_dir, filename)
        rknn_key = filename

    return output_path, rknn_key


def update_model_config_with_rknn(
    rknn_config: RKNNConfig,
    model_dir: str,
    rknn_key: str,
    pretrained_config: PretrainedConfig | None = None,
) -> None:
    """
    Update the model's config.json with the RKNN configuration.

    This follows Hugging Face patterns by accepting an already-loaded PretrainedConfig
    object and updating it in-place before saving, rather than re-loading from disk.

    Args:
        rknn_config: RKNN configuration object.
        model_dir: Directory where the model is saved.
        rknn_key: Key to use for the RKNN config (relative path to RKNN model weights).
        pretrained_config: Optional pre-loaded PretrainedConfig object. If None, will load from disk.
    """
    config_path = os.path.join(model_dir, "config.json")
    rknn_export_dict = rknn_config.to_export_dict()

    try:
        if pretrained_config is not None:
            config = pretrained_config
        elif os.path.exists(config_path):
            try:
                config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            except Exception:
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = PretrainedConfig.from_dict(config_dict)
        else:
            config = PretrainedConfig()

        model_config_dict = config.to_dict()
        if "rknn" not in model_config_dict:
            model_config_dict["rknn"] = {}
        if not isinstance(model_config_dict["rknn"], dict):
            model_config_dict["rknn"] = {}
        model_config_dict["rknn"][rknn_key] = rknn_export_dict

        config.update(model_config_dict)
        config.save_pretrained(model_dir)
        logger.info(f"Updated configuration in {config_path}")

    except Exception as e:
        logger.warning(f"Failed to update config.json: {e}")


def get_onnx_input_names(model_path: str) -> list[str] | None:
    """
    Extract input names from an ONNX model.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        List of input names from the ONNX model, or None if extraction fails
    """
    try:
        # Load ONNX model without loading external data to be fast
        onnx_model = onnx.load(model_path, load_external_data=False)
        input_names = [input.name for input in onnx_model.graph.input]
        logger.info(f"Extracted input names from ONNX model: {input_names}")
        return input_names
    except Exception as e:
        logger.warning(f"Failed to extract input names from ONNX model: {e}")
        return None


def get_rk_model_class(task: str, pretrained_config: PretrainedConfig | None = None) -> str | None:
    """
    Get the RKModel class name for a given task.

    Args:
        task: The task name (e.g., "fill-mask", "sequence-classification")
        pretrained_config: The pretrained config object

    Returns:
        The RKModel class name (e.g., "RKModelForMaskedLM") or None if unknown.
    """
    model_class_name = TASK_TO_RK_MODEL_CLASS.get(task)
    if model_class_name:
        return model_class_name

    if pretrained_config and hasattr(pretrained_config, "architectures") and pretrained_config.architectures:
        for arch in pretrained_config.architectures:
            for rk_class in TASK_TO_RK_MODEL_CLASS.values():
                # Check if architecture ends with the suffix of the RK class
                # e.g. BertForTokenClassification ends with ForTokenClassification (from RKModelForTokenClassification)
                suffix = rk_class.replace("RKModel", "")
                if arch.endswith(suffix):
                    return rk_class

        if task == "auto":
            for arch in pretrained_config.architectures:
                if arch.endswith("Model"):
                    return "RKModelForFeatureExtraction"

    return None


def resolve_hub_repo_id(hub_model_id: str | None, hub_token: str | None = None) -> str | None:
    """
    Resolve the full repository ID with user namespace.

    If hub_model_id already contains a namespace (e.g., "user/model"), returns it as-is.
    If hub_model_id doesn't contain a namespace, auto-detects username via whoami() API
    and returns "{username}/{hub_model_id}".

    Args:
        hub_model_id: The repository ID, with or without namespace
        hub_token: Optional Hugging Face API token for authentication (required for auto-detection)

    Returns:
        The fully qualified repository ID with namespace, or None if hub_model_id is None

    Raises:
        ValueError: If unable to resolve the namespace (e.g., invalid token or network issues)

    Examples:
        >>> resolve_hub_repo_id("user/my-model")
        'user/my-model'

        >>> resolve_hub_repo_id("my-model", hub_token="hf_...")
        'auto-detected-user/my-model'
    """
    if hub_model_id is None:
        return None

    # If already contains namespace, return as-is
    if "/" in hub_model_id:
        return hub_model_id

    # Auto-detect username from token via whoami()
    try:
        from huggingface_hub import whoami

        user_info = whoami(token=hub_token)
        username = user_info.get("name")

        if not username:
            raise ValueError(
                "Unable to determine username from Hugging Face token. "
                f"Please either:\n"
                f"  1. Use a fully qualified repository ID (e.g., 'username/{hub_model_id}'), or\n"
                f"  2. Ensure your token is valid."
            )

        resolved_id = f"{username}/{hub_model_id}"
        logger.info(f"Auto-detected repository ID from token: {hub_model_id} -> {resolved_id}")
        return resolved_id

    except Exception as e:
        raise ValueError(
            f"Failed to resolve repository ID '{hub_model_id}': {e}. "
            "Please either:\n"
            f"  1. Use a fully qualified repository ID (e.g., 'username/{hub_model_id}'), or\n"
            "  2. Ensure your Hugging Face token is valid."
        ) from e


def clean_intermediate_onnx_files() -> None:
    """
    Clean up intermediate ONNX files created by RKNN toolkit during export.
    These files usually match the pattern check*.onnx in the current working directory.
    """
    cwd = os.getcwd()
    for filename in os.listdir(cwd):
        if filename.startswith("check") and filename.endswith(".onnx"):
            with contextlib.suppress(Exception):
                os.remove(os.path.join(cwd, filename))


def get_file_size_str(path: str) -> str:
    """
    Get a human-readable string representation of a file's size.

    Args:
        path: Path to the file.

    Returns:
        String representation of file size (e.g., "5.0 MB", "1.2 GB").
    """
    try:
        size_bytes = os.path.getsize(path)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    except Exception:
        return "Unknown"
