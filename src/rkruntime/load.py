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
import logging
import os
from typing import Any, cast

from transformers.configuration_utils import PretrainedConfig

from .utils.import_utils import is_sentence_transformers_available

logger = logging.getLogger(__name__)


def load_rknn_model(model_name_or_path: str, config: PretrainedConfig, task_name: str, **model_kwargs):
    """
    Load an RKNN model using the rkruntime library.

    Args:
        model_name_or_path (str): The model name on Hugging Face or the path to a local model directory.
        config (PretrainedConfig): The model configuration.
        task_name (str): The task name for the model (e.g. 'feature-extraction', 'fill-mask').
        model_kwargs (dict): Additional keyword arguments for the model loading.
    """
    try:
        from sentence_transformers.backend.utils import _save_pretrained_wrapper

        from rkruntime.modeling import (
            RKRTModelForFeatureExtraction,
            RKRTModelForMaskedLM,
            RKRTModelForSequenceClassification,
        )

        # Map task names to their corresponding model classes
        task_to_model_mapping = {
            "feature-extraction": RKRTModelForFeatureExtraction,
            "fill-mask": RKRTModelForMaskedLM,
            "sequence-classification": RKRTModelForSequenceClassification,
        }

        # Get the appropriate model class based on the task name
        if task_name not in task_to_model_mapping:
            supported_tasks = ", ".join(task_to_model_mapping.keys())
            raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {supported_tasks}")

        model_cls = task_to_model_mapping[task_name]
    except ImportError as e:
        raise Exception("Using the RKNN backend requires installing the rkruntime package.") from e

    # Load the model
    model = model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        **model_kwargs,
    )

    # Wrap the save_pretrained method to save the model in the correct subfolder
    # RKNN models are typically just a single .rknn file, but we follow the pattern
    model._save_pretrained = _save_pretrained_wrapper(model._save_pretrained, subfolder="rknn")

    return model


def patch_sentence_transformer():
    """
    Patch the SentenceTransformers library to support the RKNN backend.

    1. Patch Transformer._load_model to handle backend="rknn" and load RKNN models.
    2. Patch Transformer.__init__ to set max_seq_length from rknn.json if available.
    3. Patch Transformer.tokenize to ensure padding="max_length" for RKNN models.
    """
    if not is_sentence_transformers_available():
        logger.warning("SentenceTransformers is not available. Skipping RKNN patch.")
        return

    from sentence_transformers.models import Transformer

    # Patch Transformer._load_model
    original_load_model = Transformer._load_model

    def _load_model_patched(self, model_name_or_path, config, cache_dir, backend, is_peft_model, **model_kwargs):
        if backend == "rknn":
            # Check for rknn.json to get runtime parameters
            rknn_config_path = os.path.join(model_name_or_path, "rknn.json")

            if os.path.exists(rknn_config_path):
                with open(rknn_config_path) as f:
                    rknn_config = json.load(f)
                    self._rknn_config = rknn_config

                    # Determine which model file to use
                    file_name = model_kwargs.get("file_name")
                    if file_name is None:
                        if "model.rknn" in rknn_config:
                            file_name = "model.rknn"
                        elif "model.onnx" in rknn_config:
                            file_name = "model.onnx"

                    if file_name and file_name in rknn_config:
                        model_config = rknn_config[file_name]
                        # Update config with max_seq_length if present
                        if "max_seq_length" in model_config:
                            config.max_position_embeddings = model_config.pop("max_seq_length")

                        # Pass remaining config to model_kwargs
                        model_kwargs.update(model_config)

                        # Ensure file_name is passed if we found it
                        if "file_name" not in model_kwargs:
                            model_kwargs["file_name"] = file_name

            # Load the RKNN model
            self.auto_model = load_rknn_model(
                model_name_or_path,
                config=config,
                task_name="feature-extraction",
                **model_kwargs,
            )
        else:
            original_load_model(self, model_name_or_path, config, cache_dir, backend, is_peft_model, **model_kwargs)

    Transformer._load_model = _load_model_patched

    # Patch Transformer.__init__ to apply max_seq_length after tokenizer is loaded
    original_init = Transformer.__init__

    def __init__patched(
        self,
        model_name_or_path: str,
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str | None = None,
        backend: str = "rknn",
    ) -> None:
        original_init(
            self,
            model_name_or_path,
            max_seq_length,
            model_args,
            tokenizer_args,
            config_args,
            cache_dir,
            do_lower_case,
            tokenizer_name_or_path,
            backend,
        )

        if not hasattr(self, "_rknn_config"):
            return

        # Apply RKNN max_seq_length override after tokenizer is loaded and other initializations
        self.max_seq_length = self.auto_model.config.max_position_embeddings
        # Configure tokenizer to pad to max_length for static RKNN models
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.padding = "max_length"
            self.tokenizer.pad_to_max_length = True
            self.tokenizer.model_max_length = self.max_seq_length

    Transformer.__init__ = __init__patched

    # Patch Transformer.tokenize to force max_length padding for RKNN backend
    original_tokenize = Transformer.tokenize

    def tokenize_patched(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, Any]:
        if not hasattr(self, "_rknn_config"):
            return original_tokenize(self, texts, padding)

        # For RKNN backend, always use max_length padding and return numpy arrays
        padding = "max_length"
        # Call original but override return_tensors
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            texts = cast(list[dict], texts)
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            texts = cast(list[tuple[str, str]], texts)
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",  # Return torch tensors for compatibility with Pooling
                max_length=self.max_seq_length,
            )
        )
        return output

    Transformer.tokenize = tokenize_patched

    logger.info("Patched SentenceTransformer to support RKNN backend.")
