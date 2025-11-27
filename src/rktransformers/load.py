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
from typing import Any, Literal

import numpy as np
from sentence_transformers.util.file_io import is_sentence_transformer_model
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .utils.import_utils import is_sentence_transformers_available

logger = logging.get_logger(__name__)


def load_rknn_model(model_name_or_path: str, config: PretrainedConfig, task_name: str, **model_kwargs):
    """
    Load an RKNN model using the rktransformers library.

    Args:
        model_name_or_path (str): The model name on Hugging Face or the path to a local model directory.
        config (PretrainedConfig): The model configuration.
        task_name (str): The task name for the model (e.g. 'feature-extraction', 'fill-mask').
        model_kwargs (dict): Additional keyword arguments for the model loading.
            - file_name (str, optional): Specific RKNN file to load (e.g. "model_quantized.rknn").
    """
    try:
        from sentence_transformers.backend.utils import _save_pretrained_wrapper

        from rktransformers.modeling import (
            RKRTModelForFeatureExtraction,
            RKRTModelForMaskedLM,
            RKRTModelForSequenceClassification,
        )

        task_to_model_mapping = {
            "feature-extraction": RKRTModelForFeatureExtraction,
            "fill-mask": RKRTModelForMaskedLM,
            "sequence-classification": RKRTModelForSequenceClassification,
        }

        if task_name not in task_to_model_mapping:
            supported_tasks = ", ".join(task_to_model_mapping.keys())
            raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {supported_tasks}")

        model_cls = task_to_model_mapping[task_name]
    except ImportError as e:
        raise Exception("Using the RKNN backend requires installing the sentence-transformers package.") from e

    model = model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        **model_kwargs,
    )

    # Wrap the save_pretrained method to save the model in the correct subfolder
    # RKNN models are typically just a single .rknn file, but we follow the pattern
    model._save_pretrained = _save_pretrained_wrapper(model._save_pretrained, subfolder="rknn")

    return model


def _load_rknn_config(model_name_or_path: str, config: PretrainedConfig, **model_kwargs):
    """
    Helper function to load RKNN configuration from rknn.json.

    Args:
        model_name_or_path (str): Model name or path
        config (PretrainedConfig): Model configuration
        **model_kwargs: Additional model loading arguments

    Returns:
        tuple: (rknn_config dict, updated config, updated model_kwargs dict)
    """
    from sentence_transformers.util.file_io import load_file_path

    rknn_config = {}
    cache_dir = model_kwargs.get("cache_dir")
    token = model_kwargs.get("token")
    revision = model_kwargs.get("revision")
    local_files_only = model_kwargs.get("local_files_only", False)

    rknn_json_path = load_file_path(
        model_name_or_path,
        "rknn.json",
        token=token,
        cache_folder=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )

    if rknn_json_path:
        try:
            with open(rknn_json_path) as f:
                rknn_config = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load rknn.json: {e}")

    # Determine which model file to use
    file_name = model_kwargs.get("file_name")
    if file_name is None:
        # Preference: model.rknn (unoptimized) -> first available in rknn/
        if "model.rknn" in rknn_config:
            file_name = "model.rknn"
        elif rknn_config:
            file_name = next(iter(rknn_config))  # First key in rknn_config

    if file_name:
        if file_name in rknn_config:
            rknn_config = rknn_config[file_name]
        if "max_seq_length" in rknn_config:
            config.max_position_embeddings = rknn_config["max_seq_length"]
        if "file_name" not in model_kwargs:
            model_kwargs["file_name"] = file_name

    return rknn_config, config, model_kwargs


def patch_cross_encoder():
    """
    Patch the CrossEncoder class to support the RKNN backend.

    1. Patch CrossEncoder._load_model to handle backend="rknn" and load RKNN models.
    2. Patch CrossEncoder.predict to ensure padding="max_length" for RKNN models.
    """
    if not is_sentence_transformers_available():
        raise ImportError(
            "sentence-transformers is not available. Please install it via pip: pip install sentence-transformers"
        )

    from sentence_transformers import CrossEncoder

    # Patch CrossEncoder._load_model
    original_load_model = CrossEncoder._load_model

    def _load_model_patched(
        self,
        model_name_or_path,
        config,
        backend,
        **model_kwargs,
    ):
        """
        Patched version of CrossEncoder._load_model to support RKNN backend.
        """
        if backend == "rknn":
            rknn_config, config, model_kwargs = _load_rknn_config(model_name_or_path, config, **model_kwargs)
            self._rknn_config = rknn_config

            self.model = load_rknn_model(
                model_name_or_path,
                config=config,
                task_name="sequence-classification",
                **model_kwargs,
            )
        else:
            original_load_model(
                self,
                model_name_or_path,
                config,
                backend,
                **model_kwargs,
            )

    CrossEncoder._load_model = _load_model_patched

    # Patch CrossEncoder.__init__ to apply tokenizer kwargs after initialization
    original_init = CrossEncoder.__init__

    def __init__patched(
        self,
        model_name_or_path: str,
        num_labels: int | None = None,
        max_length: int | None = None,
        activation_fn: Any | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
        model_card_data: Any | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        """
        Patched version of CrossEncoder.__init__ to apply RKNN-specific tokenizer configurations.

        This method overrides the original __init__ to configure tokenizer_kwargs BEFORE
        the tokenizer is initialized. For RKNN models, it pre-loads the config and rknn.json
        to determine the correct max_length and padding settings, then passes these via
        tokenizer_kwargs so the tokenizer is created correctly from the start.

        Args:
            model_name_or_path (str): The model name on Hugging Face or the path to a local model directory.
            num_labels (int, optional): Number of labels for the classifier.
            max_length (int, optional): Maximum sequence length.
            activation_fn (callable, optional): Activation function.
            device (str, optional): Device to use for computation.
            cache_folder (str, optional): Directory to cache the model files.
            trust_remote_code (bool): Whether to trust remote code.
            revision (str, optional): Model revision to use.
            local_files_only (bool): Whether to only use local files.
            token (bool or str, optional): Hugging Face authentication token.
            model_kwargs (dict, optional): Additional model loading arguments.
            tokenizer_kwargs (dict, optional): Additional tokenizer loading arguments.
            config_kwargs (dict, optional): Additional config loading arguments.
            model_card_data (optional): Model card data.
            backend (str): The backend to use for model loading (default: "torch").

        Pre-processing:
            - For RKNN backend, loads config and rknn.json to get max_position_embeddings
            - Modifies tokenizer_kwargs to include padding="max_length" and model_max_length
        """
        original_init(
            self,
            model_name_or_path,
            num_labels,
            max_length,
            activation_fn,
            device,
            cache_folder,
            trust_remote_code,
            revision,
            local_files_only,
            token,
            model_kwargs,
            tokenizer_kwargs,
            config_kwargs,
            model_card_data,
            backend,
        )

        if not hasattr(self, "_rknn_config"):
            return

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.padding = "max_length"
            self.tokenizer.pad_to_max_length = True
            self.tokenizer.model_max_length = self.config.max_position_embeddings

    CrossEncoder.__init__ = __init__patched

    # Patch CrossEncoder.predict
    original_predict = CrossEncoder.predict

    def predict_patched(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=None,
        activation_fn=None,
        apply_softmax=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
    ):
        """
        Patched version of CrossEncoder.predict to enforce max_length padding for RKNN models.
        """
        if not hasattr(self, "_rknn_config"):
            return original_predict(
                self,
                sentences,
                batch_size,
                show_progress_bar,
                activation_fn,
                apply_softmax,
                convert_to_numpy,
                convert_to_tensor,
            )

        import torch
        from tqdm.autonotebook import trange

        input_was_singular = False
        if sentences and isinstance(sentences, (list, tuple)) and isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_singular = True

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG  # type: ignore
            )

        if activation_fn is not None:
            self.set_activation_fn(activation_fn, set_default=False)

        # Use RKNN batch size if defined
        if self._rknn_config and "batch_size" in self._rknn_config:
            rknn_batch_size = self._rknn_config["batch_size"]
            if batch_size != rknn_batch_size:
                logger.warning_once(  # type: ignore
                    f"Overriding batch_size {batch_size} with RKNN model's configured batch_size {rknn_batch_size}"
                )
                batch_size = rknn_batch_size

        pred_scores = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            batch = sentences[start_index : start_index + batch_size]
            # Force max_length padding for RKNN
            features = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            model_predictions = self.model(**features, return_dict=True)
            logits = self.activation_fn(model_predictions.logits)

            if apply_softmax and logits.ndim > 1:
                logits = torch.nn.functional.softmax(logits, dim=1)
            pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores) if len(pred_scores) else torch.tensor([], device=self.model.device)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().float().numpy() for score in pred_scores])

        if input_was_singular:
            pred_scores = pred_scores[0]

        return pred_scores

    CrossEncoder.predict = predict_patched
    logger.info("Patched CrossEncoder to support RKNN backend.")


def patch_sentence_transformer():
    """
    Patch the SentenceTransformers library to support the RKNN backend.

    1. Patch Transformer._load_model to handle backend="rknn" and load RKNN models.
    2. Patch Transformer.__init__ to set max_seq_length from rknn.json if available.
    3. Patch Transformer.tokenize to ensure padding="max_length" for RKNN models.
    4. Patch SentenceTransformer.encode to enforce RKNN model's batch size.
    """
    if not is_sentence_transformers_available():
        raise ImportError(
            "sentence-transformers is not available. Please install it via pip: pip install sentence-transformers"
        )

    from sentence_transformers.models import Transformer

    # Patch Transformer._load_model
    original_load_model = Transformer._load_model

    def _load_model_patched(
        self,
        model_name_or_path,
        config,
        cache_dir,
        backend,
        is_peft_model,
        **model_kwargs,
    ):
        """
        Patched version of Transformer._load_model to support RKNN backend.

        This method overrides the original _load_model to handle RKNN model loading when backend="rknn".
        It loads the rknn.json configuration file to extract runtime parameters such as max_seq_length
        and batch_size, and initializes the appropriate RKNN model via load_rknn_model.

        Args:
            model_name_or_path (str): The model name on Hugging Face or the path to a local model directory.
            config (PretrainedConfig): The model configuration.
            cache_dir (str, optional): Directory to cache the model files.
            backend (str): The backend to use for model loading (e.g., "rknn", "pytorch", "onnx").
            is_peft_model (bool): Whether the model is a PEFT (Parameter-Efficient Fine-Tuning) model.
            **model_kwargs: Additional keyword arguments for model loading, such as:
                - file_name (str, optional): Specific RKNN file to load.

        Sets:
            self._rknn_config (dict): The RKNN configuration loaded from rknn.json.
            self._is_sentence_transformer (bool): Whether the model is a sentence-transformers model.
            self.auto_model: The loaded RKNN model instance.
        """
        if backend == "rknn":
            rknn_config, config, model_kwargs = _load_rknn_config(
                model_name_or_path, config, **{**model_kwargs, "cache_dir": cache_dir}
            )
            self._rknn_config = rknn_config

            self._is_sentence_transformer = is_sentence_transformer_model(model_name_or_path, cache_folder=cache_dir)

            self.auto_model = load_rknn_model(
                model_name_or_path,
                config=config,
                task_name="feature-extraction",
                **model_kwargs,
            )
        else:
            original_load_model(
                self,
                model_name_or_path,
                config,
                cache_dir,
                backend,
                is_peft_model,
                **model_kwargs,
            )

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
        """
        Patched version of Transformer.__init__ to apply RKNN-specific configurations.

        This method overrides the original __init__ to apply max_seq_length and padding settings
        from the RKNN configuration after the model and tokenizer have been loaded. For RKNN models,
        it ensures the tokenizer uses max_length padding and sets the model_max_length to match
        the model's max_position_embeddings.

        Args:
            model_name_or_path (str): The model name on Hugging Face or the path to a local model directory.
            max_seq_length (int, optional): Maximum sequence length for the model. Can be overridden by
                RKNN configuration.
            model_args (dict, optional): Additional arguments for model loading.
            tokenizer_args (dict, optional): Additional arguments for tokenizer loading.
            config_args (dict, optional): Additional arguments for config loading.
            cache_dir (str, optional): Directory to cache the model files.
            do_lower_case (bool): Whether to lowercase the input text.
            tokenizer_name_or_path (str, optional): The tokenizer name or path to use.
            backend (str): The backend to use for model loading (default: "rknn").

        Post-processing:
            - Sets self.max_seq_length from the model's max_position_embeddings if available.
            - Configures the tokenizer to use max_length padding if RKNN backend is detected.
        """
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

        if self.auto_model and hasattr(self.auto_model, "config"):
            self.max_seq_length = self.auto_model.config.max_position_embeddings  # Set from RKNN config in _load_model
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.padding = "max_length"
            self.tokenizer.pad_to_max_length = True
            self.tokenizer.model_max_length = self.max_seq_length

    Transformer.__init__ = __init__patched

    # Patch Transformer.tokenize to force max_length padding for RKNN backend
    original_tokenize = Transformer.tokenize

    def tokenize_patched(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        padding: str | bool = True,
    ) -> dict[str, Any]:
        """
        Patched version of Transformer.tokenize to enforce max_length padding for RKNN models.

        This method overrides the original tokenize to ensure that RKNN models always receive
        inputs padded to max_length, which is required for proper RKNN inference with fixed
        input shapes.

        Args:
            texts (list): A list of texts to tokenize. Can be:
                - list[str]: A list of sentences.
                - list[dict]: A list of dictionaries with text keys.
                - list[tuple[str, str]]: A list of sentence pairs.
            padding (str or bool): Padding strategy. For RKNN models, this is always overridden
                to "max_length" regardless of the provided value.

        Returns:
            dict[str, Any]: A dictionary containing tokenized inputs with keys like 'input_ids',
                'attention_mask', etc.

        RKNN-specific behavior:
            - If _rknn_config is present, padding is forced to "max_length".
            - If _rknn_config is not present, falls back to the original tokenize behavior.
        """
        if not hasattr(self, "_rknn_config"):
            return original_tokenize(self, texts, padding)

        # For RKNN backend, always use max_length padding and return numpy arrays
        return original_tokenize(self, texts, padding="max_length")

    Transformer.tokenize = tokenize_patched

    # Patch SentenceTransformer.encode
    from sentence_transformers import SentenceTransformer

    original_encode = SentenceTransformer.encode

    def encode_patched(self, sentences: str | list[str], **kwargs):
        """
        Patched version of SentenceTransformer.encode to support RKNN backend batch size requirements.

        This function enforces the batch size specified in the RKNN model configuration when encoding sentences.
        If the RKNN backend is detected (via a module attribute), the batch size is overridden to match the
        RKNN model's configured batch size.

        Parameters:
            self: The SentenceTransformer instance.
            sentences (str or list of str): The sentence or list of sentences to encode.
            **kwargs: Additional keyword arguments passed to the original encode method. The 'batch_size'
                argument will be overridden if the RKNN backend is active.

        Returns:
            np.ndarray: The encoded sentence embeddings as a numpy array.

        RKNN-specific behavior:
            - If an RKNN config with a 'batch_size' is present, the batch size is enforced.
            - For non-sentence-transformers models, manual batching is performed to ensure correct tensor shapes.
            - A warning is logged once if the user-supplied batch size differs from the RKNN model's batch size.
        """
        batch_size = kwargs.pop("batch_size", 32)  # Default batch size from SentenceTransformers
        # Check for RKNN config in modules
        rknn_config = None
        for module in self:
            if hasattr(module, "_rknn_config"):
                rknn_config = module._rknn_config
                break

        if not rknn_config or "batch_size" not in rknn_config:
            kwargs["batch_size"] = batch_size
            return original_encode(self, sentences, **kwargs)

        rknn_batch_size = rknn_config["batch_size"]
        if batch_size != rknn_batch_size:
            logger.warning_once(  # type: ignore
                f"Overriding batch_size {batch_size} with RKNN model's configured batch_size {rknn_batch_size}"
            )

        # sentence-transformers models work as usual with adjusted batch size
        is_sentence_transformer = False
        for module in self:
            if hasattr(module, "_is_sentence_transformer"):
                is_sentence_transformer = module._is_sentence_transformer
                break
        if is_sentence_transformer:
            kwargs["batch_size"] = rknn_batch_size
            return original_encode(self, sentences, **kwargs)

        # For non-sentence-transformers models (e.g. bert-base-uncased), we need to batch manually.
        # Otherwise, the Pooling layer causes a runtime error due to misshapen tensors.
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        for i in range(0, len(sentences), rknn_batch_size):
            batch = sentences[i : i + rknn_batch_size]
            # Call original_encode without batch_size to let it process the batch as-is
            batch_embeddings = original_encode(self, batch, **kwargs)
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)

    SentenceTransformer.encode = encode_patched  # type: ignore

    logger.info("Patched SentenceTransformer to support RKNN backend.")
