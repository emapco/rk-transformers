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

from typing import Any, Literal

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.util.file_io import is_sentence_transformer_model
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

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
            RKModelForFeatureExtraction,
            RKModelForMaskedLM,
            RKModelForSequenceClassification,
        )

        task_to_model_mapping = {
            "feature-extraction": RKModelForFeatureExtraction,
            "fill-mask": RKModelForMaskedLM,
            "sequence-classification": RKModelForSequenceClassification,
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


def _load_rknn_config(
    config: PretrainedConfig, model_kwargs: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Helper function to load RKNN configuration from config.json.

    Args:
        config (PretrainedConfig): Model configuration
        **model_kwargs: Additional model loading arguments

    Returns:
        tuple: (rknn_config dict, updated config, updated model_kwargs dict)
    """
    if not hasattr(config, "rknn"):
        return None, model_kwargs

    root_rknn_config = config.rknn

    file_name = model_kwargs.get("file_name")
    if file_name is None:
        # Preference: model.rknn (unoptimized) -> first available in rknn/
        if "model.rknn" in root_rknn_config:
            file_name = "model.rknn"
        elif root_rknn_config:
            file_name = next(iter(root_rknn_config))

    rknn_config = root_rknn_config.get(file_name)
    if "file_name" not in model_kwargs:
        model_kwargs["file_name"] = file_name

    return rknn_config, model_kwargs


class RKCrossEncoder(CrossEncoder):
    """
    RKNN-compatible CrossEncoder implementation.

    This class extends SentenceTransformers' CrossEncoder to support the RKNN backend.
    It overrides model loading and prediction methods to handle RKNN-specific requirements
    such as fixed input shapes and quantization.
    """

    def __init__(
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
        backend: Literal["torch", "onnx", "openvino", "rknn"] = "rknn",
    ) -> None:
        """
        Initialize the RKCrossEncoder.

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
            backend (str): The backend to use for model loading (default: "rknn").
        """
        super().__init__(
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
            backend,  # type: ignore
        )
        # Post-init configuration for RKNN
        if (
            hasattr(self, "_rknn_config")
            and self._rknn_config is not None
            and "max_seq_length" in self._rknn_config
            and hasattr(self, "tokenizer")
            and self.tokenizer is not None
        ):
            self.tokenizer.model_max_length = self._rknn_config["max_seq_length"]

    def _load_model(
        self,
        model_name_or_path: str,
        config: PretrainedConfig,
        backend: str,
        **model_kwargs,
    ) -> None:
        """
        Load the model using the specified backend.

        Overrides CrossEncoder._load_model to support 'rknn' backend.
        """
        if backend == "rknn":
            self._rknn_config, model_kwargs = _load_rknn_config(config, **model_kwargs)
            self.model = load_rknn_model(
                model_name_or_path,
                config=config,
                task_name="sequence-classification",
                **model_kwargs,
            )
        else:
            super()._load_model(model_name_or_path, config, backend, **model_kwargs)

    def predict(
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
        Performs predictions with the CrossEncoder on the given sentence pairs.

        Overrides CrossEncoder.predict to enforce max_length padding for RKNN models.
        """
        if not hasattr(self, "_rknn_config"):
            return super().predict(
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
        if self._rknn_config is not None and "batch_size" in self._rknn_config:
            rknn_batch_size = self._rknn_config["batch_size"]
            if batch_size != rknn_batch_size:
                logger.warning_once(  # type: ignore
                    f"Overriding batch_size {batch_size} with RKNN model's configured batch_size {rknn_batch_size}."
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


class RKTransformer(Transformer):
    """
    RKNN-compatible Transformer implementation.

    This class extends sentence_transformers.models.Transformer to support the RKNN backend.
    It overrides model loading, initialization, and tokenization methods to handle RKNN-specific
    requirements such as fixed input shapes and quantization.
    """

    def __init__(
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
        Initialize the RKTransformer.

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
        """
        super().__init__(
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
        # Post-init configuration for RKNN
        if hasattr(self, "_rknn_config") and self._rknn_config is not None and "max_seq_length" in self._rknn_config:
            self.max_seq_length = self._rknn_config["max_seq_length"]

    def _load_model(
        self,
        model_name_or_path: str,
        config,
        cache_dir,
        backend: str,
        is_peft_model: bool,
        **model_kwargs,
    ) -> None:
        """
        Load the model using the specified backend.

        Overrides Transformer._load_model to support 'rknn' backend.

        Args:
            model_name_or_path (str): The model name on Hugging Face or the path to a local model directory.
            config (PretrainedConfig): The model configuration.
            cache_dir (str, optional): Directory to cache the model files.
            backend (str): The backend to use for model loading (e.g., "rknn", "pytorch", "onnx").
            is_peft_model (bool): Whether the model is a PEFT (Parameter-Efficient Fine-Tuning) model.
            **model_kwargs: Additional keyword arguments for model loading, such as:
                - file_name (str, optional): Specific RKNN file to load.
        """
        if backend == "rknn":
            self._rknn_config, model_kwargs = _load_rknn_config(config, **{**model_kwargs, "cache_dir": cache_dir})
            self._is_sentence_transformer = is_sentence_transformer_model(model_name_or_path, cache_folder=cache_dir)
            self.auto_model = load_rknn_model(
                model_name_or_path,
                config=config,
                task_name="feature-extraction",
                **model_kwargs,
            )
        else:
            super()._load_model(
                model_name_or_path,
                config,
                cache_dir,
                backend,
                is_peft_model,
                **model_kwargs,
            )

    def tokenize(
        self,
        texts: list[str] | list[dict] | list[tuple[str, str]],
        padding: str | bool = True,
    ) -> dict[str, Any]:
        """
        Tokenize texts with RKNN-specific padding requirements.

        Overrides Transformer.tokenize to enforce max_length padding for RKNN models.

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
        """
        if hasattr(self, "_rknn_config"):
            padding = "max_length"
        return super().tokenize(texts, padding)


class RKSentenceTransformer(SentenceTransformer):
    """
    RKNN-compatible SentenceTransformer implementation.

    This class extends SentenceTransformers' SentenceTransformer to support the RKNN backend.
    It overrides the encode method to handle RKNN-specific batch size requirements.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules=None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name=None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token=None,
        use_auth_token=None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data=None,
        backend: str = "rknn",
    ) -> None:
        """
        Initialize the RKSentenceTransformer.

        Args:
            model_name_or_path (str, optional): Model name or path to load.
            backend (str): The backend to use for model loading (default: "rknn").
            All other arguments are passed to SentenceTransformer.__init__.
        """
        super().__init__(
            model_name_or_path,
            modules,
            device,
            prompts,
            default_prompt_name,
            similarity_fn_name,
            cache_folder,
            trust_remote_code,
            revision,
            local_files_only,
            token,
            use_auth_token,
            truncate_dim,
            model_kwargs,
            tokenizer_kwargs,
            config_kwargs,
            model_card_data,
            backend,
        )

    def _load_auto_model(
        self,
        model_name_or_path: str,
        token,
        cache_folder,
        revision,
        trust_remote_code,
        local_files_only,
        model_kwargs,
        tokenizer_kwargs,
        config_kwargs,
        has_modules,
    ):
        """
        Override _load_auto_model to use RKTransformer for RKNN backend.

        This ensures that when creating models from plain transformers (no modules.json),
        we use RKTransformer which supports the RKNN backend.
        """
        logger.warning(
            f"No sentence-transformers model found with name {model_name_or_path}. "
            "Creating a new one with mean pooling."
        )

        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = shared_kwargs if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        tokenizer_kwargs = shared_kwargs if tokenizer_kwargs is None else {**shared_kwargs, **tokenizer_kwargs}
        config_kwargs = shared_kwargs if config_kwargs is None else {**shared_kwargs, **config_kwargs}

        # Use RKTransformer instead of Transformer
        transformer_model = RKTransformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
            backend=self.backend,
        )
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "mean")
        return [transformer_model, pooling_model]

    def encode(self, sentences: str | list[str], **kwargs):
        """
        Encode sentences with RKNN backend batch size requirements.

        Overrides SentenceTransformer.encode to enforce the batch size specified in the RKNN
        model configuration when encoding sentences.

        Parameters:
            sentences (str or list of str): The sentence or list of sentences to encode.
            **kwargs: Additional keyword arguments passed to the original encode method.
                The 'batch_size' argument will be overridden if the RKNN backend is active.

        Returns:
            np.ndarray: The encoded sentence embeddings as a numpy array.
        """
        batch_size = kwargs.pop("batch_size", 32)

        # Check for RKNN config in modules
        rknn_config = None
        for module in self:
            if hasattr(module, "_rknn_config"):
                rknn_config = module._rknn_config
                break

        if rknn_config is None or "batch_size" not in rknn_config:
            kwargs["batch_size"] = batch_size
            return super().encode(sentences, **kwargs)

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
            return super().encode(sentences, **kwargs)

        # For non-sentence-transformers models, batch manually
        if isinstance(sentences, str):
            return super().encode(sentences, **kwargs)

        all_embeddings = []
        for i in range(0, len(sentences), rknn_batch_size):
            batch = sentences[i : i + rknn_batch_size]
            batch_embeddings = super().encode(batch, **kwargs)
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)
