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
import os
import re
import shutil
import sys
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi, ModelHubMixin
from huggingface_hub.constants import HF_HUB_CACHE
from optimum.utils.file_utils import find_files_matching_pattern
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.utils import logging
from transformers.utils.doc import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.utils.generic import ModelOutput
from transformers.utils.hub import cached_file, is_offline_mode

from .configuration import RKNNConfig
from .constants import (
    RKNN_FILE_PATTERN,
    RKNN_WEIGHTS_NAME,
    CoreMaskType,
    PlatformType,
)
from .utils.env_utils import get_edge_host_platform
from .utils.import_utils import (
    is_rknn_toolkit_available,
    is_rknn_toolkit_lite_available,
)
from .utils.logging_utils import suppress_output

logger = logging.get_logger(__name__)

if is_rknn_toolkit_lite_available():
    from rknnlite.api import RKNNLite  # pyright: ignore[reportMissingImports]
elif is_rknn_toolkit_available():
    # Fallback to RKNN if RKNNLite is not available. RKNN shares similar functionality as RKNNLite.
    from rknn.api import RKNN as RKNNLite  # pyright: ignore[reportMissingImports]
else:
    logger.error("RKNN Toolkit Lite is not installed. Please install it via pip:")
    logger.error("  pip install rknn-toolkit-lite2==2.3.2")
    sys.exit(-1)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
RKNN_MODEL_END_DOCSTRING = r"""
    This model inherits from [`~rktransformers.modeling.RKModel`], check its documentation for the generic methods the
    library implements for all its model (such as downloading or saving).
"""

FROM_PRETRAINED_START_DOCSTRING = r"""
    Instantiate a pretrained model from a pre-trained model configuration.

    Args:
        model_id (`Union[str, Path]`):
            Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing a model saved using [`~OptimizedModel.save_pretrained`],
                    e.g., `./my_model_directory/`.
        export (`bool`, defaults to `False`):
            Defines whether the provided `model_id` needs to be exported to the targeted format.
        force_download (`bool`, defaults to `True`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        token (`Optional[Union[bool,str]]`, defaults to `None`):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
        cache_dir (`Optional[str]`, defaults to `None`):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        subfolder (`str`, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
            specify the folder name here.
        config (`Optional[transformers.PretrainedConfig]`, defaults to `None`):
            The model configuration.
        local_files_only (`Optional[bool]`, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        trust_remote_code (`bool`, defaults to `False`):
            Whether or not to allow for custom code defined on the Hub in their own modeling. This option should only be set
            to `True` for repositories you trust and in which you have read the code, as it will execute code present on
            the Hub on your local machine.
        revision (`Optional[str]`, defaults to `None`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
"""  # noqa: E501

TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Indices of input sequence tokens in the vocabulary.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        token_type_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
            Segment token indices to indicate first and second portions of the inputs.
"""


# workaround to enable compatibility between rk-transformers models and transformers pipelines
class PreTrainedModel(ABC):  # noqa: B024
    pass


class RKNNRuntime:
    """Handles RKNN model loading and runtime initialization."""

    def __init__(
        self,
        model_path: str | Path,
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
        rknn_config: RKNNConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.platform = platform or get_edge_host_platform()
        self.core_mask = core_mask
        self.rknn_config = rknn_config
        self.rknn = self._load_rknn_model()

    def _release(self) -> None:
        """Release RKNNLite resources following RKNN-toolkit2 best practices."""
        if getattr(self, "rknn", None) is not None:
            assert self.rknn is not None
            with contextlib.suppress(Exception), suppress_output():
                self.rknn.release()
            self.rknn = None

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self._release()

    def list_model_compatible_platform(self) -> dict[str, Any] | None:
        """
        List supported platforms for the loaded model.
        Returns a dictionary of supported platforms or None if the check fails.
        """
        if self.rknn is None:
            return None
        with suppress_output():
            if hasattr(self.rknn, "list_support_target_platform") and callable(self.rknn.list_support_target_platform):  # type: ignore
                return self.rknn.list_support_target_platform(self.model_path.as_posix())  # type: ignore
        return None

    def _load_rknn_model(self) -> RKNNLite:
        """Load the RKNN graph and initialize the runtime."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        # Suppress RKNNLite C-level logging
        with suppress_output():
            rknn = RKNNLite(verbose=False)
            logger.debug("Loading RKNN model from %s", self.model_path)
            ret = rknn.load_rknn(self.model_path.as_posix())
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model")

        # Check model-platform compatibility
        supported_platforms = None
        if hasattr(rknn, "list_support_target_platform") and callable(rknn.list_support_target_platform):  # type: ignore
            with suppress_output():
                supported_platforms: dict[str, Any] | None = rknn.list_support_target_platform(  # type: ignore
                    self.model_path.as_posix()
                )

        if supported_platforms is not None:
            # The return value is an OrderedDict with a key 'support_target_platform' and 'filled_target_platform'
            # containing a list of supported platforms.
            # Note: Only the 'support_target_platform' is relevant for compatibility check.
            support_platforms = [p.lower() for p in supported_platforms.get("support_target_platform", [])]
            if self.platform is not None and len(support_platforms) > 0 and self.platform not in support_platforms:
                raise RuntimeError(
                    f"The model is not compatible with the current platform '{self.platform}'. "
                    f"Supported platforms: {support_platforms}"
                )

        logger.debug(
            "Initializing RKNN runtime (platform=%s, core_mask=%s)",
            self.platform,
            self.core_mask,
        )
        if self.platform in {"rk3588", "rk3576"}:
            core_mask_map = {
                "auto": RKNNLite.NPU_CORE_AUTO,
                "0": RKNNLite.NPU_CORE_0,
                "1": RKNNLite.NPU_CORE_1,
                "2": RKNNLite.NPU_CORE_2,
                "0_1": RKNNLite.NPU_CORE_0_1,
                "0_1_2": RKNNLite.NPU_CORE_0_1_2,
                "all": RKNNLite.NPU_CORE_ALL,
            }
            npu_core = core_mask_map.get(self.core_mask, RKNNLite.NPU_CORE_AUTO)
            with suppress_output():
                ret = rknn.init_runtime(core_mask=npu_core)
                if ret != 0:
                    ret = rknn.init_runtime(target=self.platform, core_mask=npu_core)
        else:
            with suppress_output():
                ret = rknn.init_runtime()

        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

        return rknn


class RKModel(
    RKNNRuntime,
    PreTrainedModel,
    ModelHubMixin,
    library_name="rk-transformers",
    tags=["rknn", "rockchip", "npu"],
):
    """Base class for RKNN-backed text models integrated with the Hugging Face Hub."""

    model_type: str = "rknn_model"
    auto_model_class = AutoModel

    def __init__(
        self,
        *,
        model_id: str | None = None,
        config: PretrainedConfig | None = None,
        model_path: str | Path,
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
        rknn_config: RKNNConfig | None = None,
        max_seq_length: int = 512,
        batch_size: int = 1,
    ) -> None:
        if config is None:
            raise ValueError("A Hugging Face config is required to build an RKModel.")

        super().__init__(model_path=model_path, platform=platform, core_mask=core_mask, rknn_config=rknn_config)
        self.model_id = model_id
        self.config = config

        # Set defaults for input_names, batch_size, and max_seq_length
        self.input_names = ["input_ids", "attention_mask"]
        if getattr(config, "type_vocab_size", 1) > 1:
            self.input_names.append("token_type_ids")
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        if self.rknn_config:
            if hasattr(self.rknn_config, "model_input_names") and self.rknn_config.model_input_names:
                self.input_names = self.rknn_config.model_input_names

            if hasattr(self.rknn_config, "max_seq_length") and self.rknn_config.max_seq_length is not None:
                self.max_seq_length = self.rknn_config.max_seq_length

            if hasattr(self.rknn_config, "batch_size"):
                self.batch_size = self.rknn_config.batch_size

        self.pad_token_id = 0
        self.pad_token_type_id = 0
        self.pad_attention_mask = 0  # Huggingface transformers uses 0 for padding attention mask
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is None:
                self.pad_token_id = self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, "pad_token_type_id") and self.tokenizer.pad_token_type_id is not None:
                self.pad_token_type_id = self.tokenizer.pad_token_type_id
        except Exception:
            logger.warning("Failed to load tokenizer. Using default padding IDs (0).")

        # From optimum.onnxruntime.modeling.ORTModel
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def __call__(self, *args, **kwargs):
        """Make RKModel callable to work with Transformers/SentenceTransformers"""
        return self.forward(*args, **kwargs)

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> (
        ModelOutput
        | tuple[torch.Tensor | np.ndarray]
        | tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]
        | None
    ):
        """Define the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is stored."""
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "RKModel":
        """No-op for RKModel. For compatibility with Hugging Face Transformers Pipelines."""
        return self

    def _tensor_to_numpy(self, tensor: torch.Tensor | np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
        if tensor is None:
            raise ValueError("Input tensor is required for RKNN inference.")
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            array = tensor
        else:
            array = np.asarray(tensor)
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        return array

    def _torch_if_needed(self, use_torch: bool, array: np.ndarray) -> torch.Tensor | np.ndarray:
        if use_torch:
            contiguous = np.ascontiguousarray(array)
            return torch.from_numpy(contiguous)
        return array

    def _ones_like(self, reference: torch.Tensor | np.ndarray, use_torch: bool) -> torch.Tensor | np.ndarray:
        if use_torch:
            if not isinstance(reference, torch.Tensor):  # pragma: no cover - defensive
                reference = torch.from_numpy(np.asarray(reference))
            return torch.ones_like(reference)
        return np.ones_like(np.asarray(reference))

    def _zeros_like(self, reference: torch.Tensor | np.ndarray, use_torch: bool) -> torch.Tensor | np.ndarray:
        if use_torch:
            if not isinstance(reference, torch.Tensor):  # pragma: no cover - defensive
                reference = torch.from_numpy(np.asarray(reference))
            return torch.zeros_like(reference)
        return np.zeros_like(np.asarray(reference))

    def _pad_to_model_input_dimensions(
        self,
        tensor: torch.Tensor | np.ndarray,
        padding_id: int,
        use_torch: bool,
        target_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor | np.ndarray:
        """Pad tensor to match model's expected input dimensions.

        Handles arbitrary tensor ranks (2D for standard tasks, 3D for multiple-choice, etc.)
        by padding each dimension independently to match the target shape.

        Args:
            tensor: Input tensor to pad (e.g., shape: [batch, seq_len] or [batch, num_choices, seq_len])
            padding_id: Value to use for padding
            use_torch: Whether to use PyTorch or NumPy for padding
            target_shape: Target shape for the tensor. If None, defaults to 2D padding behavior
                         using self.batch_size and self.max_seq_length.

        Returns:
            Padded tensor with shape matching target_shape
        """
        # Default to 2D padding for backward compatibility
        if target_shape is None:
            target_shape = (getattr(self, "batch_size", tensor.shape[0]), self.max_seq_length)

        if len(target_shape) != len(tensor.shape):
            raise ValueError(
                f"Target shape rank ({len(target_shape)}) must match tensor rank ({len(tensor.shape)}). "
                f"Got target_shape={target_shape}, tensor.shape={tensor.shape}"
            )

        needs_padding = any(current < target for current, target in zip(tensor.shape, target_shape, strict=True))
        if not needs_padding:
            return tensor

        # Calculate padding for each dimension
        # Padding goes at the "end" of each dimension (right/bottom)
        if use_torch:
            # PyTorch pad format: (dim_n_before, dim_n_after, ..., dim_0_before, dim_0_after)
            # We only pad at the end, so all "before" values are 0
            pad_values: list[int] = []
            for current_dim, target_dim in reversed(list(zip(tensor.shape, target_shape, strict=True))):
                pad_values.extend([0, max(0, target_dim - current_dim)])  # (before, after)
            tensor = torch.nn.functional.pad(tensor, tuple(pad_values), value=padding_id)
        else:
            # NumPy pad format: ((dim_0_before, dim_0_after), (dim_1_before, dim_1_after), ...)
            pad_width = [
                (0, max(0, target - current)) for current, target in zip(tensor.shape, target_shape, strict=True)
            ]
            tensor = np.pad(tensor, pad_width, constant_values=padding_id)

        return tensor

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None,
        token_type_ids: torch.Tensor | np.ndarray | None,
        input_shape: tuple[int, ...] | None = None,
    ) -> tuple[bool, dict[str, torch.Tensor | np.ndarray | None], tuple[int, ...]]:
        """Prepare text inputs for RKNN inference with padding.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)
            input_shape: Expected input shape (e.g., [batch_size, seq_len] for 2D,
                        [batch_size, num_choices, seq_len] for 3D). If None, defaults
                        to 2D shape using self.batch_size and self.max_seq_length.

        Returns:
            Tuple of (use_torch, model_inputs, original_shape)
        """
        if input_ids is None:
            raise ValueError("`input_ids` is required for RKModel text inference.")

        use_torch = isinstance(input_ids, torch.Tensor)
        original_shape = tuple(input_ids.shape)

        # Calculate target shape
        if input_shape is None:
            # Default 2D behavior: [batch_size, seq_len]
            target_shape = (getattr(self, "batch_size", original_shape[0]), self.max_seq_length)
        else:
            # Use provided input_shape, filling in dimensions as needed
            target_shape = input_shape

        # Pad inputs to target shape
        input_ids = self._pad_to_model_input_dimensions(
            input_ids, padding_id=self.pad_token_id, use_torch=use_torch, target_shape=target_shape
        )

        if attention_mask is None:
            attention_mask = self._ones_like(input_ids, use_torch)
        attention_mask = self._pad_to_model_input_dimensions(
            attention_mask, padding_id=self.pad_attention_mask, use_torch=use_torch, target_shape=target_shape
        )

        if "token_type_ids" in self.input_names:
            if token_type_ids is None:
                token_type_ids = self._zeros_like(input_ids, use_torch)  # Use padded input_ids as reference
            else:
                token_type_ids = self._pad_to_model_input_dimensions(
                    token_type_ids, padding_id=self.pad_token_type_id, use_torch=use_torch, target_shape=target_shape
                )

        return (
            use_torch,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
            original_shape,
        )

    def _run_text_model(
        self,
        use_torch: bool,
        model_inputs: dict[str, torch.Tensor | np.ndarray | None],
        expected_outputs: Sequence[str],
    ) -> dict[str, torch.Tensor | np.ndarray]:
        ordered_inputs: list[np.ndarray] = []
        for name in self.input_names:
            tensor = model_inputs.get(name)
            if tensor is None:
                continue
            ordered_inputs.append(self._tensor_to_numpy(tensor, np.dtype(np.int16)))

        if self.rknn is None:
            raise RuntimeError("RKNN runtime has been released and can no longer run inference.")

        # Suppress RKNN inference logs
        with suppress_output():
            if is_rknn_toolkit_lite_available():
                # data_type: int8 | uint8 | int16 | float16 | float32 - limitation with rknn MM API/Hardware
                # This an issue for models with embeddings since they require int64 inputs.
                outputs = self.rknn.inference(inputs=ordered_inputs, data_type="int16")
            else:
                outputs = self.rknn.inference(inputs=ordered_inputs)
        if outputs is None:
            input_summaries = [f"shape={arr.shape}, dtype={arr.dtype}" for arr in ordered_inputs]
            raise RuntimeError(f"RKNN inference returned None - input summary: {input_summaries}")
        if len(outputs) < len(expected_outputs):
            logger.error(
                "RKNN inference output mismatch: expected %d outputs (%s), got %d outputs",
                len(expected_outputs),
                expected_outputs,
                len(outputs),
            )
            raise RuntimeError("RKNN inference did not return the expected number of outputs.")

        prepared: dict[str, torch.Tensor | np.ndarray] = {}
        for idx, name in enumerate(expected_outputs):
            prepared[name] = self._torch_if_needed(use_torch, np.asarray(outputs[idx]))
        return prepared

    def _warn_on_unhandled_inputs(self, kwargs: dict[str, Any]) -> None:
        if kwargs:
            logger.warning_once(  # type: ignore - transformers logger util
                "%s received unsupported arguments: %s",
                self.__class__.__name__,
                ", ".join(kwargs.keys()),
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        target = save_directory / RKNN_WEIGHTS_NAME
        shutil.copyfile(self.model_path, target)

    @staticmethod
    def _cached_file(
        path_or_repo_id: str | Path,
        filename: str,
        subfolder: str = "",
        revision: str | None = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        token: bool | str | None = None,
        cache_dir: str | Path = HF_HUB_CACHE,
        proxies: dict | None = None,
    ) -> Path:
        cached_path = cached_file(
            path_or_repo_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            proxies=proxies,
        )
        if cached_path is None:
            raise FileNotFoundError(f"Unable to cache RKNN artifact `{filename}` from {path_or_repo_id}.")
        return Path(cached_path)

    @staticmethod
    def _infer_file_path(
        pattern: str,
        candidate_files: list[Path],
        standard_file_name: str,
        target_file_name: str | None = None,
    ) -> Path:
        if target_file_name is not None:
            specific = [file for file in candidate_files if file.name == target_file_name]
            if not specific:
                raise FileNotFoundError(
                    f"Could not find any RKNN files with target file name {target_file_name}. "
                    f"Candidates: {candidate_files}."
                )
            if len(specific) > 1:
                logger.warning(
                    "Found multiple RKNN files named %s, using %s.",
                    target_file_name,
                    specific[0].name,
                )
            return specific[0]

        standard = [file for file in candidate_files if file.name == standard_file_name]
        if len(standard) == 1:
            return standard[0]
        if len(standard) > 1:
            logger.warning(
                "Found multiple RKNN files named %s, using %s.",
                standard_file_name,
                standard[0].name,
            )
            return standard[0]

        pattern_files = [path for path in candidate_files if re.search(pattern, str(path))]
        if not pattern_files:
            raise FileNotFoundError(
                f"Could not find an RKNN artifact matching pattern {pattern}. Candidates: {candidate_files}."
            )
        if len(pattern_files) > 1:
            logger.warning(
                "Found multiple RKNN files matching pattern %s, using %s.",
                pattern,
                pattern_files[0].name,
            )
        return pattern_files[0]

    @staticmethod
    def _list_repo_rknn_files(
        model_id: str,
        *,
        revision: str | None,
        token: str | bool | None,
        subfolder: str,
    ) -> list[Path]:
        """Enumerate RKNN files from a remote Hugging Face repository."""
        api = HfApi(token=token if isinstance(token, str) else None)
        try:
            repo_files = api.list_repo_files(model_id, revision=revision, repo_type="model")
        except Exception as exc:  # pragma: no cover - network errors
            logger.debug("Failed to list repo files for %s: %s", model_id, exc)
            return []

        filtered: list[Path] = []
        for file_path in repo_files:
            if not re.search(RKNN_FILE_PATTERN, file_path):
                continue
            path_obj = Path(file_path)
            if subfolder:
                try:
                    path_obj.relative_to(subfolder)
                except ValueError:
                    continue
            filtered.append(path_obj)
        return filtered

    @classmethod
    def _resolve_config(
        cls,
        model_id: str,
        config: Any | None,
        *,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        trust_remote_code: bool,
        proxies: dict | None = None,
    ) -> PretrainedConfig:
        if isinstance(config, PretrainedConfig):
            return config
        if isinstance(config, dict):
            return PretrainedConfig.from_dict(config)
        try:
            return AutoConfig.from_pretrained(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                trust_remote_code=trust_remote_code,
                proxies=proxies,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to a generic config for %s because AutoConfig loading failed: %s",
                model_id,
                exc,
            )
            fallback_model_type = getattr(getattr(cls.auto_model_class, "config_class", None), "model_type", None)
            return PretrainedConfig(
                model_type=fallback_model_type or cls.model_type,
                name_or_path=model_id,
            )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        config: PretrainedConfig | None,
        # rknn options
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
        rknn_config: RKNNConfig | None = None,
        # hub options
        subfolder: str = "",
        revision: str | None = None,
        force_download: bool = False,
        resume_download: bool | None = False,
        proxies: dict | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None,
        token: str | bool | None,
        # file options
        file_name: str | None = None,
        **model_kwargs: Any,
    ) -> "RKModel":
        cache_dir = cache_dir or HF_HUB_CACHE

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        if os.path.isfile(model_id):
            model_path = Path(model_id)
        elif file_name is not None:
            model_path = cls._cached_file(
                model_id,
                filename=file_name,
                subfolder=subfolder,
                local_files_only=local_files_only,
                force_download=force_download,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                proxies=proxies,
            )
        else:
            candidate_files = find_files_matching_pattern(
                model_id,
                pattern=RKNN_FILE_PATTERN,
                glob_pattern="**/*.rknn",
                subfolder=subfolder,
                token=token,
                revision=revision,
            )
            if not candidate_files:
                candidate_files = cls._list_repo_rknn_files(
                    model_id,
                    revision=revision,
                    token=token,
                    subfolder=subfolder,
                )
            if not candidate_files:
                raise FileNotFoundError(f"Could not find any RKNN model file in {model_id}.")
            if Path(model_id).is_dir():
                candidate_files = [path.relative_to(model_id) for path in candidate_files]

            resolved_file = cls._infer_file_path(
                RKNN_FILE_PATTERN,
                candidate_files,
                standard_file_name=RKNN_WEIGHTS_NAME,
                target_file_name=file_name,
            )
            subfolder_to_use = resolved_file.parent.as_posix()
            if subfolder_to_use == ".":
                subfolder_to_use = ""

            model_path = cls._cached_file(
                model_id,
                filename=resolved_file.name,
                subfolder=subfolder_to_use,
                local_files_only=local_files_only,
                force_download=force_download,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                proxies=proxies,
            )

        resolved_config = cls._resolve_config(
            model_id,
            config,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            trust_remote_code=trust_remote_code,
            proxies=proxies,
        )

        # Try to get rknn config from the resolved config object
        rknn_configs = {}
        if hasattr(resolved_config, "rknn"):
            rknn_configs = resolved_config.rknn
        elif isinstance(resolved_config, dict) and "rknn" in resolved_config:
            rknn_configs = resolved_config["rknn"]

        model_rknn_config = None
        if rknn_configs:
            # Match model filename to keys in rknn config (e.g. "rknn/model.rknn")
            filename = model_path.name
            for key, conf in rknn_configs.items():
                if key.endswith(filename):
                    try:
                        model_rknn_config = RKNNConfig.from_dict(conf)
                        logger.info(f"Loaded RKNN config for {filename}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to parse RKNN config for {key}: {e}")

        if not model_rknn_config:
            logger.warning("RKNN config not found in config.json. Use default batch_size=1 and max_seq_length=512.")

        return cls(
            model_id=model_id,
            config=resolved_config,
            model_path=model_path,
            platform=platform,
            core_mask=core_mask,
            rknn_config=model_rknn_config,
            **model_kwargs,
        )

    @classmethod
    @add_start_docstrings(FROM_PRETRAINED_START_DOCSTRING)
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: PretrainedConfig | None = None,
        # rknn options
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
        # hub options
        subfolder: str = "",
        revision: str | None = None,
        force_download: bool = False,
        resume_download: bool | None = False,
        proxies: dict | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | Path | None = None,
        # file options
        file_name: str | None = None,
        **model_kwargs: Any,
    ) -> "RKModel":
        return super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            platform=platform,
            core_mask=core_mask,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
            proxies=proxies,
            token=token,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            file_name=file_name,
            **model_kwargs,
        )


FEATURE_EXTRACTION_EXAMPLE = r"""
    Example of feature extraction:

    ```python
    >>> from transformers import {processor_class}
    >>> from rktransformers.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 12, 384]
    ```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForFeatureExtraction(RKModel):
    """RKNN model for feature extraction tasks."""

    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKModelForFeatureExtraction",
            checkpoint="rk-transformers/all-MiniLM-L6-v2",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> BaseModelOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs, original_shape = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["last_hidden_state"])
        last_hidden_state = outputs["last_hidden_state"][: original_shape[0]]
        if not return_dict:
            return (last_hidden_state,)
        return BaseModelOutput(last_hidden_state=last_hidden_state)  # type: ignore[arg-type]


MASKED_LM_EXAMPLE = r"""
    Example of masked language modeling:

    ```python
    >>> from transformers import {processor_class}
    >>> from rktransformers.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 512, 30522]
    ```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForMaskedLM(RKModel):
    """RKNN model for masked language modeling tasks."""

    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKModelForMaskedLM",
            checkpoint="rk-transformers/bert-base-uncased",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> MaskedLMOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs, original_shape = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["logits"])
        logits = outputs["logits"][: original_shape[0]]
        if not return_dict:
            return (logits,)
        return MaskedLMOutput(logits=logits)  # type: ignore[arg-type]


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from rktransformers.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 2]
    ```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForSequenceClassification(RKModel):
    """RKNN model for sequence classification/regression tasks."""

    auto_model_class = AutoModelForSequenceClassification

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKModelForSequenceClassification",
            checkpoint="rk-transformers/distilbert-base-uncased-finetuned-sst-2-english",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs, original_shape = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["logits"])
        logits = outputs["logits"][: original_shape[0]]
        if not return_dict:
            return (logits,)
        return SequenceClassifierOutput(logits=logits)  # type: ignore[arg-type]


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering:

    ```python
    >>> from transformers import {processor_class}
    >>> from rktransformers.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="np")

    >>> outputs = model(**inputs)
    >>> start_logits = outputs.start_logits
    >>> end_logits = outputs.end_logits
    >>> list(start_logits.shape)
    [1, 512]
    >>> list(end_logits.shape)
    [1, 512]
    ```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForQuestionAnswering(RKModel):
    """RKNN Model with a QuestionAnsweringModelOutput for extractive question-answering tasks like SQuAD."""

    auto_model_class = AutoModelForQuestionAnswering

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKModelForQuestionAnswering",
            checkpoint="rk-transformers/distilbert-base-cased-distilled-squad",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> QuestionAnsweringModelOutput | tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs, original_shape = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["start_logits", "end_logits"])
        start_logits = outputs["start_logits"][: original_shape[0]]
        end_logits = outputs["end_logits"][: original_shape[0]]
        if not return_dict:
            return (start_logits, end_logits)
        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)  # type: ignore[arg-type]


TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example of token classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from rktransformers.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 512, 9]
    ```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForTokenClassification(RKModel):
    """RKNN Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks."""

    auto_model_class = AutoModelForTokenClassification

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKModelForTokenClassification",
            checkpoint="rk-transformers/bert-base-NER",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> TokenClassifierOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs, original_shape = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["logits"])
        logits = outputs["logits"][: original_shape[0]]
        if not return_dict:
            return (logits,)
        return TokenClassifierOutput(logits=logits)  # type: ignore[arg-type]


MULTIPLE_CHOICE_EXAMPLE = r"""
    Example of multiple choice:

    ```python
    >>> from transformers import {processor_class}
    >>> from rktransformers.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza is served in slices."
    >>> choice0 = "It is eaten with a fork and knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> choice2 = "It is blended into a smoothie."
    >>> choice3 = "It is folded into a taco."
    >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;))

    >>> encoding = tokenizer([prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3], return_tensors="np", padding=True)
    >>> inputs = {{k: np.expand_dims(v, 0) for k, v in encoding.items()}}

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 4]
    ```
"""  # noqa: E501


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForMultipleChoice(RKModel):
    """RKNN Model with a multiple choice classification head
    on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks."""

    auto_model_class = AutoModelForMultipleChoice

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
        + MULTIPLE_CHOICE_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKModelForMultipleChoice",
            checkpoint="rk-transformers/bert-base-uncased_SWAG",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray | None = None,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> MultipleChoiceModelOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        if input_ids is None:
            raise ValueError("`input_ids` is required for RKModel text inference.")

        # Multiple-choice inputs are 3D: [batch_size, num_choices, seq_len]
        if len(input_ids.shape) != 3:
            raise ValueError(
                f"Multiple-choice inputs must be 3D [batch_size, num_choices, seq_len]. Got shape: {input_ids.shape}"
            )

        batch_size, num_choices, seq_len = input_ids.shape

        # Get num_choices from config if available, otherwise use input shape
        expected_num_choices = None
        if self.rknn_config and hasattr(self.rknn_config, "task_kwargs") and self.rknn_config.task_kwargs:
            expected_num_choices = self.rknn_config.task_kwargs.get("num_choices")
            if expected_num_choices != num_choices:
                raise ValueError(
                    f"Number of choices in config ({expected_num_choices}) does not match input shape ({num_choices})"
                )
        else:
            self.num_choices = num_choices
            logger.warning_once("RKNN config not found in config.json. Using input_ids shape to infer num_choices.")  # type: ignore

        target_shape = (self.batch_size, num_choices, self.max_seq_length)
        use_torch, model_inputs, original_shape = self._prepare_text_inputs(
            input_ids, attention_mask, token_type_ids, input_shape=target_shape
        )
        outputs = self._run_text_model(use_torch, model_inputs, ["logits"])
        logits = outputs["logits"]

        # Reshape logits if needed: RKNN may return [batch_size * num_choices] or [batch_size, num_choices]
        if logits.ndim == 1:
            # Flatten case: [batch_size * num_choices] -> [batch_size, num_choices]
            logits = logits.reshape(original_shape[0], original_shape[1])
        elif logits.shape != (original_shape[0], original_shape[1]):
            # Trim padding if needed
            logits = logits[: original_shape[0], : original_shape[1]]

        if not return_dict:
            return (logits,)
        return MultipleChoiceModelOutput(logits=logits)  # type: ignore[arg-type]
