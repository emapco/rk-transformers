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

import importlib.util
import logging
import os
import re
import shutil
import sys
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
    AutoModelForSequenceClassification,
    PretrainedConfig,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from transformers.utils.doc import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.utils.generic import ModelOutput
from transformers.utils.hub import cached_file, is_offline_mode

from .constants import (
    RKNN_FILE_PATTERN,
    RKNN_WEIGHTS_NAME,
    CoreMaskType,
    PlatformType,
)
from .utils.env_utils import get_edge_host_platform

logger = logging.getLogger(__name__)

if importlib.util.find_spec("rknnlite") is not None:
    from rknnlite.api import RKNNLite  # pyright: ignore[reportMissingImports]
elif importlib.util.find_spec("rknn") is not None:
    # Fallback to RKNN if RKNNLite is not available. RKNN is a superset of RKNNLite.
    from rknn.api import RKNN as RKNNLite  # pyright: ignore[reportMissingImports]
else:
    logger.error("RKNN Toolkit Lite is not installed. Please install it via pip:")
    logger.error("  pip install rknn-toolkit-lite2==2.3.2")
    sys.exit(-1)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
RKNN_MODEL_END_DOCSTRING = r"""
    This model inherits from [`~rkruntime.modeling.RKRTModel`], check its documentation for the generic methods the
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


class RKRTModel(
    ModelHubMixin,
    library_name="rkruntime",
    tags=["rknn"],
):
    """Base class for RKNN-backed text models integrated with the Hugging Face Hub."""

    model_type: str = "rknn_model"
    auto_model_class = AutoModel

    def __init__(
        self,
        *,
        config: PretrainedConfig | None = None,
        model_path: str | Path,
        platform: PlatformType | None = None,
        core_mask: CoreMaskType = "auto",
    ) -> None:
        if config is None:
            raise ValueError("A Hugging Face config is required to build an RKRT model.")

        self.config = config
        self.model_path = Path(model_path)
        self.model_filename = self.model_path.name
        self.platform = platform or get_edge_host_platform()
        self.core_mask = core_mask

        self.input_names = ["input_ids", "attention_mask"]
        if getattr(config, "type_vocab_size", 1) > 1:
            self.input_names.append("token_type_ids")

        self.rknn = self._load_rknn_model()

        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    def forward(self, *args: Any, **kwargs: Any) -> ModelOutput | tuple[torch.Tensor | np.ndarray] | None:
        """Define the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        """Make RKRTModel callable to work with Transformer/SentenceTransformer"""
        return self.forward(*args, **kwargs)

    def _release(self) -> None:
        """Release RKNNLite resources following RKNN-toolkit2 best practices."""
        if getattr(self, "rknn", None) is not None:
            assert self.rknn is not None
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
        if hasattr(self.rknn, "list_support_target_platform") and callable(self.rknn.list_support_target_platform):  # type: ignore
            return self.rknn.list_support_target_platform(self.model_path.as_posix())  # type: ignore
        return None

    def _load_rknn_model(self) -> RKNNLite:
        """Load the RKNN graph and initialize the runtime."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        rknn = RKNNLite(verbose=False)
        logger.debug("Loading RKNN model from %s", self.model_path)
        ret = rknn.load_rknn(self.model_path.as_posix())
        if ret != 0:
            raise RuntimeError("Failed to load RKNN model")

        # Check model-platform compatibility
        supported_platforms = None
        if hasattr(rknn, "list_support_target_platform") and callable(rknn.list_support_target_platform):  # type: ignore
            supported_platforms: dict[str, Any] | None = rknn.list_support_target_platform(self.model_path.as_posix())  # type: ignore

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
        if self.platform in {"rk3588", "rk3576", "filled_target_platform", "support_target_platform"}:
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
            ret = rknn.init_runtime(core_mask=npu_core)
        else:
            ret = rknn.init_runtime()

        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

        return rknn

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

    def _prepare_text_inputs(
        self,
        input_ids: torch.Tensor | np.ndarray | None,
        attention_mask: torch.Tensor | np.ndarray | None,
        token_type_ids: torch.Tensor | np.ndarray | None,
    ) -> tuple[bool, dict[str, torch.Tensor | np.ndarray | None]]:
        tensors = [input_ids, attention_mask, token_type_ids]
        first_tensor = next((tensor for tensor in tensors if tensor is not None), None)
        if first_tensor is None:
            raise ValueError("At least one tensor input must be provided to the RKRT model.")

        use_torch = isinstance(first_tensor, torch.Tensor)
        if input_ids is None:
            raise ValueError("`input_ids` is required for RKRT text inference.")
        if attention_mask is None:
            attention_mask = self._ones_like(input_ids, use_torch)
        if "token_type_ids" in self.input_names and token_type_ids is None:
            token_type_ids = self._zeros_like(input_ids, use_torch)

        return use_torch, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def _run_text_model(
        self,
        use_torch: bool,
        model_inputs: dict[str, torch.Tensor | np.ndarray | None],
        expected_outputs: Sequence[str],
    ) -> dict[str, torch.Tensor | np.ndarray]:
        ordered_inputs: list[np.ndarray] = []
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            tensor = model_inputs.get(name)
            if tensor is None:
                continue
            ordered_inputs.append(self._tensor_to_numpy(tensor, np.dtype(np.int64)))

        if self.rknn is None:
            raise RuntimeError("RKNN runtime has been released and can no longer run inference.")

        outputs = self.rknn.inference(inputs=ordered_inputs)
        if outputs is None:
            raise RuntimeError("RKNN inference returned None.")
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
            logger.warning(
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
        resume_download: bool | None = False,
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
            resume_download=resume_download,
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
        resume_download: bool | None = False,
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
                resume_download=resume_download,
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
    ) -> "RKRTModel":
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
                resume_download=resume_download,
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
                resume_download=resume_download,
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
            resume_download=resume_download,
            proxies=proxies,
        )

        return cls(
            config=resolved_config,
            model_path=model_path,
            platform=platform,
            core_mask=core_mask,
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
    ) -> "RKRTModel":
        return super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            platform=platform,
            core_mask=core_mask,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
            resume_download=resume_download,
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
    >>> from rkruntime.modeling import {model_class}
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
class RKRTModelForFeatureExtraction(RKRTModel):
    """RKNN model for feature extraction tasks."""

    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKRTModelForFeatureExtraction",
            checkpoint="chem-mrl/rknn-text-encoder",
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
    ) -> BaseModelOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["last_hidden_state"])
        last_hidden_state = outputs["last_hidden_state"]
        if not return_dict:
            return (last_hidden_state,)
        return BaseModelOutput(last_hidden_state=last_hidden_state)  # type: ignore[arg-type]


MASKED_LM_EXAMPLE = r"""
    Example of masked language modeling:

    ```python
    >>> from transformers import {processor_class}
    >>> from rkruntime.modeling import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="np")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 8, 28996]
    ```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKRTModelForMaskedLM(RKRTModel):
    """RKNN model for masked language modeling tasks."""

    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKRTModelForMaskedLM",
            checkpoint="chem-mrl/rknn-text-mlm",
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
    ) -> MaskedLMOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["logits"])
        logits = outputs["logits"]
        if not return_dict:
            return (logits,)
        return MaskedLMOutput(logits=logits)  # type: ignore[arg-type]


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from rkruntime.modeling import {model_class}
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
class RKRTModelForSequenceClassification(RKRTModel):
    """RKNN model for sequence classification/regression tasks."""

    auto_model_class = AutoModelForSequenceClassification

    @add_start_docstrings_to_model_forward(
        TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="RKRTModelForSequenceClassification",
            checkpoint="chem-mrl/rknn-text-classifier",
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
    ) -> SequenceClassifierOutput | tuple[torch.Tensor | np.ndarray]:
        self._warn_on_unhandled_inputs(kwargs)
        use_torch, model_inputs = self._prepare_text_inputs(input_ids, attention_mask, token_type_ids)
        outputs = self._run_text_model(use_torch, model_inputs, ["logits"])
        logits = outputs["logits"]
        if not return_dict:
            return (logits,)
        return SequenceClassifierOutput(logits=logits)  # type: ignore[arg-type]
