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

import textwrap
from collections.abc import Callable
from typing import Any

from transformers.utils.doc import get_docstring_indentation_level


# Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L45
#
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
#
# Modified to work with Sphinx and removed <Tip> section as it isn't relevant for RK models.
def add_start_docstrings_to_model_forward(*docstr) -> Callable[..., Any]:
    def docstring_decorator(fn):
        class_name = f":class:`~{fn.__qualname__.split('.')[0]}`"
        intro = rf"""The {class_name} forward method, overrides the :meth:`~rktransformers.modeling.RKModel.__call__` special method.

        """  # noqa: E501

        correct_indentation = get_docstring_indentation_level(fn)
        current_doc = fn.__doc__ if fn.__doc__ is not None else ""

        docs = docstr
        # Always reindent the added docstrings to the correct indentation level
        docs = [textwrap.indent(textwrap.dedent(doc), " " * correct_indentation) for doc in docstr]
        intro = textwrap.indent(textwrap.dedent(intro), " " * correct_indentation)

        docstring = "".join(docs) + current_doc
        fn.__doc__ = intro + docstring
        return fn

    return docstring_decorator


TOKENIZER_FOR_DOC = "AutoTokenizer"
RKNN_MODEL_END_DOCSTRING = r"""
This model inherits from :class:`~rktransformers.modeling.RKModel`, check its documentation for the generic methods the
library implements for all its model (such as downloading or saving).
"""  # noqa: E501

FROM_PRETRAINED_START_DOCSTRING = r"""
Instantiate a pretrained model from a pre-trained model configuration.

Args:
    model_id (`Union[str, Path]`):
        Can be either:
            - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                user or organization name, like `rk-transformers/bert-base-uncased`.
            - A path to a *directory* containing a model previously exported using :py:func:`~rktransformers.exporters.rknn.convert.export_rknn`,
                e.g., `./my_model_directory/`.
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
    input_ids (`Union[torch.Tensor, np.ndarray]` of shape `({0})`):
        Indices of input sequence tokens in the vocabulary.
        `What are input IDs? <https://huggingface.co/docs/transformers/glossary#input-ids>`_
    attention_mask (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    token_type_ids (`Union[torch.Tensor, np.ndarray, None]` of shape `({0})`, defaults to `None`):
        Segment token indices to indicate first and second portions of the inputs.
    return_dict (`bool`, optional, defaults to `None`):
        Whether or not to return a subclass of :class:`~transformers.modeling_outputs.ModelOutput` instead of a tuple.
        Tensors will be np.ndarrays or torch.Tensors depending on the original input_ids type.
"""  # noqa: E501
