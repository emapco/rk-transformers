# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
# Modified output classes to reflect the returned type and values from RKNN models.

from dataclasses import dataclass

import numpy as np
import torch
from transformers.cache_utils import Cache
from transformers.utils.generic import ModelOutput


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs.

    Args:
        last_hidden_state (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape
            :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
    """

    last_hidden_state: torch.Tensor | np.ndarray


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape
            :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token
            before SoftMax).
    """

    logits: torch.Tensor | np.ndarray


@dataclass
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    Args:
        logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape :obj:`(batch_size, num_choices)`):
            Classification scores (before SoftMax). ``num_choices`` is the second dimension of the input tensors.
    """

    logits: torch.Tensor | np.ndarray


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        start_logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
    """

    start_logits: torch.Tensor | np.ndarray
    end_logits: torch.Tensor | np.ndarray


@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape
            :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if ``config.num_labels==1``) scores (before SoftMax).
    """

    logits: torch.Tensor | np.ndarray


@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape
            :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
    """

    logits: torch.Tensor | np.ndarray


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        logits (:class:`torch.Tensor` or :class:`numpy.ndarray` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:class:`~transformers.cache_utils.Cache` or :class:`numpy.ndarray`, *optional*):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used
            to speed up sequential decoding. See the `kv cache guide <https://huggingface.co/docs/transformers/en/kv_cache>`_
            for more details.
    """  # noqa: E501

    logits: torch.Tensor | np.ndarray
    past_key_values: Cache | np.ndarray | None = None
