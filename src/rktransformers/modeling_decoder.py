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
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging
from transformers.utils.doc import add_end_docstrings

from .configuration import RKNNConfig
from .constants import (
    CoreMaskType,
    PlatformType,
)
from .modeling import RKModel
from .modeling_outputs import CausalLMOutputWithPast
from .utils.docs import (
    RKNN_MODEL_END_DOCSTRING,
    TOKENIZER_FOR_DOC,
    add_start_docstrings_to_model_forward,
)
from .utils.logging_utils import suppress_output

logger = logging.get_logger(__name__)


CAUSALLM_RKNN_MODEL_DOCSTRING = r"""
Args:
    input_ids (`torch.LongTensor`):
        Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
    attention_mask (`torch.LongTensor`):
        Mask to avoid performing attention on padding token indices, of shape
        `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
    past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
        Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
        The tuple is of length `config.n_layer` with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""


TEXT_GENERATION_EXAMPLE = r"""
Example of text generation:

.. code-block:: python
    from transformers import {processor_class}
    from rktransformers import {model_class}

    tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    model = {model_class}.from_pretrained("{checkpoint}")

    inputs = tokenizer("My name is Arthur and I live in", return_tensors="np")

    gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
```

Example using `transformers.pipelines`:

```python
    from transformers import {processor_class}, pipeline
    from rktransformers import {model_class}

    tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    model = {model_class}.from_pretrained("{checkpoint}")
    rknn_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    text = "My name is Arthur and I live in"
    gen = rknn_gen(text)
```
"""


@add_end_docstrings(RKNN_MODEL_END_DOCSTRING)
class RKModelForCausalLM(
    RKModel[CausalLMOutputWithPast, tuple[torch.Tensor | np.ndarray, tuple[tuple[torch.Tensor]] | None]],
    GenerationMixin,
):
    """
    RKNN model with a causal language modeling head for RKNN Runtime inference.
    This class officially supports bloom, codegen, falcon, gpt2, gpt-bigcode, gpt_neo, gpt_neox, gptj, llama.
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"
    _supports_cache_class = False
    _is_stateful = False

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
        use_cache: bool = True,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id=model_id,
            config=config,
            model_path=model_path,
            platform=platform,
            core_mask=core_mask,
            rknn_config=rknn_config,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )

        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        # RKNN models don't expose output names in the same way as ONNX, so we might need to infer or rely on convention
        # For now, we assume outputs are logits + KV pairs if use_cache is True
        # self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]

        # We can't easily check input/output names from RKNN runtime without loading,
        # but we can infer capability from config
        self.can_use_cache = use_cache  # Simplified assumption for now, ideally check model inputs
        self.is_merged = False  # "use_cache_branch" in self.input_names # TODO: Check if RKNN supports merged models

        if generation_config is None:
            if model_id is not None:
                with contextlib.suppress(OSError, TypeError):
                    generation_config = GenerationConfig.from_pretrained(
                        model_id,
                    )

            if generation_config is None:
                logger.info("Generation config file not found, creating a new one from model config.")
                generation_config = GenerationConfig.from_model_config(config)

        self.generation_config = generation_config
        self.generation_config.use_cache = use_cache

        # Handle model-specific KV cache dimensions
        if self.config.model_type in {"gemma", "gpt_oss", "nemotron"}:
            self.embed_size_per_head = self.config.head_dim
        elif self.config.model_type == "deepseek_v3":
            # For deepseek_v3, keys and values have different head dimensions
            self.qk_head_dim = self.config.qk_rope_head_dim + self.config.qk_nope_head_dim
            self.v_head_dim = self.config.v_head_dim
        else:
            # Default assumption
            self.embed_size_per_head = self.config.hidden_size // self.config.num_attention_heads

        if self.config.model_type in {
            "arcee",
            "deepseek_v3",
            "cohere",
            "gemma",
            "glm",
            "granite",
            "gpt_oss",
            "helium",
            "mistral",
            "llama",
            "nemotron",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "smollm3",
            "stablelm",
        }:
            self.num_key_value_heads = self.config.num_key_value_heads
        elif self.config.model_type == "falcon":
            if self.config.new_decoder_architecture or not self.config.multi_query:
                self.num_key_value_heads = self.config.num_kv_heads
            else:
                self.num_key_value_heads = 1
        elif self.config.model_type == "gpt_bigcode":
            if self.config.multi_query:
                self.num_key_value_heads = 1
            else:
                self.num_key_value_heads = self.config.num_attention_heads
        else:
            self.num_key_value_heads = self.config.num_attention_heads

    @add_start_docstrings_to_model_forward(
        CAUSALLM_RKNN_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=TOKENIZER_FOR_DOC,
            model_class="RKModelForCausalLM",
            checkpoint="optimum/gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor | np.ndarray,
        attention_mask: torch.LongTensor | np.ndarray | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        position_ids: torch.LongTensor | np.ndarray | None = None,
        use_cache: bool | None = None,
        *,
        return_dict: bool = True,
        **kwargs,
    ):
        self._warn_on_unhandled_inputs(kwargs)

        use_torch = isinstance(input_ids, torch.Tensor)
        use_cache = use_cache if use_cache is not None else self.can_use_cache

        if use_cache and not self.can_use_cache:
            raise ValueError(
                "`use_cache=True` was passed but the model does not support KV cache reuse. "
                "Export the model with `task='auto'` or `task='text-generation-with-past'` to enable KV cache."
            )

        # Get dimensions
        batch_size, in_seq_len = input_ids.shape
        past_seq_len = past_key_values[0][1].shape[-2] if past_key_values is not None else 0
        out_seq_len = past_seq_len + in_seq_len

        # Prepare position_ids if needed
        if position_ids is None and "position_ids" in self.input_names:
            if use_torch:
                position_ids = (
                    torch.arange(past_seq_len, out_seq_len, dtype=torch.long, device=input_ids.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
            else:
                position_ids = (
                    np.arange(past_seq_len, out_seq_len, dtype=np.int64).reshape(1, -1).repeat(batch_size, axis=0)
                )

        # Prepare attention_mask
        if attention_mask is None:
            attention_mask = self._ones_like(input_ids, use_torch)

        # Construct model inputs dict
        model_inputs: dict[str, torch.Tensor | np.ndarray | None] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids

        # Handle past_key_values for models with KV cache
        if len(self.key_value_input_names) > 0:
            if past_key_values is None:
                # Initialize empty past_key_values for prefill phase
                k_shape = (batch_size, self.num_key_value_heads, 0, self.embed_size_per_head)
                v_shape = (batch_size, self.num_key_value_heads, 0, self.embed_size_per_head)

                if use_torch:
                    k_tensor = torch.zeros(k_shape, dtype=torch.float32, device=self.device)
                    v_tensor = torch.zeros(v_shape, dtype=torch.float32, device=self.device)
                else:
                    k_tensor = np.zeros(k_shape, dtype=np.float32)
                    v_tensor = np.zeros(v_shape, dtype=np.float32)

                past_key_values = tuple(k_tensor if ".key" in name else v_tensor for name in self.key_value_input_names)
            elif isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
                # Flatten from tuple of tuples to flat tuple
                past_key_values = sum(past_key_values, ())

            # Add to model_inputs
            model_inputs.update(zip(self.key_value_input_names, past_key_values, strict=False))

        # Prepare ordered inputs for RKNN inference
        ordered_inputs: list[np.ndarray] = []
        for name in self.input_names:
            tensor = model_inputs.get(name)
            if tensor is None:
                continue

            # Determine dtype based on input type
            if any(keyword in name for keyword in ["input_ids", "attention_mask", "position_ids", "token_type_ids"]):
                target_dtype = np.int16  # NPU-optimized integer type
            else:
                target_dtype = np.float32  # KV cache and other float tensors

            ordered_inputs.append(self._tensor_to_numpy(tensor, np.dtype(target_dtype)))

        # Run RKNN inference
        if self.rknn is None:
            raise RuntimeError("RKNN runtime has been released.")

        with suppress_output():
            outputs = self.rknn.inference(inputs=ordered_inputs)

        if outputs is None:
            input_summaries = [f"shape={arr.shape}, dtype={arr.dtype}" for arr in ordered_inputs]
            raise RuntimeError(f"RKNN inference returned None. Input summary: {input_summaries}")

        # Process outputs
        logits = self._torch_if_needed(use_torch, outputs[0])

        new_past_key_values = None
        if use_cache and len(outputs) > 1:
            # Reconstruct past_key_values from flat output list
            pkv_outputs = [self._torch_if_needed(use_torch, out) for out in outputs[1:]]
            new_past_key_values = tuple(pkv_outputs[i : i + 2] for i in range(0, len(pkv_outputs), 2))

        if not return_dict:
            return (logits, new_past_key_values)

        return CausalLMOutputWithPast(logits=logits, past_key_values=new_past_key_values)  # type: ignore

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Reuse ORT logic or adapt
        # ORT has `_prepare_inputs_for_generation_legacy` and super().
        # We can probably just use a simplified version similar to ORT.

        position_ids = kwargs.get("position_ids")

        if past_key_values is not None:
            # Cut input_ids if we have past
            # ORT logic:
            # past_seq_len = ...
            # input_ids = input_ids[:, past_seq_len:]

            # We need to calculate past_seq_len
            # Assuming tuple(tuple(k, v))
            past_seq_len = past_key_values[0][1].shape[-2]

            if input_ids.shape[1] > past_seq_len:
                input_ids = input_ids[:, past_seq_len:]

            # Update position_ids to reflect the new input_ids length
            if position_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache"),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Reuse ORT logic for reordering cache during beam search
        # ... (Implementation similar to ORT)
        return tuple(
            tuple(past.index_select(0, beam_idx.to(past.device)) for past in layer_past)
            for layer_past in past_key_values
        )
