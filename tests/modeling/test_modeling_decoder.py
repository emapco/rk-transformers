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

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from rktransformers.modeling_decoder import RKModelForCausalLM


class TestRKModelForCausalLM:
    @pytest.fixture
    def pretrained_config(self):
        config = PretrainedConfig(
            model_type="gpt2",
            vocab_size=1000,
            n_layer=2,
            n_head=4,
            n_embd=32,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=1,
            hidden_size=32,
            num_attention_heads=4,
        )
        return config

    def test_initialization(self, mock_rknn_lite, dummy_rknn_file, pretrained_config):
        model = RKModelForCausalLM(config=pretrained_config, model_path=dummy_rknn_file)
        assert model.config.model_type == "gpt2"
        assert model.can_use_cache is True

    def test_forward_no_cache(self, mock_rknn_lite, dummy_rknn_file, pretrained_config):
        mock_rknn = mock_rknn_lite.return_value
        # Mock output: logits [batch, seq_len, vocab_size], past_key_values (2 layers * 2 tensors)
        batch_size = 1
        seq_len = 5
        vocab_size = 1000
        head_dim = 8  # 32 // 4

        logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        # 2 layers * 2 (k,v) = 4 tensors
        kv_outputs = [np.random.randn(batch_size, 4, seq_len, head_dim).astype(np.float32) for _ in range(4)]

        mock_rknn.inference.return_value = [logits] + kv_outputs

        model = RKModelForCausalLM(config=pretrained_config, model_path=dummy_rknn_file)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        outputs = model(input_ids=input_ids)

        assert outputs.logits.shape == (batch_size, seq_len, vocab_size)
        assert len(outputs.past_key_values) == 2  # 2 layers
        assert len(outputs.past_key_values[0]) == 2  # k, v
        assert outputs.past_key_values[0][0].shape == (batch_size, 4, seq_len, head_dim)

    def test_forward_with_cache(self, mock_rknn_lite, dummy_rknn_file, pretrained_config):
        mock_rknn = mock_rknn_lite.return_value
        batch_size = 1
        seq_len = 1
        vocab_size = 1000
        head_dim = 8

        logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        kv_outputs = [np.random.randn(batch_size, 4, seq_len + 5, head_dim).astype(np.float32) for _ in range(4)]

        mock_rknn.inference.return_value = [logits] + kv_outputs

        model = RKModelForCausalLM(config=pretrained_config, model_path=dummy_rknn_file)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Create dummy past_key_values
        past_key_values = tuple(
            (torch.randn(batch_size, 4, 5, head_dim), torch.randn(batch_size, 4, 5, head_dim)) for _ in range(2)
        )

        outputs = model(input_ids=input_ids, past_key_values=past_key_values)

        assert outputs.logits.shape == (batch_size, seq_len, vocab_size)
        assert outputs.past_key_values[0][0].shape == (batch_size, 4, 6, head_dim)

    def test_prepare_inputs_for_generation(self, mock_rknn_lite, dummy_rknn_file, pretrained_config):
        model = RKModelForCausalLM(config=pretrained_config, model_path=dummy_rknn_file)

        input_ids = torch.randint(0, 1000, (1, 10))
        past_key_values = tuple((torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8)) for _ in range(2))

        inputs = model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)

        # Should have sliced input_ids to last 5 tokens (10 - 5)
        # Wait, if input_ids has length 10 and past has length 5.
        # The new tokens are the last 5?
        # Usually prepare_inputs_for_generation is called with the FULL input_ids sequence.
        # And we want to pass only the NEW tokens if we have past.
        # So if input_ids is length 10, and past is length 5, we expect input_ids to become length 5.

        assert inputs["input_ids"].shape == (1, 5)
        assert inputs["past_key_values"] is not None
