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

"""Integration tests for RKNN modeling classes."""

import random

import numpy as np
import pytest
from transformers import AutoTokenizer

from rktransformers.modeling import (
    RKRTModelForFeatureExtraction,
    RKRTModelForMaskedLM,
    RKRTModelForMultipleChoice,
    RKRTModelForQuestionAnswering,
    RKRTModelForSequenceClassification,
    RKRTModelForTokenClassification,
)
from rktransformers.utils.env_utils import is_rockchip_platform
from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
    is_rknn_toolkit_lite_available,
)

pytestmark = pytest.mark.skipif(
    not is_rockchip_platform() or (not is_rknn_toolkit_lite_available() and not is_rknn_toolkit_available()),
    reason="Skipping RKNN tests on non-Rockchip platform or missing RKNN Toolkit Lite library.",
)


# RKNN backend gets overloaded when running all tests at once
@pytest.mark.flaky(reruns=1, reruns_delay=random.uniform(15, 30), only_rerun=["RuntimeError"])
@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.integration
@pytest.mark.requires_rknpu
class TestRKModelingIntegration:
    """Integration tests for RKNN modeling classes."""

    def test_feature_extraction(self) -> None:
        """Test RKRTModelForFeatureExtraction integration."""
        checkpoint = "rk-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RKRTModelForFeatureExtraction.from_pretrained(checkpoint)

        inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        assert last_hidden_state.shape == (1, 512, 384)  # (batch_size, seq_length, hidden_size)

    def test_masked_lm(self) -> None:
        """Test RKRTModelForMaskedLM integration."""
        checkpoint = "rk-transformers/bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RKRTModelForMaskedLM.from_pretrained(checkpoint)

        inputs = tokenizer("The capital of France is [MASK].", return_tensors="np")
        outputs = model(**inputs)
        logits = outputs.logits
        assert logits.shape == (1, 512, 30522)  # (batch_size, sequence_length, config.vocab_size)

    def test_sequence_classification(self) -> None:
        """Test RKRTModelForSequenceClassification integration."""
        checkpoint = "rk-transformers/distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RKRTModelForSequenceClassification.from_pretrained(checkpoint)

        inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        outputs = model(**inputs)
        logits = outputs.logits
        assert logits.shape == (1, 2)  # (batch_size, config.num_labels)

    def test_question_answering(self) -> None:
        """Test RKRTModelForQuestionAnswering integration."""
        checkpoint = "rk-transformers/distilbert-base-cased-distilled-squad"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RKRTModelForQuestionAnswering.from_pretrained(checkpoint)

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        inputs = tokenizer(question, text, return_tensors="np")
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        assert start_logits.shape == (1, 512)  # (batch_size, sequence_length)
        assert end_logits.shape == (1, 512)  # (batch_size, sequence_length)

    def test_token_classification(self) -> None:
        """Test RKRTModelForTokenClassification integration."""
        checkpoint = "rk-transformers/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RKRTModelForTokenClassification.from_pretrained(checkpoint)

        inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="np")
        outputs = model(**inputs)
        logits = outputs.logits
        assert logits.shape == (1, 512, 9)  # (batch_size, sequence_length, config.num_labels)

    def test_multiple_choice(self) -> None:
        """Test RKRTModelForMultipleChoice integration."""
        checkpoint = "rk-transformers/bert-base-uncased_SWAG"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = RKRTModelForMultipleChoice.from_pretrained(checkpoint)
        # swag dataset and model use 4 choices
        prompt = "In Italy, pizza is served in slices."
        choice0 = "It is eaten with a fork and knife."
        choice1 = "It is eaten while held in the hand."
        choice2 = "It is blended into a smoothie."
        choice3 = "It is folded into a taco."

        encoding = tokenizer(
            [prompt, prompt, prompt, prompt], [choice0, choice1, choice2, choice3], return_tensors="np", padding=True
        )
        inputs = {k: np.expand_dims(v, 0) for k, v in encoding.items()}

        outputs = model(**inputs)
        logits = outputs.logits
        assert logits.shape == (1, 4)  # (batch_size, num_choices)
