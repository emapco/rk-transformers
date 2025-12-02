# Copyright 2025 Emmanuel Cortes. All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

from rktransformers.configuration import RKNNConfig
from rktransformers.exporters.rknn.convert import export_rknn


class TestDecoderExport(unittest.TestCase):
    def setUp(self):
        self.config = RKNNConfig(
            model_name_or_path="local/path/to/model.onnx",
            output_path="output/model.rknn",
            target_platform="rk3588",
            batch_size=1,
            max_seq_length=128,
        )

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.load_model_config")
    @patch("rktransformers.exporters.rknn.convert.get_onnx_input_names")
    def test_export_decoder_model(self, mock_get_inputs, mock_load_config, mock_rknn_cls):
        # Mock RKNN
        mock_rknn = MagicMock()
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_cls.return_value = mock_rknn

        # Mock inputs (Decoder style)
        mock_get_inputs.return_value = [
            "input_ids",
            "attention_mask",
            "past_key_values.0.key",
            "past_key_values.0.value",
        ]

        # Mock config
        mock_model_config = MagicMock()
        mock_model_config.num_attention_heads = 4
        mock_model_config.hidden_size = 64  # head_dim = 16
        mock_model_config.num_key_value_heads = 4
        mock_load_config.return_value = mock_model_config

        # Run export
        export_rknn(self.config)

        # Verify load_onnx call args
        call_args = mock_rknn.load_onnx.call_args
        self.assertIsNotNone(call_args)
        kwargs = call_args.kwargs
        input_size_list = kwargs["input_size_list"]

        # Expected shapes:
        # input_ids: [1, 1]
        # attention_mask: [1, 128] (fallback to seq_len)
        # past_key_values: [1, 4, 127, 16] (128 - 1)

        # Note: In my implementation I used:
        # input_ids: [batch_size, 1]
        # attention_mask: [batch_size, sequence_length]
        # past_key_values: [batch_size, num_heads, sequence_length, head_dim]

        expected_sizes = [
            [1, 1],  # input_ids
            [1, 128],  # attention_mask
            [1, 4, 127, 16],  # past_key_values.0.key
            [1, 4, 127, 16],  # past_key_values.0.value
        ]

        self.assertEqual(input_size_list, expected_sizes)

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.load_model_config")
    @patch("rktransformers.exporters.rknn.convert.get_onnx_input_names")
    def test_export_encoder_model(self, mock_get_inputs, mock_load_config, mock_rknn_cls):
        # Mock RKNN
        mock_rknn = MagicMock()
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn_cls.return_value = mock_rknn

        # Mock inputs (Encoder style)
        mock_get_inputs.return_value = [
            "input_ids",
            "attention_mask",
        ]

        # Mock config
        mock_model_config = MagicMock()
        mock_load_config.return_value = mock_model_config

        # Run export
        export_rknn(self.config)

        # Verify load_onnx call args
        call_args = mock_rknn.load_onnx.call_args
        self.assertIsNotNone(call_args)
        kwargs = call_args.kwargs
        input_size_list = kwargs["input_size_list"]

        # Expected shapes:
        # input_ids: [1, 128]
        # attention_mask: [1, 128]

        expected_sizes = [
            [1, 128],
            [1, 128],
        ]

        self.assertEqual(input_size_list, expected_sizes)
