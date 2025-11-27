import unittest
from unittest.mock import MagicMock, patch

from rktransformers.configuration import RKNNConfig
from rktransformers.exporters.rknn.convert import export_rknn
from rktransformers.kernels.cum_sum import cstCumSum

# E   AssertionError: <class 'rktransformers.kernels.cum_sum.cstCumSum'> is not an instance of
# <class 'rktransformers.kernels.cum_sum.cstCumSum'>


class TestCustomOpRegistration(unittest.TestCase):
    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.load_model_config")
    @patch("rktransformers.exporters.rknn.convert.get_onnx_input_names")
    @patch("rktransformers.exporters.rknn.convert.main_export")
    @patch("rktransformers.exporters.rknn.convert.snapshot_download")
    @patch("rktransformers.exporters.rknn.convert.download_sentence_transformer_modules_weights")
    def test_custom_op_registration(
        self,
        mock_download_modules,
        mock_snapshot_download,
        mock_main_export,
        mock_get_input_names,
        mock_load_config,
        mock_rknn_cls,
        mock_onnx,
    ):
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0
        mock_rknn.reg_custom_op.return_value = 0

        mock_load_config.return_value = {}
        mock_get_input_names.return_value = ["input_ids", "attention_mask"]

        # Mock ONNX model loading and saving
        mock_model = MagicMock()
        mock_node = MagicMock()
        mock_node.op_type = "cstCumSum"
        mock_model.graph.node = [mock_node]
        mock_onnx.load.return_value = mock_model

        # Create config with custom kernels enabled
        config = RKNNConfig(
            model_id_or_path="dummy_model.onnx",
            enable_custom_kernels=True,
            target_platform="rk3588",
        )

        # Run export
        export_rknn(config)

        # Verify reg_custom_op was called
        mock_rknn.reg_custom_op.assert_called()

        # Verify it was called with a cstCumSum instance
        args, _ = mock_rknn.reg_custom_op.call_args
        self.assertIsInstance(args[0], cstCumSum)

        # Verify ONNX patching
        mock_onnx.load.assert_called_with("dummy_model.onnx")
        self.assertEqual(mock_node.op_type, "cstCumSum")
        mock_onnx.save.assert_called()

        # Verify load_onnx called with patched model path
        mock_rknn.load_onnx.assert_called()
        call_args = mock_rknn.load_onnx.call_args
        self.assertIn("_patched_cstCumSum.onnx", call_args.kwargs["model"])

    @patch("rktransformers.exporters.rknn.convert.RKNN")
    @patch("rktransformers.exporters.rknn.convert.load_model_config")
    @patch("rktransformers.exporters.rknn.convert.get_onnx_input_names")
    @patch("rktransformers.exporters.rknn.convert.main_export")
    @patch("rktransformers.exporters.rknn.convert.snapshot_download")
    @patch("rktransformers.exporters.rknn.convert.download_sentence_transformer_modules_weights")
    def test_custom_op_no_registration(
        self,
        mock_download_modules,
        mock_snapshot_download,
        mock_main_export,
        mock_get_input_names,
        mock_load_config,
        mock_rknn_cls,
        mock_onnx,
    ):
        mock_rknn = MagicMock()
        mock_rknn_cls.return_value = mock_rknn
        mock_rknn.load_onnx.return_value = 0
        mock_rknn.build.return_value = 0
        mock_rknn.export_rknn.return_value = 0

        mock_load_config.return_value = {}
        mock_get_input_names.return_value = ["input_ids", "attention_mask"]

        config = RKNNConfig(
            model_id_or_path="dummy_model.onnx",
            enable_custom_kernels=False,  # Disable custom kernels
            target_platform="rk3588",
        )

        export_rknn(config)

        mock_rknn.reg_custom_op.assert_not_called()
        # Verify ONNX patching was NOT done
        mock_onnx.load.assert_not_called()
