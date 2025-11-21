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

import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock rknnlite before importing modeling
# This needs to happen before rkruntime.modeling is imported
if "rknnlite" not in sys.modules:
    sys.modules["rknnlite"] = MagicMock()
    sys.modules["rknnlite"].__spec__ = MagicMock()
    sys.modules["rknnlite.api"] = MagicMock()
    sys.modules["rknnlite.api"].__spec__ = MagicMock()

    # Mock RKNNLite class
    class MockRKNNLite:
        NPU_CORE_AUTO = 0
        NPU_CORE_0 = 1
        NPU_CORE_1 = 2
        NPU_CORE_2 = 4
        NPU_CORE_0_1 = 3
        NPU_CORE_0_1_2 = 7
        NPU_CORE_ALL = 0xFFFF

        def __init__(self, verbose=False):
            self.verbose = verbose

        def load_rknn(self, path):
            return 0

        def init_runtime(self, core_mask=None):
            return 0

        def list_support_target_platform(self, model_path):
            # Default behavior: return a dict of platforms
            return OrderedDict([("filled_target_platform", ["rk3588"]), ("support_target_platform", ["RK3588"])])

    sys.modules["rknnlite.api"].RKNNLite = MockRKNNLite

# Import module under test
from rkruntime.modeling import PretrainedConfig, RKRTModel


class TestRKNNPlatformCheck(unittest.TestCase):
    def setUp(self):
        self.config = PretrainedConfig(model_type="rknn_model")
        self.model_path = Path("dummy.rknn")
        self.model_path.touch()  # Create dummy file

    def tearDown(self):
        if self.model_path.exists():
            self.model_path.unlink()

    @patch("rkruntime.modeling.RKNNLite")
    def test_compatible_platform(self, MockRKNNLiteClass):
        # Setup mock
        mock_rknn = MockRKNNLiteClass.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["RK3588"])]
        )

        # Test with compatible platform
        model = RKRTModel(config=self.config, model_path=self.model_path, platform="rk3588")
        self.assertIsNotNone(model.rknn)

        # Verify list_model_compatible_platform
        supported = model.list_model_compatible_platform()
        self.assertEqual(
            supported,
            OrderedDict([("filled_target_platform", ["rk3588"]), ("support_target_platform", ["RK3588"])]),
        )

    @patch("rkruntime.modeling.RKNNLite")
    def test_incompatible_platform(self, MockRKNNLiteClass):
        # Setup mock
        mock_rknn = MockRKNNLiteClass.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["RK3588"])]
        )

        # Test with incompatible platform
        with self.assertRaises(RuntimeError) as cm:
            RKRTModel(
                config=self.config,
                model_path=self.model_path,
                platform="rk3566",  # Not in supported list
            )
        self.assertIn("not compatible", str(cm.exception))
        self.assertIn("rk3566", str(cm.exception))

    @patch("rkruntime.modeling.RKNNLite")
    def test_case_insensitivity(self, MockRKNNLiteClass):
        # Setup mock
        mock_rknn = MockRKNNLiteClass.return_value
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        # Mock returns uppercase, we pass lowercase
        mock_rknn.list_support_target_platform.return_value = OrderedDict(
            [("filled_target_platform", ["rk3588"]), ("support_target_platform", ["RK3588"])]
        )

        # Test with compatible platform (lowercase)
        model = RKRTModel(config=self.config, model_path=self.model_path, platform="rk3588")
        self.assertIsNotNone(model.rknn)
