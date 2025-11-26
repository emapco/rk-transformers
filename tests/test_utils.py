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

"""Tests for utility modules."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock, mock_open, patch

import pytest

from rktransformers.utils.env_utils import (
    get_edge_host_platform,
    get_librknnrt_version,
    get_rknn_toolkit_version,
    get_rockchip_board,
    is_rockchip_platform,
)
from rktransformers.utils.import_utils import check_package_availability, is_sentence_transformers_available


class TestImportUtils:
    """Tests for import_utils module."""

    @pytest.mark.parametrize(
        "metadata_return,expected",
        [
            ({"Name": "my-package", "Home-page": "https://github.com/owner/repo"}, True),
            ({"Name": "my-package", "Home-page": "https://github.com/other/repo"}, False),
            ({"Name": "my-package"}, True),
        ],
    )
    def test_check_package_availability(self, metadata_return, expected) -> None:
        """Test check_package_availability with various metadata scenarios."""
        with patch("rktransformers.utils.import_utils.metadata") as mock_metadata:
            mock_metadata.return_value = metadata_return
            assert check_package_availability("my-package", "owner") is expected

    def test_check_package_availability_not_found(self) -> None:
        """Test check_package_availability when package is missing."""
        with patch("rktransformers.utils.import_utils.metadata") as mock_metadata:
            mock_metadata.side_effect = PackageNotFoundError
            assert check_package_availability("missing-package", "owner") is False

    def test_is_sentence_transformers_available(self) -> None:
        """Test is_sentence_transformers_available check."""
        with patch("rktransformers.utils.import_utils.check_package_availability") as mock_check:
            mock_check.return_value = True
            assert is_sentence_transformers_available() is True
            mock_check.assert_called_with("sentence-transformers", "huggingface")


class TestEnvUtils:
    """Tests for env_utils module."""

    def test_get_librknnrt_version_success(self) -> None:
        """Test successful retrieval of librknnrt version."""
        with patch("subprocess.check_output") as mock_sub:
            mock_sub.return_value = b"librknnrt version: 1.0.0 (build)"
            assert get_librknnrt_version() == "1.0.0"

    def test_get_librknnrt_version_failure(self) -> None:
        """Test failure handling for librknnrt version."""
        with patch("subprocess.check_output", side_effect=Exception):
            assert get_librknnrt_version() == "Not Detected"

    @pytest.mark.parametrize(
        "path_exists,file_content,expected",
        [
            ("/proc/device-tree/model", b"Orange Pi 5 Plus\x00", "Orange Pi 5 Plus"),
            ("/sys/firmware/devicetree/base/model", b"Orange Pi 5 Plus\x00", "Orange Pi 5 Plus"),
        ],
    )
    def test_get_rockchip_board_success(self, path_exists, file_content, expected) -> None:
        """Test successful detection of Rockchip board."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.side_effect = lambda p: p == path_exists
            with patch("builtins.open", mock_open(read_data=file_content)):
                assert get_rockchip_board() == expected

    def test_get_rockchip_board_not_detected(self) -> None:
        """Test get_rockchip_board when no board is detected."""
        with patch("os.path.exists", return_value=False):
            assert get_rockchip_board() == "Not Detected"

    def test_get_rknn_toolkit_version_toolkit2(self) -> None:
        """Test detection of rknn-toolkit2."""
        with patch("importlib.metadata.version") as mock_ver:
            mock_ver.return_value = "2.0.0"
            assert get_rknn_toolkit_version() == "rknn-toolkit2==2.0.0"

    def test_get_rknn_toolkit_version_lite2(self) -> None:
        """Test detection of rknn-toolkit-lite2."""
        with patch("importlib.metadata.version") as mock_ver:

            def side_effect(name):
                if name == "rknn-toolkit2":
                    raise PackageNotFoundError
                if name == "rknn-toolkit-lite2":
                    return "1.6.0"
                raise PackageNotFoundError

            mock_ver.side_effect = side_effect
            assert get_rknn_toolkit_version() == "rknn-toolkit-lite2==1.6.0"

    def test_get_rknn_toolkit_version_fallback(self) -> None:
        """Test fallback detection via importlib."""
        with (
            patch("importlib.metadata.version", side_effect=PackageNotFoundError),
            patch("importlib.util.find_spec") as mock_spec,
        ):
            # Test rknn.api fallback
            mock_spec.side_effect = lambda name: MagicMock() if name == "rknn.api" else None
            assert get_rknn_toolkit_version() == "rknn-toolkit2 (version not found)"

    def test_get_rknn_toolkit_version_not_installed(self) -> None:
        """Test when no toolkit is installed."""
        with (
            patch("importlib.metadata.version", side_effect=PackageNotFoundError),
            patch("importlib.util.find_spec", return_value=None),
        ):
            assert get_rknn_toolkit_version() == "Not Installed"

    @pytest.mark.parametrize(
        "file_content,expected",
        [
            ("rockchip,rk3588", "rk3588"),
            ("rockchip,rk3566", "rk3566"),
        ],
    )
    def test_get_edge_host_platform_success(self, file_content, expected) -> None:
        """Test successful detection of edge host platform."""
        with patch("builtins.open", mock_open(read_data=file_content)):
            assert get_edge_host_platform() == expected

    def test_get_edge_host_platform_none(self) -> None:
        """Test get_edge_host_platform when file is missing."""
        with patch("builtins.open", side_effect=OSError):
            assert get_edge_host_platform() is None

    @pytest.mark.parametrize("platform_return,expected", [("rk3588", True), (None, False)])
    def test_is_rockchip_platform(self, platform_return, expected) -> None:
        """Test is_rockchip_platform check."""
        with patch("rktransformers.utils.env_utils.get_edge_host_platform", return_value=platform_return):
            assert is_rockchip_platform() is expected
