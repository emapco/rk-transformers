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

from unittest.mock import patch

import pytest

from rktransformers.utils.import_utils import (
    is_rknn_toolkit_available,
)

pytestmark = pytest.mark.skipif(
    not is_rknn_toolkit_available(),
    reason="Skipping tests that require the `export` extra but it's not installed.",
)

from rktransformers.exporters.rknn.utils import resolve_hub_repo_id  # noqa: E402


class TestResolveHubRepoId:
    """Tests for Hub repository ID resolution and namespace auto-detection."""

    def test_repo_id_with_namespace_unchanged(self):
        """Test that repo IDs with namespace are returned unchanged."""
        # Should work without token since it doesn't need auto-detection
        result = resolve_hub_repo_id("username/my-model")
        assert result == "username/my-model"

        result = resolve_hub_repo_id("org/sub/model")
        assert result == "org/sub/model"

    def test_repo_id_none_returns_none(self):
        """Test that None input returns None."""
        result = resolve_hub_repo_id(None)
        assert result is None

        result = resolve_hub_repo_id(None, hub_token="fake_token")
        assert result is None

    @patch("huggingface_hub.whoami")
    def test_auto_detect_username_success(self, mock_whoami):
        """Test successful username auto-detection via whoami() API."""
        # Mock successful whoami response
        mock_whoami.return_value = {"name": "testuser"}

        result = resolve_hub_repo_id("my-model", hub_token="fake_token")
        assert result == "testuser/my-model"

        # Verify whoami was called with correct token
        mock_whoami.assert_called_once_with(token="fake_token")

    @patch("huggingface_hub.whoami")
    def test_auto_detect_username_no_name_in_response(self, mock_whoami):
        """Test error handling when whoami() returns no username."""
        # Mock whoami response without 'name' field
        mock_whoami.return_value = {}

        with pytest.raises(ValueError) as exc_info:
            resolve_hub_repo_id("my-model", hub_token="fake_token")

        error_msg = str(exc_info.value)
        assert "Unable to determine username" in error_msg
        assert "Use a fully qualified repository ID" in error_msg

    @patch("huggingface_hub.whoami")
    def test_auto_detect_username_api_failure(self, mock_whoami):
        """Test error handling when whoami() API call fails."""
        # Mock whoami to raise an exception (e.g., network error, invalid token)
        mock_whoami.side_effect = Exception("Invalid authentication token")

        with pytest.raises(ValueError) as exc_info:
            resolve_hub_repo_id("my-model", hub_token="invalid_token")

        error_msg = str(exc_info.value)
        assert "Failed to resolve repository ID" in error_msg
        assert "my-model" in error_msg
        assert "Use a fully qualified repository ID" in error_msg
