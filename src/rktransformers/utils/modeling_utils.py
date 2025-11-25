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

"""Utilities for model inference"""

import contextlib
import os

from transformers.utils.hub import cached_file


def check_sentence_transformer_support(
    output_dir: str,
    model_id_or_path: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> bool:
    """
    Check if the model is a Sentence Transformer model.

    Supports both local paths and remote HuggingFace Hub models.

    Args:
        output_dir: Output directory to check for ST config.
        model_id_or_path: Model ID or path to check (local or remote).
        cache_dir: Optional cache directory for HuggingFace Hub downloads.
        revision: Optional model revision to use.
        token: Optional HuggingFace Hub token.
        local_files_only: If True, only check local files.

    Returns:
        True if Sentence Transformer configs are found.
    """
    st_config_files = ["config_sentence_transformers.json", "sentence_bert_config.json"]

    # Check output directory (always local)
    for config_file in st_config_files:
        if os.path.exists(os.path.join(output_dir, config_file)):
            return True

    # Check source directory/model_id
    # First try local path check (fast and works for export use case)
    if os.path.isdir(model_id_or_path):
        for config_file in st_config_files:
            if os.path.exists(os.path.join(model_id_or_path, config_file)):
                return True
    else:
        # model_id_or_path might be a remote model ID or a cached model
        # Try to check using HuggingFace Hub utilities
        for config_file in st_config_files:
            with contextlib.suppress(Exception):
                cached_path = cached_file(
                    model_id_or_path,
                    filename=config_file,
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                    local_files_only=local_files_only,
                    _raise_exceptions_for_missing_entries=False,
                )
                if cached_path is not None:
                    return True

    return False
