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
"""Utilities for import checks"""

from importlib.metadata import PackageNotFoundError, metadata


def check_package_availability(package_name: str, owner: str) -> bool:
    """
    Checks if a package is available from the correct owner.
    """
    try:
        meta = metadata(package_name)
        home_page = meta.get("Home-page")
        if home_page is None:
            return meta["Name"] == package_name
        return meta["Name"] == package_name and owner in home_page
    except PackageNotFoundError:
        return False


def is_sentence_transformers_available() -> bool:
    """
    Returns True if the SentenceTransformers library is available.
    """
    return check_package_availability("sentence-transformers", "huggingface")
