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
"""Utilities for environment detection"""

import contextlib
import importlib.util
import os
import re
import subprocess


def get_librknnrt_version():
    """Read librknnrt library version"""
    with contextlib.suppress(Exception):
        output = subprocess.check_output(
            ["sh", "-c", 'strings /usr/lib/librknnrt.so | grep "librknnrt version:"'],
            stderr=subprocess.DEVNULL,
        )
        ver = output.decode("utf-8").strip()
        ver = ver.replace("librknnrt version: ", "")
        return re.sub(r"\s*\(.*\)$", "", ver)
    return "Not Detected"


def get_rknpu_driver_version():
    """Read RKNPU2 driver version"""
    try:
        with open("/sys/kernel/debug/rknpu/version") as f:
            return f.read().strip().split(":")[-1].strip()
    except Exception:
        return "Not Detected"


def get_rockchip_board():
    """Read Rockchip board name from device tree.

    Returns:
        str: Rockchip board name, or "Not Detected".
    """
    paths = ["/proc/device-tree/model", "/sys/firmware/devicetree/base/model"]
    for p in paths:
        if os.path.exists(p):
            with contextlib.suppress(Exception):
                with open(p, "rb") as f:
                    data = f.read().rstrip(b"\x00")
                model = data.decode("ascii", errors="ignore").strip()
                if model:
                    return model
    return "Not Detected"


def get_rknn_toolkit_version():
    """Read rknn-toolkit2 or rknn-toolkit-lite2 version

    Returns:
        str: Version string of the installed RKNN toolkit, or "Not Installed".
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return "Unknown (importlib.metadata not available)"

    # Check for rknn-toolkit2
    try:
        ver = version("rknn-toolkit2")
        return f"rknn-toolkit2=={ver}"
    except PackageNotFoundError:
        pass

    # Check for rknn-toolkit-lite2
    try:
        ver = version("rknn-toolkit-lite2")
        return f"rknn-toolkit-lite2=={ver}"
    except PackageNotFoundError:
        pass

    # Fallback to checking imports if package metadata is missing
    if importlib.util.find_spec("rknn.api"):
        return "rknn-toolkit2 (version not found)"
    elif importlib.util.find_spec("rknnlite.api"):
        return "rknn-toolkit-lite2 (version not found)"

    return "Not Installed"


def get_edge_host_platform() -> str | None:
    """Detect the edge device host platform by reading `/proc/device-tree/compatible`.

    Returns:
        str | None: Lowercase platform identifier if detected, otherwise None.
    """
    try:
        with open("/proc/device-tree/compatible") as file:
            # Compatible devices taken from rknn.api.RKNN.init_runtime source code.
            compatible = file.read().lower()
            if "rk3588" in compatible:
                return "rk3588"
            if "rk3576" in compatible:
                return "rk3576"
            if "rk3568" in compatible:
                return "rk3568"
            if "rk3566" in compatible:
                return "rk3566"
            if "rk3562" in compatible:
                return "rk3562"
            if "rv1126b" in compatible:
                return "rv1126b"
            if "rv1106b" in compatible:
                return "rv1106b"
            if "rv1106" in compatible:
                return "rv1106"
            if "rv1103b" in compatible:
                return "rv1103b"
            if "rv1103" in compatible:
                return "rv1103"
    except OSError:
        pass
    return None


def is_rockchip_platform() -> bool:
    """Check if the current platform is a Rockchip device."""
    return get_edge_host_platform() is not None
