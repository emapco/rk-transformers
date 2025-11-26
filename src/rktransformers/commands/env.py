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

import platform

from optimum.version import __version__ as optimum_version
from transformers import __version__ as transformers_version
from transformers.utils.import_utils import is_torch_available

from rktransformers.commands.base import BaseRKNNCLICommand, CommandInfo
from rktransformers.utils.env_utils import (
    get_edge_host_platform,
    get_librknnrt_version,
    get_rknn_toolkit_version,
    get_rktransformers_version,
    get_rockchip_board,
)


class RKNNEnvCommand(BaseRKNNCLICommand):
    COMMAND = CommandInfo(name="env", help="Get information about the environment used.")

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"

    def run(self):
        pt_version = "not installed"
        if is_torch_available():
            import torch

            pt_version = torch.__version__

        info = {
            "Operating system": platform.platform(),
            "Rockchip Board": get_rockchip_board(),
            "Rockchip SoC": get_edge_host_platform() or "Not Detected",
            "RKNN Runtime version": get_librknnrt_version(),
            "RKNN Toolkit version": get_rknn_toolkit_version(),
            "Python version": platform.python_version(),
            "PyTorch version": pt_version,
            "HuggingFace transformers version": transformers_version,
            "HuggingFace optimum version": optimum_version,
            "rk-transformers version": get_rktransformers_version(),
        }

        print("\nCopy-and-paste the text below in your GitHub issue:\n")
        print(self.format_dict(info))

        return info
