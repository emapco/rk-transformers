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

import numpy as np

from rktransformers.utils.import_utils import is_rknn_toolkit_available

if is_rknn_toolkit_available():
    from rknn.api.custom_op import get_node_attr
else:
    raise ImportError(
        "rknn-toolkit is not available. Please install the `export` extra. e.g., `pip install rk-transformers[export]`"
    )


class cstCumSum:
    op_type = "cstCumSum"

    def shape_infer(self, node, in_shapes, in_dtypes):
        out_shapes = in_shapes.copy()
        out_dtypes = in_dtypes.copy()
        return out_shapes, out_dtypes

    def compute(self, node, inputs):
        x = inputs[0]
        axis = get_node_attr(node, "axis")
        # In newer ONNX opsets, axis might be the second input
        # if axis is None and len(inputs) > 1:
        #     axis_input = inputs[1]
        #     axis = int(axis_input.item()) if isinstance(axis_input, np.ndarray) else int(axis_input)

        outputs = [np.cumsum(x, axis=axis)]
        return outputs
