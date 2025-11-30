RKNN Limitations
=================

Understanding RKNN's limitations is crucial for successful and efficient deployment.

Dynamic Inputs & Static Shapes
-------------------------------

Current RKNN support for dynamic inputs is **experimental and not fully functional**. As a result, models exported via RK-Transformers CLI use **static input shapes** defined at export time.
Dynamic inputs can be enabled using the programmatic export API for advanced users.

Performance Impact
~~~~~~~~~~~~~~~~~~

The NPU allocates memory based on the static shape. If you export with ``max_seq_length=512`` but only infer on 10 tokens, the NPU still processes the full 512-token padding, leading to inefficient inference.

**Example**:

.. code-block:: python

   # Model exported with max_seq_length=512, batch_size=1
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2"
   )

   # Short input (10 tokens)
   inputs = tokenizer("short text", return_tensors="np")
   # Automatically padded to 512 tokens
   # NPU processes all 512 tokens, wasting computation

.. _input-padding:

Input Padding
~~~~~~~~~~~~~~~

RK-Transformers automatically pads inputs so they match the static shape used at export time:

- input_ids: padded from the actual length (e.g. [1, 10]) to the export length (e.g. [1, 512]) using tokenizer.pad_token_id
- attention_mask: padded with zeros to the export length
- token_type_ids: padded to the export length using tokenizer.pad_token_type_id

This guarantees correct static tensor shapes for the NPU, but may increase computation when the exported sequence length is much larger than typical inputs.

.. warning::
   RK-Transformers only performs padding, not truncation. You must ensure that your input batch size and sequence length are less than or equal to the model's compiled input shapes. Inputs exceeding these dimensions will result in a runtime error.

Recommendations
~~~~~~~~~~~~~~~

1. **Export multiple model variants** for different sequence lengths:

.. code-block:: bash

   # Short sequences (faster)
   rk-transformers-cli export --model bert-base-uncased --max-seq-length 128

   # Medium sequences
   rk-transformers-cli export --model bert-base-uncased --max-seq-length 256

   # Long sequences (slower)
   rk-transformers-cli export --model bert-base-uncased --max-seq-length 512

2. **Export multiple batch sizes** if workload varies:

.. code-block:: bash

   # Single inference
   rk-transformers-cli export --model bert-base-uncased --batch-size 1

   # Batch inference
   rk-transformers-cli export --model bert-base-uncased --batch-size 4

3. **Choose optimal sequence length** based on your data:

.. code-block:: python

   # Analyze your dataset to find optimal max_seq_length
   import math

   import numpy as np
   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
   texts = ["your", "dataset", "texts"]

   # Batch tokenize and obtain lengths if supported
   try:
       enc = tokenizer(
           texts,
           truncation=False,
           padding=False,
           return_length=True,  # Fast tokenizer support
       )
       lengths = np.array(enc["length"], dtype=int)
   except Exception:
       # Fallback for non-fast tokenizers
       lengths = np.array(
           [len(tokenizer.encode(text, truncation=False)) for text in texts],
           dtype=int,
       )

   if lengths.size == 0:
       raise ValueError("No texts to analyze for percentile calculation")

   target_percentile = 0.95
   k = max(0, min(len(lengths) - 1, math.ceil(len(lengths) * target_percentile) - 1))
   target_length = int(np.partition(lengths, k)[k])

   print(f"Mean length: {lengths.mean():.2f}")
   print(f"{int(target_percentile * 100)}th percentile: {target_length}")

.. code-block:: bash

   rk-transformers-cli export --max-seq-length <target_length>

Quantization Support
--------------------

While the tool supports various quantization data types, many are **experimental**.

Supported Datatypes
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Datatype
     - Status
     - Notes
   * - ``float16``
     - Supported
     - No quantization, larger model size
   * - ``w8a8``
     - **Recommended**
     - Widely supported and tested. 8-bit weights and activations
   * - ``w8a16``
     - Experimental
     - May fail on certain models, operators, or SoC platforms
   * - ``w16a16i``
     - Experimental
     - May fail on certain models, operators, or SoC platforms
   * - ``w16a16i_dfp``
     - Experimental
     - May fail on certain models, operators, or SoC platforms
   * - ``w4a16``
     - Experimental
     - May fail on certain models, operators, or SoC platforms

Recommendations
~~~~~~~~~~~~~~~

1. **Always use w8a8 for production**: It's the most stable and widely supported
2. **Test thoroughly** before deploying other datatypes
3. **Fallback to float16** if quantization fails

Example:

.. code-block:: bash

   rk-transformers-cli export \
      --model bert-base-uncased \
      --platform rk3588 \
      --quantize \
      --dtype w8a8 \
      --dataset sentence-transformers/natural-questions \
      --dataset-split train \
      --dataset-columns answer \
      --dataset-size 128 \
      --max-seq-length 128 \
      --batch-size 1

.. _operator-support:

Operator Support
----------------

RKNN currently supports a `subset of ONNX operators <https://github.com/airockchip/rknn-toolkit2/blob/master/doc/RKNNToolKit2_OP_Support-2.3.2.md>`_.

Unsupported Operators
~~~~~~~~~~~~~~~~~~~~~

If your model uses unsupported operators, export may fail with errors like:

.. code-block:: text

   E RKNN: [<time-stamp>] Unsupport NPU op: <operator-name>
   E RKNN: [<time-stamp>] Unsupport CPU op: <operator-name>

Solutions
~~~~~~~~~

**Easy Methods** (limited success):

1. **Change ONNX opset version**:

.. code-block:: bash

   rk-transformers-cli export --model bert-base-uncased --opset 19
   # Try different versions: 14, 15, 16, 17, 18, 19

2. **Run operators on CPU** (requires custom configuration):

Modify export code to specify CPU fallback for specific operators.

.. code-block:: python

   from rktransformers import RKNNConfig

   rknn_config = RKNNConfig(
       op_target={
           "op_id": "cpu",
       }
   )

**Difficult Methods**:

1. **Modify ONNX graph**: Replace unsupported ops with supported alternatives
2. **Register custom operators**: Use ``rknn.register_custom_op()`` in export code. *Currently requires source code modification.*

Checking Operator Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before exporting, you can check if your model's operators are supported:

1. Export model to ONNX:

.. code-block:: bash

   optimum-cli export onnx --model bert-base-uncased onnx_output/

2. Inspect ONNX model:

.. code-block:: python

   import onnx
   model = onnx.load("onnx_output/model.onnx")
   
   # List all operators
   ops = set()
   for node in model.graph.node:
       ops.add(node.op_type)
   
   print("Operators used:", sorted(ops))

3. Compare with `RKNN supported operators <https://github.com/airockchip/rknn-toolkit2/blob/master/doc/RKNNToolKit2_OP_Support-2.3.2.md>`_

Dtype Limitations
-----------------

Input Tensor Types
~~~~~~~~~~~~~~~~~~

RKNN NPUs only support specific input tensor dtypes:

- ``int8``
- ``uint8``
- ``int16``
- ``float16``
- ``float32``

**Not Supported**: ``int64`` (commonly used for ``input_ids`` in transformers)

Impact
^^^^^^

Transformer models typically use ``int64`` for input IDs, but RKNN requires ``int16``. RK-Transformers automatically converts inputs, but this causes:

1. **Type conversion overhead** (minor performance impact)
2. **Potential precision loss** if vocabulary size > 32,767 (rare)

.. code-block:: python

   # Internal conversion (automatic)
   # input_ids: torch.int64 -> np.int16 (for RKNN inference)

Model Weight Types
~~~~~~~~~~~~~~~~~~

- **Float16 models**: Weights stored as ``float16``
- **Quantized models (w8a8)**: Weights stored as ``int8``

.. _memory-constraints:

Memory Constraints
------------------

NPU Memory Limits
~~~~~~~~~~~~~~~~~

Rockchip NPUs have `limited addressable memory (4GB) <https://github.com/ggml-org/llama.cpp/issues/722#issuecomment-1887694026>`_. Large models or long sequences may exceed available memory.

If you encounter memory errors such as segmentation faults or allocation failures during export or inference, try the following:

1. Reduce ``max_seq_length``
2. Reduce ``batch_size``
3. Use quantization (``w8a8``)
4. Modify the model architecture (if possible)

Platform Compatibility
----------------------

Supported Platforms
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Platform
     - NPU Cores
     - TOPS
     - Notes
   * - RK3588
     - 3 cores
     - 6 TOPS
     - Full tested, best performance
   * - RK3576
     - 2 cores
     - 6 TOPS
     - Supported by RKNN 2.3.2
   * - RK3568
     - 1 core
     - 1 TOPS
     - Supported by RKNN 2.3.2
   * - RK3566
     - 1 core
     - 1 TOPS
     - Supported by RKNN 2.3.2
   * - RK3562
     - 1 core
     - 1 TOPS
     - Supported by RKNN 2.3.2

Export Requirements
~~~~~~~~~~~~~~~~~~~

- **Platform**: Linux (x86_64 or arm64)
- **Python**: 3.10-3.12
- **RKNN Toolkit**: 2.3.2

Inference Requirements
~~~~~~~~~~~~~~~~~~~~~~

- **Platform**: Rockchip device with RKNPU2
- **OS**: Linux (Ubuntu, Debian, Armbian, etc.)
- **RKNN Runtime**: 2.3.2 (must match toolkit version)

Version Compatibility
~~~~~~~~~~~~~~~~~~~~~

.. warning::
   RKNN toolkit version must match RKNN runtime version. A model exported with toolkit 2.3.2 requires runtime 2.3.2.

Known Issues
------------

1. **Very Long Sequences**: Sequences > 4096 may cause memory issues

   **Workaround**: Reduce max_seq_length or use model chunking

2. **Cross-Attention Models**: Limited support for encoder-decoder and decoder models. Support additional model architecture is planned.

   **Workaround**: Use encoder-only models when possible

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/emapco/rk-transformers/issues>`_
2. Run diagnostics: ``rk-transformers-cli env``
3. Review `RKNN documentation <https://github.com/airockchip/rknn-toolkit2/tree/master/doc>`_
4. Open a new issue with full error output and environment details
