Model Export
============

This guide explains how to export Hugging Face Transformer models to RKNN format for deployment on Rockchip NPUs.

Export Workflow
---------------

The export process consists of three main steps:

1. **ONNX Export**: Convert Hugging Face model to ONNX using Optimum
2. **RKNN Build**: Load ONNX and build RKNN model with quantization/optimization
3. **Configuration**: Save model configuration for runtime loading

Command-Line Export
-------------------

Basic Export
~~~~~~~~~~~~

.. code-block:: bash

   rk-transformers-cli export \
     --model sentence-transformers/all-MiniLM-L6-v2 \
     --platform rk3588 \
     --optimization-level 3

Key Parameters
~~~~~~~~~~~~~~

Required Arguments
^^^^^^^^^^^^^^^^^^

- ``-m, --model``: Path to ONNX model file or Hugging Face model ID.
- ``output``: Path indicating the directory or file where to store the generated RKNN model. Defaults to the parent directory of the model file or the Hugging Face model directory.

Optional Arguments
^^^^^^^^^^^^^^^^^^

- ``-bs, --batch-size``: Batch size for input shapes (default: 1). Example: batch_size=1 → [1, seq_len].
- ``-msl, --max-seq-length``: Max sequence length for input shapes. Auto-detected from model config if not specified (fallback: 512).
- ``--task-kwargs``: Task-specific keyword arguments for ONNX export as comma-separated key=value pairs. Example: ``num_choices=4``.
- ``--model-inputs``: Comma-separated list of model input names (e.g., 'input_ids,attention_mask'). Auto-detected based on model's type_vocab_size.
- ``--platform``: Target platform. Choices: rk3588, rk3576, rk3568, rk3566, rk3562 (default: rk3588).

Optimization Arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``-o, --optimization-level``: RKNN Optimization level (0-3). Default: 0.
- ``-fa, --flash-attention``: Enable Flash Attention optimization.
- ``--compress-weight``: Compress model weights to reduce RKNN model size.
- ``--single-core-mode``: Enable single NPU core mode (only applicable for rk3588). Reduces model size.
- ``--enable-custom-kernels``: Enable custom kernels (e.g., CumSum) for operations not supported by RKNN.

Quantization Arguments
^^^^^^^^^^^^^^^^^^^^^^

- ``-q, --quantize``: Enable quantization. Otherwise, the model will be exported as float16.
- ``-dt, --dtype``: Quantization data type. Options: w8a8 (default), w8a16, w16a16i, w16a16i_dfp, w4a16.
- ``-a, --algorithm``: Quantization algorithm. Options: normal (default), mmse, kl_divergence, gdq.
- ``-qm, --quantized-method``: Quantization method. Options: layer, channel (default).
- ``--auto-hybrid-cos-thresh``: Cosine distance threshold for automatic hybrid quantization (default: 0.98).
- ``--auto-hybrid-euc-thresh``: Euclidean distance threshold for automatic hybrid quantization (default: None).

Dataset Arguments
^^^^^^^^^^^^^^^^^

- ``-d, --dataset``: HuggingFace dataset name for quantization (e.g. 'sentence-transformers/natural-questions').
- ``-dsb, --dataset-subset``: Subset name for the dataset.
- ``-dsz, --dataset-size``: Number of samples to use for quantization (default: 128).
- ``-dsp, --dataset-split``: Comma-separated list of dataset splits to use. Auto-detected if not specified.
- ``-dc, --dataset-columns``: Comma-separated list of dataset columns to use for calibration.

Optimum Arguments
^^^^^^^^^^^^^^^^^

- ``--opset``: ONNX opset version (default: 19). Recommended: 18+.
- ``--task``: ONNX task type for export (default: auto).

Hugging Face Hub Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``--push-to-hub``: Push the exported model to the Hugging Face Hub.
- ``--model-id``: The repository ID to push to on the Hugging Face Hub.
- ``--token``: The token to use to push to the Model Hub.
- ``--private-repo``: Indicates whether the repository created should be private.
- ``--create-pr``: Whether to create a Pull Request instead of pushing directly to the main branch.

Optimization Levels
-------------------

RKNN supports 4 optimization levels (0-3):

.. list-table::
   :header-rows: 1
   :widths: 10 40 20 30

   * - Level
     - Description
     - Speed
     - Model Size
   * - O0
     - No optimization
     - Slowest
     - Largest
   * - O1
     - Basic optimizations
     - Fast
     - Medium
   * - O2
     - Moderate optimizations
     - Faster
     - Smaller
   * - O3
     - Aggressive optimizations (recommended)
     - Fastest
     - Smallest

Programmatic Export
-------------------

For more control, use the Python API. You can configure the export process using :class:`~rktransformers.configuration.RKNNConfig`, :class:`~rktransformers.configuration.OptimizationConfig`, and :class:`~rktransformers.configuration.QuantizationConfig`.

.. code-block:: python

   from rktransformers import (
       OptimizationConfig,
       QuantizationConfig,
       RKNNConfig,
   )
   from rktransformers.exporters.rknn.convert import export_rknn

   config = RKNNConfig(
       model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
       output_path="./my-exported-model",
       target_platform="rk3588",
       batch_size=1,
       max_seq_length=128,
       quantization=QuantizationConfig(
           do_quantization=True,
           quantized_dtype="w8a8",
           dataset_name="wikitext",
           dataset_size=100,
       ),
       optimization=OptimizationConfig(
           optimization_level=3,
           enable_flash_attention=True,
       ),
   )

   export_rknn(config)

See :class:`~rktransformers.configuration.RKNNConfig` for all available options.

Push to Hugging Face Hub
-------------------------

Share your exported model on the Hugging Face Hub:

.. code-block:: bash

   rk-transformers-cli export \
     --model sentence-transformers/all-MiniLM-L6-v2 \
     --platform rk3588 \
     --push-to-hub \
     --repo-id my-username/my-model-rk3588

This will:

1. Export the model to RKNN format
2. Generate a model card with usage examples
3. Push to the specified repository on Hugging Face Hub

Troubleshooting
---------------

Unsupported Operators
~~~~~~~~~~~~~~~~~~~~~

If your model uses operators not supported by RKNN, please refer to the :ref:`operator-support` documentation for more details on supported operators and workarounds.

Conversion Failures
~~~~~~~~~~~~~~~~~~~

If export fails:

1. Check RKNN toolkit version compatibility
2. Verify model architecture is supported
3. Try different optimization levels
4. Try different opset versions
5. Try different quantization settings
6. Disable flash attention
7. Use different batch size or sequence length
8. Submit a clear and descriptive issue on GitHub with the error message and model details

Memory Issues
~~~~~~~~~~~~~

For large models or long sequences, you may encounter memory issues. Please refer to the :ref:`memory-constraints` documentation for more details on memory constraints and optimization strategies.
