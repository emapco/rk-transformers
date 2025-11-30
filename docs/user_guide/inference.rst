Inference
=========

This guide covers loading and running inference with RKNN models on Rockchip NPUs.

Loading Models
--------------

From Hugging Face Hub
~~~~~~~~~~~~~~~~~~~~~

Load pre-exported models directly from the Hugging Face Hub:

.. code-block:: python

   from rktransformers import RKModelForFeatureExtraction

   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       core_mask="auto"
   )

From Local Path
~~~~~~~~~~~~~~~

Load from a local directory:

.. code-block:: python

   model = RKModelForFeatureExtraction.from_pretrained(
       "./my-exported-model",
   )

Selecting Specific Model File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a repository contains multiple RKNN files (e.g., different quantization levels):

.. code-block:: python

   # Load INT8 quantized version
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       file_name="rknn/model_w8a8.rknn"
   )

   # Load float16 version
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       file_name="rknn/model.rknn"
   )

Sentence Transformers
~~~~~~~~~~~~~~~~~~~~~

RK-Transformers provides drop-in replacements for ``sentence-transformers`` models.
See the :ref:`Sentence Transformers <sentence-transformers>` integration guide for details on using:

- :ref:`RKSentenceTransformer <rk-sentence-transformer>`: For bi-encoder models
- :ref:`RKCrossEncoder <rk-cross-encoder>`: For cross-encoder models

Running Inference
-----------------

Basic Inference
~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import AutoTokenizer
   from rktransformers import RKModelForFeatureExtraction

   tokenizer = AutoTokenizer.from_pretrained("rk-transformers/all-MiniLM-L6-v2")
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2"
   )

   # Tokenize input
   inputs = tokenizer(
       "Sample text for embedding",
       padding="max_length",
       truncation=True,
       return_tensors="np",
   )

   # Run inference
   outputs = model(**inputs)
   print(outputs.last_hidden_state.shape)

Batch Inference
~~~~~~~~~~~~~~~

Process multiple inputs at once:

.. code-block:: python

   sentences = [
       "First sentence",
       "Second sentence",
       "Third sentence",
   ]

   inputs = tokenizer(
       sentences,
       padding="max_length",
       truncation=True,
       return_tensors="np",
   )

   outputs = model(**inputs)
   print(outputs.last_hidden_state.shape)  # (3, seq_len, hidden_size)

.. note::
   The batch size must match the ``batch_size`` configured during export.
   If you export with ``--batch-size 1``, you can only process one input at a time.

Input Padding
~~~~~~~~~~~~~

See :ref:`Limitations: Input Padding <input-padding>` for details.

Return Types
~~~~~~~~~~~~

Choose between `Hugging Face ModelOutput <https://huggingface.co/docs/transformers/main_classes/output>`_-based dictionary or tuple output:

.. code-block:: python

   # dictionary output (default)
   outputs = model(**inputs, return_dict=True)
   hidden_state = outputs.last_hidden_state

   # Tuple output
   outputs = model(**inputs, return_dict=False)
   hidden_state = outputs[0]

Supported Tasks
---------------

RK-Transformers supports various tasks through specific model classes. See the :ref:`API Reference <task-specific-models>` for full details.

- :class:`~rktransformers.modeling.RKModelForFeatureExtraction`: For computing embeddings
- :class:`~rktransformers.modeling.RKModelForSequenceClassification`: For text classification
- :class:`~rktransformers.modeling.RKModelForTokenClassification`: For named entity recognition, etc.
- :class:`~rktransformers.modeling.RKModelForQuestionAnswering`: For extractive QA
- :class:`~rktransformers.modeling.RKModelForMaskedLM`: For masked language modeling
- :class:`~rktransformers.modeling.RKModelForMultipleChoice`: For multiple choice tasks

Performance Optimization
------------------------

Core Mask Selection
~~~~~~~~~~~~~~~~~~~

Choose the optimal core mask for your workload (see :doc:`npu_cores`):

.. code-block:: python

   # Auto mode (recommended)
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       core_mask="auto"
   )

   # All cores for maximum performance
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       core_mask="all"
   )

Input Tensor Format
~~~~~~~~~~~~~~~~~~~

Use NumPy arrays for better performance (avoid conversion overhead):

.. code-block:: python

   # NumPy (recommended for RKNN)
   inputs = tokenizer(text, return_tensors="np")

   # PyTorch (works but adds conversion overhead)
   inputs = tokenizer(text, return_tensors="pt")

Model Selection
~~~~~~~~~~~~~~~

Choose the right model variant:

- **Float16**: Better accuracy, larger size, slower inference
- **INT8 (w8a8)**: Decent accuracy, smaller size, faster inference

.. code-block:: python

   # Load quantized model for speed
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       file_name="rknn/model_w8a8.rknn"  # INT8 quantized
   )

Error Handling
--------------

Runtime Errors
~~~~~~~~~~~~~~

If inference fails:

1. Verify platform matches your device
2. Check RKNN toolkit version compatibility
3. Ensure sufficient system memory
4. Try different core mask settings
5. Ensure proper input shape

Shape Mismatches
~~~~~~~~~~~~~~~~

If you get shape mismatch errors, check:

1. Input batch size doesn't exceed the model's configured batch size
2. Sequence length doesn't exceed max_seq_length
3. Input tensors have correct dimensions

.. code-block:: python

   # Check model configuration
   print(f"Batch size: {model.batch_size}")
   print(f"Max seq length: {model.max_seq_length}")
   print(f"Input names: {model.input_names}")
