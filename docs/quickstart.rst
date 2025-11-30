Quick Start
===========

This guide will get you started with RK-Transformers in minutes.

Export a Model to RKNN
-----------------------

RK-Transformers CLI Help
~~~~~~~~~~~~~~~~~~~~~~~~

Display help message for the export command:

.. code-block:: bash

   rk-transformers-cli export -h


Basic Export (Float16)
~~~~~~~~~~~~~~~~~~~~~~~

Export a Sentence Transformer model from Hugging Face Hub:

.. code-block:: bash

   rk-transformers-cli export \
     --model sentence-transformers/all-MiniLM-L6-v2 \
     --platform rk3588 \
     --flash-attention \
     --optimization-level 3

Export with Quantization (INT8)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export with custom dataset for quantization:

.. code-block:: bash

   rk-transformers-cli export \
     --model sentence-transformers/all-MiniLM-L6-v2 \
     --platform rk3588 \
     --flash-attention \
     --quantize \
     --dtype w8a8 \
     --dataset sentence-transformers/natural-questions \
     --dataset-split train \
     --dataset-columns answer \
     --dataset-size 128 \
     --max-seq-length 128

Export Local ONNX Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   rk-transformers-cli export \
     --model ./my-model/model.onnx \
     --platform rk3588 \
     --flash-attention \
     --batch-size 4

Programmatic Export
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rktransformers.export import (
       OptimizationConfig,
       QuantizationConfig,
       RKNNConfig,
       export_rknn,
   )
 
   config = RKNNConfig(
       model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
       output_path="./my-exported-model",
       target_platform="rk3588",
       batch_size=1,
       max_seq_length=128,
       quantization=QuantizationConfig(
           quantized_dtype="w8a8",
           dataset_name="wikitext",
           dataset_size=100,
       ),
       optimization=OptimizationConfig(optimization_level=3),
   )
 
   export_rknn(config)

Run Inference
-------------

Using Sentence Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SentenceTransformer**

.. code-block:: python

   from rktransformers import RKSentenceTransformer

   model = RKSentenceTransformer(
       "rk-transformers/all-MiniLM-L6-v2",
       model_kwargs={
           "core_mask": "all",
       },
   )

   sentences = ["This is a test sentence", "Another example"]
   embeddings = model.encode(sentences)
   print(embeddings.shape)  # (2, 384)

   # Load specific quantized model file
   model = RKSentenceTransformer(
       "rk-transformers/all-MiniLM-L6-v2",
       model_kwargs={
           "file_name": "rknn/model_w8a8.rknn",
       },
   )

**CrossEncoder**

.. code-block:: python

   from rktransformers import RKCrossEncoder

   model = RKCrossEncoder(
       "rk-transformers/ms-marco-MiniLM-L12-v2",
       model_kwargs={"core_mask": "auto"},
   )

   pairs = [
       ["How old are you?", "What is your age?"],
       ["Hello world", "Hi there!"],
       ["What is RKNN?", "This is a test."],
   ]
   scores = model.predict(pairs)
   print(scores)

   query = "Hi there!"
   documents = [
       "What is going on?",
       "I am 25 years old.",
       "This is a test.",
       "RKNN is a neural network toolkit.",
   ]
   results = model.rank(query, documents)
   print(results)

Using RK-Transformers API
~~~~~~~~~~~~~~~~~~~~~~~~~~

See task-specific models and their usage in the API docs:
:ref:`task-specific-models`.

.. code-block:: python

   from transformers import AutoTokenizer
   from rktransformers import RKModelForFeatureExtraction

   # Load tokenizer and model
   tokenizer = AutoTokenizer.from_pretrained("rk-transformers/all-MiniLM-L6-v2")
   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       core_mask="auto"
   )

   # Tokenize and run inference
   inputs = tokenizer(
       ["Sample text for embedding"],
       padding="max_length",
       truncation=True,
       return_tensors="np",
   )

   outputs = model(**inputs)
   embeddings = outputs.last_hidden_state.mean(axis=1)  # Mean pooling
   print(embeddings.shape)  # (1, 384)

Using Transformers Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import pipeline
   from rktransformers import RKModelForMaskedLM

   # Load the RKNN model
   model = RKModelForMaskedLM.from_pretrained(
       "rk-transformers/bert-base-uncased",
       file_name="rknn/model_w8a8.rknn"
   )

   # Create a fill-mask pipeline with the RKNN-accelerated model
   fill_mask = pipeline(
       "fill-mask",
       model=model,
       tokenizer="rk-transformers/bert-base-uncased",
       framework="pt",  # required for RKNN
   )

   # Run inference
   results = fill_mask("Paris is the [MASK] of France.")
   print(results)

Next Steps
----------

- Read about :doc:`user_guide/export` for advanced export options
- Learn about :doc:`user_guide/npu_cores` configuration
- Explore the :doc:`api/index` for detailed API documentation
