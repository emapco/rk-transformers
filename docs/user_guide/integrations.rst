Framework Integrations
======================

RK-Transformers provides seamless integration with popular transformer frameworks.

.. _sentence-transformers:

Sentence Transformers
---------------------

RK-Transformers includes drop-in replacements for Sentence Transformers classes that use RKNN acceleration.

.. _rk-sentence-transformer:

RKSentenceTransformer
~~~~~~~~~~~~~~~~~~~~~

Replace ``SentenceTransformer`` with :class:`~rktransformers.integrations.sentence_transformers.RKSentenceTransformer` to use RKNN-accelerated models:

.. code-block:: python

   from rktransformers import RKSentenceTransformer

   # Load model from Hugging Face Hub
   model = RKSentenceTransformer(
       "rk-transformers/all-MiniLM-L6-v2",
       model_kwargs={
           "platform": "rk3588",
           "core_mask": "auto",
       },
   )

   # Encode sentences
   sentences = ["This is a test sentence", "Another example"]
   embeddings = model.encode(sentences)
   print(embeddings.shape)  # (2, 384)

   # Similarity computation
   similarity = model.similarity(embeddings[0], embeddings[1])
   print(similarity)

Supported Methods
^^^^^^^^^^^^^^^^^

- ``encode()``: Generate embeddings
- ``similarity()``: Compute similarity between embeddings

.. _rk-cross-encoder:

RKCrossEncoder
~~~~~~~~~~~~~~

Replace ``CrossEncoder`` with :class:`~rktransformers.integrations.sentence_transformers.RKCrossEncoder` for RKNN-accelerated cross-encoding:

.. code-block:: python

   from rktransformers import RKCrossEncoder
   # or if already using CrossEncoder:
   # from rktransformers import RKCrossEncoder as CrossEncoder

   model = RKCrossEncoder(
       "rk-transformers/ms-marco-MiniLM-L12-v2",
       model_kwargs={"platform": "rk3588"},
   )

   # Predict scores for sentence pairs
   pairs = [
       ["How old are you?", "What is your age?"],
       ["Hello world", "Hi there!"],
   ]
   scores = model.predict(pairs)
   print(scores)

   # Rank documents for a query
   query = "What is RKNN?"
   documents = [
       "RKNN is a neural network toolkit.",
       "This is unrelated text.",
       "Rockchip NPU acceleration.",
   ]
   results = model.rank(query, documents)
   for result in results:
       print(f"Score: {result['score']:.4f} - {result['corpus_id']}: {documents[result['corpus_id']]}")

Supported Methods
^^^^^^^^^^^^^^^^^

- ``predict()``: Score sentence pairs
- ``rank()``: Rank documents for a query

Hugging Face Transformers
--------------------------

RK-Transformers model classes are compatible with Hugging Face Transformers pipelines and APIs.

Using Pipelines
~~~~~~~~~~~~~~~

.. code-block:: python

   from transformers import pipeline
   from rktransformers import RKModelForMaskedLM

   # Load RKNN model
   model = RKModelForMaskedLM.from_pretrained(
       "rk-transformers/bert-base-uncased",
       core_mask="auto",
   )

   # Create pipeline
   fill_mask = pipeline(
       "fill-mask",
       model=model,
       tokenizer="rk-transformers/bert-base-uncased",
       framework="pt",
   )

   # Run inference
   results = fill_mask("Paris is the [MASK] of France.")
   print(results)

Supported Pipelines
^^^^^^^^^^^^^^^^^^^

- ``feature-extraction``: Extract embeddings
- ``fill-mask``: Masked language modeling
- ``text-classification``: Sequence classification
- ``token-classification``: Named entity recognition, POS tagging
- ``question-answering``: Extractive question answering

RK-Transformers Model Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use RK models like standard Transformers models:

.. code-block:: python

   from transformers import AutoTokenizer
   from rktransformers import RKModelForSequenceClassification

   tokenizer = AutoTokenizer.from_pretrained("rk-transformers/bert-base-uncased")
   model = RKModelForSequenceClassification.from_pretrained(
       "rk-transformers/bert-base-uncased"
   )

   inputs = tokenizer("This movie is great!", return_tensors="np")
   outputs = model(**inputs)
   
   # Get predictions
   import numpy as np
   predictions = np.argmax(outputs.logits, axis=-1)
   print(f"Predicted class: {predictions}")

AutoModel Support
^^^^^^^^^^^^^^^^^

RK models register with AutoModel classes:

.. code-block:: python

   from transformers import AutoConfig
   from rktransformers import RKModelForFeatureExtraction

   # AutoModel.from_pretrained will use the registered RKModel class
   config = AutoConfig.from_pretrained("rk-transformers/distilbert-base-cased-distilled-squad")

Custom Integrations
-------------------

Building Custom Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~

You can build custom integrations by composing RK model classes:

.. code-block:: python

   import numpy as np
   from transformers import AutoTokenizer

   from rktransformers import RKModelForFeatureExtraction
   from rktransformers.constants import CoreMaskType


   class CustomEmbeddingModel:
      def __init__(self, model_id, core_mask: CoreMaskType = "all"):
         self.model = RKModelForFeatureExtraction.from_pretrained(model_id, core_mask=core_mask)
         self.tokenizer = AutoTokenizer.from_pretrained(model_id)

      def embed(self, texts: list[str]) -> np.ndarray:
         """Embed a list of texts using the model batch size and masked mean pooling.

         The function batches the input texts using the model's configured `batch_size`,
         tokenizes each batch, runs inference, and applies masked mean pooling to produce
         one vector per input text.
         """
         if not texts:
               # return an empty array with appropriate hidden size if possible
               hidden_size = getattr(self.model.config, "hidden_size", None)
               if hidden_size is None:
                  return np.empty((0, 0))
               return np.empty((0, hidden_size))

         model_batch_size = getattr(self.model, "batch_size", 1) or 1
         embeddings_batches: list[np.ndarray] = []

         for i in range(0, len(texts), model_batch_size):
               batch_texts = texts[i : i + model_batch_size]
               max_seq_length = getattr(self.model, "max_seq_length", 512)
               inputs = self.tokenizer(
                  batch_texts,
                  padding="max_length",
                  truncation=True,
                  max_length=max_seq_length,
                  return_tensors="np",
               )
               outputs = self.model(**inputs)
               last_hidden_state = np.asarray(outputs.last_hidden_state)

               # Masked mean pooling
               _tmp_mask = inputs.get("attention_mask")
               attention_mask = np.asarray(_tmp_mask) if _tmp_mask is not None else None
               if attention_mask is None:
                  # fallback to simple mean over time dimension
                  batch_embeddings = np.mean(last_hidden_state, axis=1)
               else:
                  mask_expanded = np.expand_dims(attention_mask, -1)
                  sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
                  sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                  batch_embeddings = sum_embeddings / sum_mask

               embeddings_batches.append(batch_embeddings)

         return np.vstack(embeddings_batches)

   # Usage
   model = CustomEmbeddingModel("rk-transformers/all-MiniLM-L6-v2")
   embeddings = model.embed(["Hello world", "Test sentence"])
   print(embeddings.shape)

Integration with Other Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**LangChain Integration** (example):

.. code-block:: python

   from langchain.embeddings.base import Embeddings

   from rktransformers import RKSentenceTransformer


   class RKNNEmbeddings(Embeddings):
      def __init__(self, model_name, core_mask="all"):
         self.model = RKSentenceTransformer(model_name, model_kwargs={"core_mask": core_mask})

      def embed_documents(self, texts):
         return self.model.encode(texts).tolist()

      def embed_query(self, text):
         return self.model.encode([text])[0].tolist()

   # Usage with LangChain
   embeddings = RKNNEmbeddings("rk-transformers/all-MiniLM-L6-v2")
   embedded = embeddings.embed_query("Hello, world!")
   print(len(embedded))  # 384

   doc_embeddings = embeddings.embed_documents(["Hello world", "Test sentence"])
   print(len(doc_embeddings))  # 2

Migration Guide
---------------

From Sentence Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before** (CPU/CUDA):

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   embeddings = model.encode(["Hello world"])

**After** (RKNN):

.. code-block:: python

   from rktransformers import RKSentenceTransformer
   
   model = RKSentenceTransformer(
       "rk-transformers/all-MiniLM-L6-v2",
       model_kwargs={"core_mask": "auto"}
   )
   embeddings = model.encode(["Hello world"])

From Transformers
~~~~~~~~~~~~~~~~~

**Before** (CPU/CUDA):

.. code-block:: python

   from transformers import AutoModelForMaskedLM, AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
   model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

   inputs = tokenizer("Hello world", return_tensors="pt")
   outputs = model(**inputs)

**After** (RKNN):

.. code-block:: python

   from transformers import AutoTokenizer
   from rktransformers import RKModelForMaskedLM
   
   tokenizer = AutoTokenizer.from_pretrained("rk-transformers/bert-base-uncased")
   model = RKModelForMaskedLM.from_pretrained(
       "rk-transformers/bert-base-uncased",
       model_kwargs={"core_mask": "auto"}
   )

   inputs = tokenizer("Hello world", return_tensors="np")  # Use "np" instead of "pt"
   outputs = model(**inputs)
