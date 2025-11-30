Modeling
========

Model classes for running RKNN inference.

Base Classes
------------

.. autoclass:: rktransformers.modeling.RKNNRuntime
   :members:
   :show-inheritance:

.. autoclass:: rktransformers.modeling.RKModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

.. _task-specific-models:

Task-Specific Models
--------------------

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. autoclass:: rktransformers.modeling.RKModelForFeatureExtraction
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: model_type, auto_model_class

Sequence Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rktransformers.modeling.RKModelForSequenceClassification
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: model_type, auto_model_class

Token Classification
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rktransformers.modeling.RKModelForTokenClassification
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: model_type, auto_model_class

Question Answering
~~~~~~~~~~~~~~~~~~

.. autoclass:: rktransformers.modeling.RKModelForQuestionAnswering
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: model_type, auto_model_class

Masked Language Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rktransformers.modeling.RKModelForMaskedLM
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: model_type, auto_model_class

Multiple Choice
~~~~~~~~~~~~~~~

.. autoclass:: rktransformers.modeling.RKModelForMultipleChoice
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: model_type, auto_model_class
