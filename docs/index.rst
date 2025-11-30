RK-Transformers Documentation
===============================

**RK-Transformers** is a runtime library that seamlessly integrates Hugging Face ``transformers`` and ``sentence-transformers`` with Rockchip's RKNN Neural Processing Units (NPUs). It enables efficient and facile deployment of transformer models on edge devices powered by Rockchip SoCs (RK3588, RK3576, etc.).

.. image:: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E
   :target: https://huggingface.co/rk-transformers
   :alt: Hugging Face Models

.. image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
   :target: https://www.python.org/downloads/
   :alt: Python 3.10-3.12

.. image:: https://img.shields.io/pypi/v/rk-transformers
   :target: https://pypi.org/project/rk-transformers/
   :alt: PyPI Version

.. image:: https://img.shields.io/github/actions/workflow/status/emapco/rk-transformers/ci.yaml
   :target: https://github.com/emapco/rk-transformers/actions/workflows/ci.yaml
   :alt: CI Status

.. image:: https://img.shields.io/github/license/emapco/rk-transformers?logo=github
   :target: https://github.com/emapco/rk-transformers/blob/main/LICENSE
   :alt: License

Key Features
------------

Model Export & Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Automatic ONNX Export**: Converts Hugging Face models to ONNX with input detection
- **RKNN Optimization**: Exports to RKNN format with configurable optimization levels (0-3)
- **Quantization**: INT8 (w8a8) quantization with calibration dataset support
- **Push to Hub**: Direct integration with Hugging Face Hub for model versioning

High-Performance Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **NPU Acceleration**: Leverage Rockchip's hardware NPU for 10-20x speedup
- **Multi-Core Support**: Automatic core selection and load balancing across NPU cores
- **Memory Efficient**: Optimized for edge devices with limited RAM

Framework Integration
~~~~~~~~~~~~~~~~~~~~~~

- **Sentence Transformers**: Drop-in replacement with ``RKSentenceTransformer`` and ``RKCrossEncoder``
- **Transformers API**: Compatible with standard Hugging Face pipelines

Quick Links
-----------

.. toctree::
   :maxdepth: 1

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   development

Community & Support
-------------------

- **GitHub Repository**: `emapco/rk-transformers <https://github.com/emapco/rk-transformers>`_
- **Issue Tracker**: `Report a bug <https://github.com/emapco/rk-transformers/issues>`_
- **Hugging Face Hub**: `rk-transformers models <https://huggingface.co/rk-transformers>`_

License
-------

This project is licensed under the `Apache License 2.0 <https://github.com/emapco/rk-transformers/blob/main/LICENSE>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
