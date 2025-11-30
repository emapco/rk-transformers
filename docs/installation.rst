Installation
============

Prerequisites
-------------

- Python 3.10 | 3.11 | 3.12
- Linux-based OS (Ubuntu 24.04+ recommended)
- For export: PC with x86_64/arm64 architecture
- For inference: Rockchip device with RKNPU2 support (RK3588, RK3576, etc.)

Quick Install
-------------

``uv`` is recommended for faster installation and smaller environment footprint.

For Inference (on Rockchip devices [arm64])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv venv
   uv pip install rk-transformers[inference]

This installs runtime dependencies including:

- ``rknn-toolkit-lite2`` (2.3.2)
- ``sentence-transformers`` (5.x)
- ``numpy``, ``torch``, ``transformers``

For Model Export (on development machines [x86_64, arm64])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv venv
   uv pip install rk-transformers[dev,export]
   # Workaround for rknn-toolkit2 dependency and RCE vulnerability in torch<=2.5.1
   uv pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

This installs export dependencies including:

- ``rknn-toolkit2`` (2.3.2)
- ``sentence-transformers`` (5.x)
- ``numpy``, ``torch``, ``transformers``, ``optimum[onnx]``, ``datasets``

For Development
~~~~~~~~~~~~~~~

See the development guide: :doc:`development`.

Using pip
~~~~~~~~~

If you prefer to use ``pip`` instead of ``uv``:

.. code-block:: bash

   # For inference
   pip install rk-transformers[inference]

   # For export
   pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu # Minimum for ARM64 rknn-toolkit2
   pip install rk-transformers[dev,export]
   pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu # RCE vulnerability torch<=2.5.1
