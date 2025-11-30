Local Development
=================

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/emapco/rk-transformers.git
   cd rk-transformers
   uv venv
   uv pip install -e .[dev,export]
   # workaround for rknn-toolkit2 dependency
   uv pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
   pre-commit install

Running Tests
-------------

.. code-block:: bash

   # Run all tests (excludes manual tests)
   make test

   # Run with coverage report
   make test-cov

   # Run specific test categories
   pytest -m integration tests -v          # Integration tests only
   pytest -m "not slow" tests -v           # Skip slow tests
   pytest -m requires_rknpu tests -v        # Tests requiring Rockchip hardware

Linting and Formatting
----------------------

.. code-block:: bash

   # Auto-fix linting issues and format code
   make lint

   # Run pre-commit hooks manually
   pre-commit run --all-files

Environment Diagnostics
-----------------------

Check your Rockchip environment and library versions:

.. code-block:: bash

   rk-transformers-cli env

Output example:

.. code-block:: text

   Copy-and-paste the text below in your GitHub issue:

   - Operating system: Linux-5.10.160-rockchip-rk3588
   - Rockchip Board: Orange Pi 5 Plus
   - Rockchip SoC: rk3588
   - RKNN Runtime version: 2.3.2
   - RKNN Toolkit version: rknn-toolkit-lite2==2.3.2
   - Python version: 3.12.9
   - PyTorch version: 2.6.0+cpu
   - HuggingFace transformers version: 4.55.4
   - HuggingFace optimum version: 2.0.0
   - rk-transformers version: 0.1.0
