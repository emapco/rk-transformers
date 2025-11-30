NPU Core Configuration
======================

Rockchip SoCs with multiple NPU cores support flexible core allocation strategies through the ``core_mask`` parameter. Choosing the right core mask can optimize performance based on your workload and system conditions.

.. note::
   ``core_mask`` is specified at inference time, not during export.

Available Core Masks
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Value
     - Description
     - Use Case
   * - ``"auto"``
     - Automatic mode - selects idle cores dynamically
     - **Recommended**: Best for most scenarios
   * - ``"0"``
     - NPU Core 0 only
     - Fixed core assignment
   * - ``"1"``
     - NPU Core 1 only
     - Fixed core assignment
   * - ``"2"``
     - NPU Core 2 only
     - Fixed core assignment (RK3588 only)
   * - ``"0_1"``
     - NPU Core 0 and 1 simultaneously
     - Parallel execution across 2 cores for larger models
   * - ``"0_1_2"``
     - NPU Core 0, 1, and 2 simultaneously
     - Maximum parallelism (RK3588 only) for demanding models
   * - ``"all"``
     - All available NPU cores
     - Equivalent to ``"0_1_2"`` on RK3588, ``"0_1"`` on RK3576

Platform-Specific Notes
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Platform
     - Available Cores
     - Recommended Default
   * - **RK3588**
     - 0, 1, 2 (3 cores)
     - ``"auto"`` or ``"0_1_2"`` for large models
   * - **RK3576**
     - 0, 1 (2 cores)
     - ``"auto"`` or ``"0_1"`` for large models
   * - **RK3566/RK3568**
     - 0 (1 core)
     - ``"0"`` (only option)

.. warning::
   Attempting to use unavailable cores (e.g., ``"2"`` on RK3576) may result in a runtime error.

Usage Examples
--------------

RK-Transformers API
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rktransformers import RKModelForFeatureExtraction

   model = RKModelForFeatureExtraction.from_pretrained(
       "rk-transformers/all-MiniLM-L6-v2",
       core_mask="all"
   )

Sentence Transformers Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rktransformers import RKSentenceTransformer

   model = RKSentenceTransformer(
       "rk-transformers/all-MiniLM-L6-v2",
       model_kwargs={
           "platform": "rk3588",
           "core_mask": "auto",
       },
   )

CrossEncoder Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rktransformers import RKCrossEncoder

   model = RKCrossEncoder(
       "rk-transformers/ms-marco-MiniLM-L12-v2",
       model_kwargs={
           "platform": "rk3588",
           "core_mask": "auto",
       },
   )

Performance Considerations
--------------------------

Single Core vs Multi-Core
~~~~~~~~~~~~~~~~~~~~~~~~~

**Single Core** (``"0"``, ``"1"``, ``"2"``):

- Lower power consumption
- Predictable latency
- Good for lightweight models
- Useful when cores are allocated to different tasks

**Multi-Core** (``"0_1"``, ``"0_1_2"``, ``"all"``):

- Higher throughput
- Better for large models
- Potentially higher latency due to synchronization
- Higher power consumption

Auto vs Manual Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

**Auto Mode** (``"auto"``):

- Pros:
  - RKNN runtime provides automatic load balancing
  - Adapts to system load
  - No manual tuning needed
  
- Cons:
  - Less predictable core assignment
  - May not be optimal for all scenarios

**Manual Mode** (specific cores):

- Pros:
  - Predictable behavior
  - Fine-grained control
  - Better for multi-model/multi-instance deployment
  
- Cons:
  - Requires manual tuning
  - May not adapt to changing conditions

Best Practices
--------------

1. **Start with "auto"**: Begin with automatic mode and measure performance
2. **Benchmark different configurations**: Test various core masks for your specific workload
3. **Consider power constraints**: Use fewer cores if power consumption is a concern
4. **Monitor core utilization**: Check which cores are busy before manual assignment. `ajokela/rktop <https://github.com/ajokela/rktop>`_ can help monitor NPU core usage.

Multi-Model Deployment
~~~~~~~~~~~~~~~~~~~~~~

When running multiple models simultaneously:

.. code-block:: python

   # Model 1 on core 0
   model1 = RKModelForFeatureExtraction.from_pretrained(
       "model1",
       core_mask="0"
   )

   # Model 2 on core 1
   model2 = RKModelForSequenceClassification.from_pretrained(
       "model2",
       core_mask="1"
   )

   # Model 3 on core 2
   model3 = RKModelForMultipleChoice.from_pretrained(
       "model3",
       core_mask="2"
   )

Troubleshooting
---------------

Performance Issues
~~~~~~~~~~~~~~~~~~

If performance is not as expected:

1. Try different core mask configurations
2. Ensure no other NPU-intensive tasks are running
3. Check CPU and memory usage (may be bottlenecks)
4. Verify model is properly quantized for the platform
5. Monitor NPU temperature (thermal throttling may occur)
