Benchmarks
==========

The primary advantage of using `fastrad` over traditional CPU-bound libraries is performance. The repository contains a rigorous benchmarking suite `run_benchmark.py` that validates both computation time and numeric stability. 

`fastrad` accelerates intensive texture matrix constructions (e.g., GLCM, GLRLM, GLDM) dramatically, scaling exceptionally well on NVIDIA GPUs.

GPU Performance
---------------

Below is a direct measurement comparing a standard `PyRadiomics` extraction against `fastrad` utilizing an NVIDIA GPU. 

*(Metrics taken from a 512x512 volumetric TCIA matrix using a custom 15mm-radius spherical ROI mask, comparing native single-threaded PyRadiomics feature calculation against `fastrad` PyTorch CUDA offloading)*

.. list-table:: GPU Runtime Acceleration (TCIA 512x512 matrix)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Feature Class
     - PyRadiomics (CPU)
     - fastrad (CUDA)
     - Speedup
   * - Firstorder
     - 2.33s
     - 0.034s
     - **69x**
   * - Shape
     - 2.34s
     - 0.063s
     - **37x**
   * - GLCM
     - 2.29s
     - 0.073s
     - **31x**
   * - GLRLM
     - 2.29s
     - 0.078s
     - **29x**
   * - GLDM
     - 2.30s
     - 0.219s
     - **10x**
   * - NGTDM
     - 2.29s
     - 0.292s
     - **8x**
   * - **TOTAL (Excl. GLSZM)**
     - **13.83s**
     - **0.76s**
     - **~18x**

Optimizing GLSZM (cuCIM)
------------------------
By default, the *Gray Level Size Zone Matrix (GLSZM)* relies heavily on Connected-Component union-find algorithms that often struggle with atomic contention on GPU hardware.

`fastrad` is specifically architected with a hybrid bypass framework that evaluates your hardware configuration. If the pipeline detects a CUDA target, it will attempt to route the GLSZM generation uniquely through **RAPIDS cuCIM** (`cucim.core.operations.morphology.label`) for true heterogeneous acceleration, cleanly circumventing the tensor-loop bottleneck.

Stability Guarantee
-------------------
`fastrad` includes a feature reproducibility and stability analysis tool that performs rigorous affine transformation test-retests and applies simulated Gaussian signal noise.

Through this framework (`benchmarks/fastrad_stability_analysis.py`), `fastrad` proves perfectly equivalent perturbation stability relative to `PyRadiomics` under degraded imaging conditions, ensuring robust scientific reliability down to the decimal.
