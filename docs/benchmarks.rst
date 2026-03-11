Benchmarks
==========

The primary advantage of using `fastrad` over traditional CPU-bound libraries is performance. The repository contains a rigorous benchmarking suite `run_benchmark.py` that validates both computation time and numeric stability. 

`fastrad` accelerates intensive texture matrix constructions (e.g., GLCM, GLRLM, GLDM) dramatically, scaling exceptionally well on NVIDIA GPUs.

GPU Performance
---------------

Below is a direct measurement comparing a standard `PyRadiomics` extraction against `fastrad` utilizing an NVIDIA GPU (Intel i9 14th Gen, RTX 4070 Ti, 96GB RAM). 

*(Metrics taken from a 512x512 volumetric TCIA matrix using a custom 15mm-radius spherical ROI mask, comparing native single-threaded PyRadiomics feature calculation against `fastrad` PyTorch CUDA offloading)*

.. list-table:: GPU Runtime Acceleration (TCIA 512x512 matrix)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Feature Class
     - PyRadiomics (CPU)
     - fastrad (CUDA)
     - Speedup
   * - Firstorder
     - 2.29s
     - 0.034s
     - **68x**
   * - Shape
     - 2.33s
     - 0.063s
     - **37x**
   * - GLCM
     - 2.29s
     - 0.072s
     - **32x**
   * - GLRLM
     - 2.29s
     - 0.078s
     - **29x**
   * - GLSZM
     - 2.30s
     - 0.271s
     - **8.5x**
   * - GLDM
     - 2.29s
     - 0.051s
     - **45x**
   * - NGTDM
     - 2.29s
     - 0.058s
     - **40x**
   * - **TOTAL**
     - **16.09s**
     - **0.63s**
     - **~25.7x**

CPU Performance
---------------

Even without a dedicated GPU, `fastrad` natively optimizes matrix shifts and crops via tensor algorithms to cleanly exceed typical single-threaded PyRadiomics evaluation instances.

.. list-table:: CPU Runtime Acceleration (TCIA 512x512 matrix)
   :widths: 33 33 33
   :header-rows: 1

   * - Hardware Architecture
     - PyRadiomics (1t)
     - fastrad (1t)
   * - Apple M3 Max (ARM)
     - 17.0s
     - 4.8s (**3.5x**)
   * - Intel Core i9 14th Gen (x86)
     - 16.1s
     - 8.4s (**1.9x**)

Optimizing GLSZM (cuCIM)
------------------------
By default, the *Gray Level Size Zone Matrix (GLSZM)* relies heavily on Connected-Component union-find algorithms that often struggle with atomic contention on GPU hardware.

`fastrad` is specifically architected with a hybrid bypass framework that evaluates your hardware configuration. On CPU targets, `fastrad` utilizes a highly efficient bounding-box pre-crop strategy via `scipy.ndimage` to isolate gray level structures prior to connected-components labeling, resulting in extreme CPU processing speeds that comfortably outclass traditional scalar baselines. If the pipeline detects a CUDA target, it will attempt to route the GLSZM generation uniquely through **RAPIDS cuCIM** (`cucim.core.operations.morphology.label`) for true heterogeneous acceleration, cleanly circumventing the tensor-loop bottleneck.

Stability Guarantee
-------------------
`fastrad` includes a rigorous feature reproducibility and stability analysis utilizing the RIDER Lung CT scan-rescan pairs to compute Intraclass Correlation Coefficients (ICC) alongside physical tensor perturbations (Translation and Gaussian Noise).

Memory Footprint Optimization
-----------------------------
Because of its dense tensor streaming architecture rather than scalar iterations, `fastrad` fundamentally eliminates massive sequence allocation limits native to PyRadiomics routines mapping on large ROIs. At an ROI diameter of 30mm (67k voxels), `fastrad` operates using 90% less peak RAM footprint dynamically allocating sparse equivalent matrices.

For full rigorous automated metrics across our 6-pillar validation setup, refer to the dynamically generated `fastrad_scientific_report.md` at the root of the repository.
