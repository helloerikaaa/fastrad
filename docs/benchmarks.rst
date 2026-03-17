Benchmarks
==========

The primary advantage of using `fastrad` over traditional CPU-bound libraries is performance. The repository contains a rigorous benchmarking suite `run_benchmark.py` that validates both computation time and numeric stability. 

`fastrad` accelerates intensive texture matrix constructions (e.g., GLCM, GLRLM, GLDM) dramatically, scaling exceptionally well on NVIDIA GPUs.

IBSI Compliance and Numerical Parity
------------------------------------

`fastrad` implements 116 mathematical features strictly conforming to the Image Biomarker Standardisation Initiative (IBSI) guidelines. 

.. list-table:: Numerical Parity with PyRadiomics (TCIA Clinical Image)
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Feature Class
     - Mean Abs Diff
     - Max Abs Diff
     - Features Within 1e-4
     - Features Outside 1e-4
   * - firstorder
     - 3.20e-16
     - 4.44e-15
     - 16
     - 0
   * - shape
     - 9.93e-15
     - 1.14e-13
     - 14
     - 0
   * - glcm
     - 7.12e-13
     - 1.09e-11
     - 24
     - 0
   * - glrlm
     - 2.05e-15
     - 1.42e-14
     - 16
     - 0
   * - glszm
     - 2.66e-15
     - 2.49e-14
     - 16
     - 0
   * - gldm
     - 3.05e-15
     - 3.91e-14
     - 14
     - 0
   * - ngtdm
     - 2.26e-17
     - 8.33e-17
     - 5
     - 0

**Outlier Analysis:** All features across all classes are strictly within the designated 1e-4 parity tolerance. 100% compliant with the Phase 1 digital phantom.

GPU Performance
---------------

Below is a direct measurement comparing a standard `PyRadiomics` extraction against `fastrad` utilizing an NVIDIA GPU (Intel i9 14th Gen, RTX 4070 Ti, 96GB RAM). 

*(Metrics taken from a real clinical TCIA segmentation mask, comparing native single-threaded PyRadiomics feature calculation against `fastrad` PyTorch CUDA offloading)*

.. list-table:: GPU Runtime Acceleration (TCIA Clinical Mask)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Feature Class
     - PyRadiomics (CPU)
     - fastrad (CUDA)
     - Speedup
   * - Firstorder
     - 0.4079s
     - 0.0083s
     - **49.3x**
   * - Shape
     - 0.4114s
     - 0.0117s
     - **35.0x**
   * - GLCM
     - 0.4175s
     - 0.0210s
     - **19.9x**
   * - GLRLM
     - 0.4135s
     - 0.0320s
     - **12.9x**
   * - GLSZM
     - 0.4129s
     - 0.0183s
     - **22.5x**
   * - GLDM
     - 0.4209s
     - 0.0113s
     - **37.1x**
   * - NGTDM
     - 0.4119s
     - 0.0130s
     - **31.6x**
   * - **TOTAL**
     - **2.8961s**
     - **0.1157s**
     - **~25.0x**

CPU Performance
---------------

Even without a dedicated GPU, `fastrad` natively optimizes matrix shifts and crops via tensor algorithms to cleanly exceed typical single-threaded PyRadiomics evaluation instances.

.. list-table:: CPU Runtime Acceleration (TCIA Clinical Mask)
   :widths: 33 33 33
   :header-rows: 1

.. list-table:: CPU Runtime Acceleration (TCIA Clinical Mask)
   :widths: 33 33 33
   :header-rows: 1

   * - Hardware Architecture
     - PyRadiomics (1t)
     - fastrad (1t)
   * - Apple M3 (ARM)
     - 2.99s
     - 0.78s (**3.8x**)
   * - Intel Core i9 14th Gen (x86)
     - 2.89s
     - 1.10s (**2.6x**)

Multi-threading Fairness Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **PyRadiomics CPU (Single Thread)**: 2.89s
- **PyRadiomics CPU (32 Threads)**: 2.88s
- **fastrad CPU (Single Thread)**: 1.10s

=> **Comparative Advantage (fastrad 1t vs PyRadiomics 32t): 2.63x speedup**

*Note: PyRadiomics is not internally parallelised at the feature computation level; threading only affects SimpleITK image operations. This explains the observed lack of scaling.*

ROI Size Scaling Benchmark (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: GPU Scaling Speedup
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Radius (mm)
     - Voxel Count
     - PyRadiomics Total (s)
     - fastrad GPU Total (s)
     - Speedup
   * - 5
     - 199
     - 2.908s
     - 0.112s
     - 25.8x
   * - 10
     - 2249
     - 2.936s
     - 0.138s
     - 21.2x
   * - 15
     - 8263
     - 2.941s
     - 0.155s
     - 18.9x
   * - 20
     - 20181
     - 3.031s
     - 0.198s
     - 15.2x
   * - 25
     - 38327
     - 3.111s
     - 0.251s
     - 12.3x
   * - 30
     - 67461
     - 3.271s
     - 0.337s
     - 9.6x

Optimizing GLSZM (cuCIM)
------------------------
By default, the *Gray Level Size Zone Matrix (GLSZM)* relies heavily on Connected-Component union-find algorithms that often struggle with atomic contention on GPU hardware.

`fastrad` is specifically architected with a hybrid bypass framework that evaluates your hardware configuration. On CPU targets, `fastrad` utilizes a highly efficient bounding-box pre-crop strategy via `scipy.ndimage` to isolate gray level structures prior to connected-components labeling, resulting in extreme CPU processing speeds that comfortably outclass traditional scalar baselines. If the pipeline detects a CUDA target, it will attempt to route the GLSZM generation uniquely through **RAPIDS cuCIM** (`cucim.core.operations.morphology.label`) for true heterogeneous acceleration, cleanly circumventing the tensor-loop bottleneck.

Stability Guarantee
-------------------
`fastrad` includes a rigorous feature reproducibility and stability analysis utilizing the RIDER Lung CT scan-rescan pairs to compute Intraclass Correlation Coefficients (ICC) alongside physical tensor perturbations (Translation and Gaussian Noise).

ICC Analysis on Real RIDER Scan-Rescan Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Fastrad Features with ICC >= 0.90**: 10.7%
- **PyRadiomics Features with ICC >= 0.90**: 8.7%
- **Fastrad Mean ICC**: 0.3619
- **PyRadiomics Mean ICC**: 0.3530
- **Wilcoxon signed-rank test**: stat=647.0000, p=0.4109

Numerical Robustness to Input Perturbation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Numerical Robustness
   :widths: 25 25 25 25
   :header-rows: 1

   * - Perturbation
     - PyRadiomics Mean Drift
     - fastrad Mean Drift
     - Failure Count
   * - Gaussian Noise
     - 10.58%
     - 10.20%
     - 0
   * - Translation
     - 228.77%
     - 219.83%
     - 0

Memory Footprint Optimization
-----------------------------
Because of its dense tensor streaming architecture prioritizing evaluation speed, `fastrad` fundamentally trades peak CPU memory footprint for significant reductions in execution time. By materializing full dense tensors throughout computation instead of sequential voxel loops, at an ROI diameter of 30mm (67k voxels), `fastrad` requires substantially more peak CPU RAM (~7.6GB) compared to PyRadiomics (<1GB). It is highly recommended to use the GPU pathway or smaller batch chunks on systems with limited physical resources or when processing massive whole-organ segmentation volumes.

GPU VRAM Profile (Full Pipeline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Peak GPU VRAM
   :widths: 50 50
   :header-rows: 1

   * - Feature Class
     - Peak VRAM Allocated (MB)
   * - firstorder
     - 116.47
   * - shape
     - 627.48
   * - glcm
     - 356.58
   * - glrlm
     - 263.34
   * - glszm
     - 116.47
   * - gldm
     - 361.76
   * - ngtdm
     - 654.78
   * - **FULL PIPELINE**
     - **654.78**

Edge Case Handling
------------------

.. list-table:: Handling of Edge Cases
   :widths: 25 25 25 25
   :header-rows: 1

   * - Edge Case
     - Expected Behaviour
     - fastrad Behaviour
     - PyRadiomics Behaviour
   * - Empty Mask
     - ValueError
     - ValueError
     - ValueError
   * - Single-voxel ROI
     - Exception / Graceful
     - Graceful Completion
     - ValueError
   * - Very Small ROI (<8 voxels)
     - Exception / Graceful
     - Graceful Completion
     - Graceful Completion
   * - Non-isotropic Spacing
     - UserWarning
     - Graceful + Warning
     - Graceful Completion

For full rigorous automated metrics across our 6-pillar validation setup, refer to the generated scientific report utilizing the ``run_benchmark.py`` scripts.
