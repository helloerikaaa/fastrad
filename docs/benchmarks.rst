Benchmarks
==========

The primary advantage of using `fastrad` over traditional CPU-bound libraries is performance. The repository contains a rigorous benchmarking suite `run_benchmark.py` that validates both computation time and numeric stability. 

`fastrad` accelerates intensive texture matrix constructions (e.g., GLCM, GLRLM, GLDM) dramatically, scaling exceptionally well on NVIDIA GPUs.

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

   * - Hardware Architecture
     - PyRadiomics (1t)
     - fastrad (1t)
   * - Apple M3 (ARM)
     - 2.99s
     - 0.78s (**3.8x**)
   * - Intel Core i9 14th Gen (x86)
     - 2.89s
     - 1.10s (**2.6x**)

Optimizing GLSZM (cuCIM)
------------------------
By default, the *Gray Level Size Zone Matrix (GLSZM)* relies heavily on Connected-Component union-find algorithms that often struggle with atomic contention on GPU hardware.

`fastrad` is specifically architected with a hybrid bypass framework that evaluates your hardware configuration. On CPU targets, `fastrad` utilizes a highly efficient bounding-box pre-crop strategy via `scipy.ndimage` to isolate gray level structures prior to connected-components labeling, resulting in extreme CPU processing speeds that comfortably outclass traditional scalar baselines. If the pipeline detects a CUDA target, it will attempt to route the GLSZM generation uniquely through **RAPIDS cuCIM** (`cucim.core.operations.morphology.label`) for true heterogeneous acceleration, cleanly circumventing the tensor-loop bottleneck.

Stability Guarantee
-------------------
`fastrad` includes a rigorous feature reproducibility and stability analysis utilizing the RIDER Lung CT scan-rescan pairs to compute Intraclass Correlation Coefficients (ICC) alongside physical tensor perturbations (Translation and Gaussian Noise).

Memory Footprint Optimization
-----------------------------
Because of its dense tensor streaming architecture prioritizing evaluation speed, `fastrad` fundamentally trades peak CPU memory footprint for significant reductions in execution time. By materializing full dense tensors throughout computation instead of sequential voxel loops, at an ROI diameter of 30mm (67k voxels), `fastrad` requires substantially more peak CPU RAM (~7.6GB) compared to PyRadiomics (<1GB). It is highly recommended to use the GPU pathway or smaller batch chunks on systems with limited physical resources or when processing massive whole-organ segmentation volumes.

For full rigorous automated metrics across our 6-pillar validation setup, refer to the dynamically generated `fastrad_scientific_report.md` at the root of the repository.
