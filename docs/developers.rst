Developers Guide
================

Welcome to the `fastrad` backend architecture overview. This section details how you can contribute new features, integrate novel tensor operations, and scientifically validate against established benchmarks.

Architecture: PyTorch over SimpleITK
------------------------------------
The core mechanism separating `fastrad` from `PyRadiomics` lies in its abandonment of localized, Pythonic sequential arrays, instead leveraging `PyTorch` (`torch.Tensor`).

* All volumes are projected into 3D tensors natively formatted as standard IEEE Floats (`torch.float32`).
* Region of Interest (ROI) boundaries natively crop absolute 3D bounds rather than zeroing pixels.
* Operations must never attempt iteration (`for` loops over 3-dimensional indices). All spatial queries must mathematically collapse utilizing vector dot products, multi-dimensional array shifts (e.g., `torch.roll`), and histogram binnings (`torch.bincount`, `torch.unique`).

Adding the Baseline (Feature Classes)
-------------------------------------
Every feature module (e.g., `features/glcm.py`) must implement a `compute()` entry-point accepting `image_tensor`, `mask_tensor`, and `settings`. 

When introducing new textural derivations:
1. Initialize the respective spatial dependency matrix (e.g., NGTDM).
2. Utilize the robust, internal `get_binned_image()` logic for discrete gray-level compliance rather than writing implicit quantization algorithms array-side.
3. Construct mathematical aggregates utilizing PyTorch primitive operations to retain device agnosticism (operating equally over `.cpu()` and `.cuda()` tensors).

Scientific Parity Testing
-------------------------
We strictly enforce **Image Biomarker Standardisation Initiative (IBSI)** compliance on every PR.
We provide automated suite tools located within `benchmarks/run_numerical_parity.py`.

Before merging any internal mathematical manipulation, you must run:

.. code-block:: bash

    python benchmarks/run_numerical_parity.py

Output parity validates the identical nature of your extracted values mapped against legacy `PyRadiomics` execution results. A strict **Absolute Relative Deviation** cutoff of `1e-4` dictates PR rejection bounds mechanically.

Out-Of-Memory (OOM) GPU Catchers
--------------------------------
Because we rely on dense PyTorch generation rules, developers must securely ensure large tensor allocation schemas inside specific features don't crash the global Python script natively. Do not catch VRAM exceptions inherently within the feature calculation functions: `FeatureExtractor.extract()` centralizes all `OutOfMemoryError` handlers natively via a hardware step-down CPU re-routing logic.
