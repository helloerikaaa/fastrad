Frequently Asked Questions (FAQ)
================================

Feature Extraction: Input, Customization, and Reproducibility
-------------------------------------------------------------

**How does fastrad compare to PyRadiomics mathematically?**
`fastrad` implements a fully vectorized PyTorch-based execution backend instead of scalar loops. However, the mathematical definitions of the features are perfectly matched. `fastrad` officially passes the **IBSI Phase 1 Reference Set** with numerical parity exactly mirroring `PyRadiomics` under a strict `1e-4` absolute tolerance bound. A paired Wilcoxon robust reproducibility test confirms their output distributions are statistically identical (`p=0.4109`).

**Why am I receiving an "Anisotropic spacing" warning?**
Like `PyRadiomics`, `fastrad` advises evaluating spatial textural matrices (such as GLCM, GLRLM) across isotropic voxels (e.g., $1\\times1\\times1$ mm). If your volume possesses asymmetric dimensions (e.g., $0.5\\times0.5\\times3.0$ mm), the geometric shift definitions $\\delta$ inherent to IBSI definitions skew textural distance assumptions. While `fastrad` will compute the values, we recommend utilizing SimpleITK resampling pre-processing pipes before feeding arrays into `MedicalImage`.

Hardware and Errors
-------------------

**What is the difference between CPU and CUDA devices in Settings?**
Providing `device="cpu"` utilizes standard Python multi-threading across system architecture exactly as legacy radiomics libraries do, though significantly faster (approx. 2.6x-3.8x baseline speedup). Providing `device="cuda"` automatically moves all volume arrays onto your NVIDIA Graphics Processing Unit (GPU) Video RAM (VRAM), unlocking up to **25.0x speedups** by evaluating thousands of spatial bounds simultaneously.

**Why does fastrad use so much memory? (CUDA `OutOfMemoryError`)**
Because PyRadiomics processes sequentially (pixel by pixel or slide by slide), its memory footprint is extremely low and static. `fastrad` trades memory for speed: it generates dense mathematical matrices for the entire Region of Interest (ROI) instantaneously. 
If your targeted clinical mask is excessively massive (e.g., >80,000 positive voxels) and your GPU VRAM is restricted (<4GB), PyTorch may trigger an `OutOfMemoryError`. 
*Don't panic:* `fastrad` is designed to internally catch this physical memory failure, clear the graphical cache implicitly, and automatically re-route that specific feature extraction class onto the system CPU.

Common Exceptions
-----------------

**ValueError: "Mask contains no positive voxels."**
Your `fastrad.Mask` object evaluated entirely to `0`. Either the mask file represents a blank structural slice, or your input thresholding array mechanism failed to binarize target anatomies above numeric threshold `.0`.

**ValueError: "Image shape and mask shape do not match."**
The dimensions of your `MedicalImage.tensor` and `Mask.tensor` must be mathematically identical. Ensure both images originate from identical geometric spaces and standard DICOM dimensions.
