# fastrad

[![PyPI Version](https://img.shields.io/pypi/v/fastrad.svg?t=1)](https://pypi.org/project/fastrad/)
[![GitHub Release](https://img.shields.io/github/v/release/helloerikaaa/fastrad.svg?t=1)](https://github.com/helloerikaaa/fastrad/releases)
[![Tests Status](https://github.com/helloerikaaa/fastrad/actions/workflows/python-app.yml/badge.svg?t=1)](https://github.com/helloerikaaa/fastrad/actions)
[![Documentation Status](https://readthedocs.org/projects/fastrad/badge/?version=latest&t=1)](https://fastrad.readthedocs.io/en/latest/)
[![Python Versions](https://img.shields.io/pypi/pyversions/fastrad.svg?t=1)](https://pypi.org/project/fastrad/)
[![License](https://img.shields.io/github/license/helloerikaaa/fastrad.svg?t=1)](https://github.com/helloerikaaa/fastrad/blob/main/LICENSE)

FastRadiomics (fastrad) is a GPU-accelerated Python library for calculating radiomics features from medical images. It aims to be a high-performance alternative to PyRadiomics, built from the ground up to achieve deep feature extraction parity with significantly faster execution times by utilizing PyTorch for tensor operations.

## Features
- **Full Parity with PyRadiomics:** Extracts the exact same features as the original PyRadiomics implementations.
- **Hardware Acceleration:** Native PyTorch backend seamlessly targets multi-core CPUs, CUDA-enabled GPUs, and Apple Silicon (MPS)* out of the box.
- **IBSI Compliant:** Passed standard Image Biomarker Standardisation Initiative (IBSI) digital phantom validation testing.
- **Memory vs Speed Trade-off:** By materializing full tensors throughout computation, `fastrad` fundamentally trades CPU memory footprint for significant reductions in execution time.
- **Supported Feature Classes:**
  - First Order
  - Shape (2D and 3D)
  - GLCM (Gray Level Co-occurrence Matrix)
  - GLRLM (Gray Level Run Length Matrix)
  - GLSZM (Gray Level Size Zone Matrix)
  - GLDM (Gray Level Dependence Matrix)
  - NGTDM (Neighbourhood Gray Tone Difference Matrix)
- **DICOM Handling:** Accepts DICOM paths out-of-the-box (powered by `pydicom` and `SimpleITK`).

*\* Note: Some PyTorch MPS operations do not natively support FP64 floating-point operations. Since statistical features require extremely high precision for parity, CUDA or CPU are the primary recommended execution targets.*

## Installation
You can install `fastrad` and its core dependencies easily:

```bash
# Clone the repository
git clone https://github.com/helloerikaaa/fastrad.git
cd fastrad

# Install standard CPU dependencies
pip install .

# For NVIDIA GPU hardware acceleration (CUDA)
pip install ".[cuda]"
```

## Basic Usage

`fastrad` employs a feature extractor interface similar to `PyRadiomics` but significantly optimized for volume operations.

### Quick Start
### Quick Start
```python
from fastrad import FeatureSettings, FeatureExtractor
import fastrad

# 1. Load and align an image and mask (automatically handles NIfTI/DICOM)
# Automatically validates physical geometry and crops the tensor using SimpleITK
img, mask = fastrad.load_and_align(
    image_path="/path/to/image.nii.gz",
    mask_path="/path/to/mask.nrrd",
    resample_spacing=(1.0, 1.0, 1.0), # Optional: force isotropic resampling natively!
    crop=True
)

# 2. [Optional] Apply built-in Mathematical Image Filters natively on PyTorch GPU
filter_params = {
    "Original": {},
    "LoG": {"sigma": [1.0, 3.0]}, # Laplacian of Gaussian strictly matched to PyRadiomics!
    "Square": {},
    "Logarithm": {}
}
filtered_images_dict = fastrad.apply_builtin_filters(img, filter_params)

# 3. Configure extractor settings 
settings = FeatureSettings(
    feature_classes=["firstorder", "shape", "glcm"], # specify needed features
    bin_width=25.0,
    device="auto" # use "cuda", "cpu", or "auto"
)

# 4. Extract Features across generated filters
extractor = FeatureExtractor(settings)

for filter_name, filtered_img in filtered_images_dict.items():
    print(f"--- Extracting {filter_name} ---")
    features = extractor.extract(filtered_img, mask)
    for feature_name, value in features.items():
        print(f"{filter_name}_{feature_name}: {value}")
```

### CPU vs CUDA Performance
Offloading to GPU using `fastrad` dramatically reduces Feature Extraction runtime latency compared to PyRadiomics on CPU. Utilizing `"cuda"` in your `FeatureSettings` initializes tensors on the GPU automatically.

```python
# Utilizing CUDA for hardware-accelerated processing
settings_gpu = FeatureSettings(
    feature_classes=["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
    bin_width=25.0,
    device="cuda"
)

extractor = FeatureExtractor(settings_gpu)
features = extractor.execute(image, mask)
```

### Advanced: JIT Acceleration via `torch.compile`
On PyTorch 2.0+, `fastrad` supports native kernel fusion to lower Python overhead and prevent traversing tensors repeatedly. This is particularly fast when running batch extraction over multiple images since the compiled core mathematical loops will be heavily optimized in C++.
Enable it via `compile=True`:
```python
settings_compiled = FeatureSettings(
    feature_classes=["glcm", "glrlm"],
    bin_width=25.0,
    device="cuda",
    compile=True,
    compile_mode="reduce-overhead" # PyTorch default compilation flag
)
```
*(Note: JIT acceleration inherently contains a cold-start compilation overhead on the first executed image, so it is best utilized when processing patient cohorts rather than single-image evaluation. Additionally, Mac Apple Silicon MPS compatibility is currently limited in PyTorch, so fallback behavior may occur).*

## Documentation & Examples
`fastrad` ships with an extensive set of clinical tutorials housed in the `examples/` directory of this repository. They demonstrate advanced applications using actual hospital dataset configurations:

- `01_basic_extraction.py`: Fundamental integration setup and extraction limits.
- `02_gpu_acceleration.py`: Native setup for deploying parallel operations upon NVidia devices.
- `03_batch_processing_tcia.py`: Writing loops evaluating clinical cohorts simultaneously.
- `04_advanced_configuration.py`: Managing geometric warping warnings, spacing overrides, and PyTorch dynamic memory error mitigations safely.

To read the robust framework User Guide, see the `docs/user_guide.rst` and `docs/learn.rst` modules.

## Benchmarks & Scientific Stability
The repository contains a fully automated benchmarking suite (`benchmarks/report_generator.py`) that evaluates computation time, memory efficiency, and numeric stability against PyRadiomics to produce the `fastrad_scientific_report.md`.

Highlights from the latest scientific validation (`fastrad_scientific_report.md`):
- **IBSI Compliance**: 100% compliant with the Phase 1 digital phantom (all absolute relative deviations ≤ 1e-13%).
- **Numerical Parity**: 100% of internal features match PyRadiomics within a `1e-4` tolerance on real TCIA clinical segmentation masks.
- **Runtime CPU Performance**: Up to **25.3x** single-threaded CPU speedup per feature class, and **4.31x** overall speedup compared to a multi-thread PyRadiomics execution.
- **GPU Acceleration**: Up to **25.0x** overall pipeline speedup (and **49x** for firstorder features) natively utilizing CUDA PyTorch tensor streams.
- **Memory Efficiency**: Utilizes full dense tensors rather than scalar loops, resulting in a higher peak RAM footprint on large clinical ROIs (requires ~7.6 GB for a 30mm radius mask) in exchange for parallel speed.
- **Stability**: Tested rigorously under perturbations and scan-rescan test-retest datasets, confirming statistically equivalent ICC variances with PyRadiomics (paired Wilcoxon p>0.40).
- **Robustness**: Provides graceful, PyRadiomics-parity handling of edge cases (Empty Masks, Single-Voxel masks).

Run the full scientific benchmark suite locally to regenerate the physical report:

```bash
python benchmarks/report_generator.py
```

## Running Tests
Tests assert extraction values strictly adhere to the baseline output created by PyRadiomics to ensure fidelity for scientific rigor, including compliance with the IBSI (Image Biomarker Standardisation Initiative) phantom benchmarks.

To run the test suite:
```bash
pip install ".[test]"
pytest
```
