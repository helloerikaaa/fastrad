# fastrad
FastRadiomics (fastrad) is a GPU-accelerated Python library for calculating radiomics features from medical images. It aims to be a high-performance alternative to PyRadiomics, built from the ground up to achieve deep feature extraction parity with significantly faster execution times by utilizing PyTorch for tensor operations.

## Features
- **Full Parity with PyRadiomics:** Extracts the exact same features as the original PyRadiomics implementations.
- **Hardware Acceleration:** Native PyTorch backend seamlessly targets multi-core CPUs, CUDA-enabled GPUs, and Apple Silicon (MPS)* out of the box.
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
git clone https://github.com/organization/fastrad.git
cd fastrad

# Install standard CPU dependencies
pip install .

# For NVIDIA GPU hardware acceleration (CUDA)
pip install ".[cuda]"
```

## Basic Usage

`fastrad` employs a feature extractor interface similar to `PyRadiomics` but significantly optimized for volume operations.

### Quick Start
```python
from fastrad import MedicalImage, Mask, FeatureExtractor, FeatureSettings

# 1. Load an image (DICOM directory) and mask
image = MedicalImage("/path/to/dicom_dir")
mask = Mask("/path/to/mask.nrrd")

# 2. Configure extractor settings 
settings = FeatureSettings(
    feature_classes=["firstorder", "shape", "glcm"], # specify needed features
    bin_width=25.0,
    device="auto" # use "cuda", "cpu", or "auto"
)

# 3. Extract Features
extractor = FeatureExtractor(settings)
features = extractor.execute(image, mask)

for feature_name, value in features.items():
    print(f"{feature_name}: {value}")
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

## Benchmarks & Stability
The repository contains a benchmarking suite that validates computation time and numeric stability. `fastrad` accelerates intensive texture matrix (GLSZM, GLCM, GLDM) constructions dramatically scaling with image dimensions.

### Performance Sample
Comparing pure single-threaded CPU execution and CUDA acceleration mapped against PyRadiomics 3.0 on a 512x512 volumetric TCIA matrix (`benchmarks/run_benchmark.py`):

**System 1: Apple M3 Max (ARM)**
```text
--- PyRadiomics Benchmark ---
PyRadiomics TOTAL       : 17.0s

--- Fastrad Benchmark (CPU, 1 Thread) ---
Fastrad CPU (1t) TOTAL  : 4.8s (3.5x speedup)
```

**System 2: Intel Core i9 14th Gen & RTX 4070 Ti (x86/CUDA)**
```text
--- PyRadiomics Benchmark ---
PyRadiomics TOTAL       : 16.1s

--- Fastrad Benchmark (CPU, 1 Thread) ---
Fastrad CPU (1t) TOTAL  : 8.4s (1.9x speedup)

--- Fastrad Benchmark (CUDA GPU) ---
Fastrad GPU TOTAL       : 0.6s (25.7x speedup)
```

*(Fastrad inherently parallelizes operations via PyTorch yielding nearly 2x to 3.5x speedups even on a single CPU thread, and scales natively via CUDA processing for up to 25x+ improvements on compatible GPUs)*

It also includes a feature stability analysis tool that performs test-retest validation, simulating minor 3D affine transformations and Gaussian noise to prove equivalent stability parity with PyRadiomics under perturbed imaging conditions.

Run the benchmarks locally:

```bash
python benchmarks/run_benchmark.py
python benchmarks/fastrad_stability_analysis.py
```

## Running Tests
Tests assert extraction values strictly adhere to the baseline output created by PyRadiomics to ensure fidelity for scientific rigor, including compliance with the IBSI (Image Biomarker Standardisation Initiative) phantom benchmarks.

To run the test suite:
```bash
pip install ".[test]"
pytest
```
