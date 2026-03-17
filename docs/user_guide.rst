User Guide
===========

The ``fastrad`` User Guide offers a comprehensive deep-dive into the architectural mechanics, standard objects, and configuration logic necessary for integrating the library inside clinical or research analysis ecosystems. 

Core Philosophy: PyTorch over Sequences
---------------------------------------
Traditional libraries execute complex iteration loops over 3-dimensional volumes, resulting in linear compute bottlenecks. We bypass these limitations by mathematically transforming radiomics feature extraction into highly parallelizable physical matrix algebra inherently supported via the PyTorch backend. 

* The library loads the image volume onto targeted device boundaries (CPU multi-host RAM or native CUDA VRAM structures).
* Regions of Interest (ROIs) are strictly cropped dynamically bounding the target anatomical location.
* Mathematical convolutions, histograms, and shifted tensor alignments compute statistical aggregations nearly instantly.

Object Orientated Structure
---------------------------
The framework utilizes three primary classes mirroring the legacy architectures.

1. **MedicalImage**
   The foundation storing spatial dimensions natively. Automatically constructed either via standard external packages loading array pointers, or natively utilizing absolute paths to DICOM directories natively inside `MedicalImage.from_dicom(path)`.

2. **Mask**
   Binary 3D Region of Interest logic. Explicitly validates that the mask internal shapes bounds stringently align with the target `MedicalImage` to ensure mathematically reliable operations. Values mathematically >0 evaluate dynamically to an absolute internal boolean logic mask.

3. **FeatureSettings**
   The configuration blueprint initializing the extractor logic. 
   Supported settings:
   - ``feature_classes`` (List[str]): Array mapping to strictly compute elements (default handles all 7 supported texture models).
   - ``bin_width`` (float): The default value dictating geometric numerical distributions explicitly mandated per the IBSI guidelines (e.g., typically `25.0` for clinical generic CT operations).
   - ``device`` (str): "cpu", "cuda", or "auto". Dynamic detection handling internal routing logic.

Extraction Workflow Execution
-----------------------------
``FeatureExtractor.extract()`` evaluates the given configuration safely against provided hardware parameters. Edge cases are robustly anticipated to prevent silent numerical contamination operations:

* **Empty Masks:** Raising absolute ValueError constraints prior to loading array geometries securely bounds invalid pipeline evaluations.
* **Isotropic Distortions:** `fastrad` warns users actively performing 3D textural queries when spacing boundaries indicate heavy geometric warping factors inherently undermining GLCM shift interpretations without crashing generic script models organically.
* **VRAM Fallbacks:** Safely deploying against clinical workstations requires respecting local resource limitations. Should the target matrix allocations inherently exceed available absolute Nvidia CUDA graphical budgets, the extractor securely captures internal `torch.cuda.OutOfMemoryError` signals explicitly allocating dynamic offloads directly executing utilizing standard multi-threaded CPU environments for the offending module solely while retaining system stability parameters dynamically.
