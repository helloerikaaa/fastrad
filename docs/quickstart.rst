Quickstart
==========

`fastrad` is designed to be highly ergonomic, employing a modular API similar in spirit to traditional radiomics packages but architected for volume operations.

Basic Usage (CPU)
-----------------

Here is a complete end-to-end example of loading a DICOM volume, a segmentation mask, and extracting features.

.. code-block:: python

   from fastrad import MedicalImage, Mask, FeatureExtractor, FeatureSettings

   # 1. Load an image (e.g., a DICOM directory) and mask (e.g., NRRD file)
   image = MedicalImage.from_dicom("/path/to/dicom_dir")
   mask = Mask.from_dicom("/path/to/mask_dir") # or load a NIFTI/NRRD

   # 2. Configure extractor settings 
   settings = FeatureSettings(
       feature_classes=["firstorder", "shape", "glcm"], # specify needed features
       bin_width=25.0,
       device="cpu" # execution target
   )

   # 3. Extract Features
   extractor = FeatureExtractor(settings)
   features = extractor.extract(image, mask)

   # 4. View results
   for feature_name, value in features.items():
       print(f"{feature_name}: {value}")

GPU Acceleration
----------------

Offloading to the GPU dramatically reduces Feature Extraction runtime latency. By simply changing the ``device`` parameter to ``"cuda"``, `fastrad` automatically initializes and routes all tensor operations, matrix constructions, and aggregations directly on the GPU.

.. code-block:: python

   # Utilizing CUDA for hardware-accelerated processing
   settings_gpu = FeatureSettings(
       feature_classes=[
           "firstorder", "shape", "glcm", "glrlm", 
           "glszm", "gldm", "ngtdm"
       ],
       bin_width=25.0,
       device="cuda"
   )

   extractor = FeatureExtractor(settings_gpu)
   # The extractor handles tensor movement and OOM fallbacks automatically
   features = extractor.extract(image, mask)

If a specific feature computation exceeds the VRAM available on your GPU, `fastrad` handles `OutOfMemoryError` gracefully, falls back to CPU computation for that specific feature class, clears the CUDA cache, and then resumes normal GPU execution for the remainder of the extraction target.
