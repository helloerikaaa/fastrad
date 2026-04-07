Quickstart: Evaluating a Lung Nodule
====================================

``fastrad`` is engineered to be a seamless, high-performance drop-in upgrade for standard clinical radiomics pipelines. This quickstart guide will walk you through a practical, end-to-end scenario: **Extracting 100+ textural biomarkers from a 3D structural lung nodule utilizing GPU acceleration.**

Prerequisites
-------------
Ensure you have installed ``fastrad`` alongside a DICOM handler like ``SimpleITK``.

.. code-block:: bash

   pip install fastrad SimpleITK torch

Step 1: Load the Clinical Volume
--------------------------------
``fastrad`` separates the physical geometry of the scan from the Region of Interest (the segmentation). In this scenario, we possess a directory of DICOM files containing a patient CT scan, and a structural NIfTI file containing the explicit 3D boundaries of the radiologist's tracing.

.. code-block:: python

   import SimpleITK as sitk
   import torch
   from fastrad import MedicalImage, Mask

   # Load the structural CT Array
   dicom_dir = "/data/TCIA/RIDER-102/CT_Scan"
   image = MedicalImage.from_dicom(dicom_dir)

   # Alternatively, load the explicit Mask using SimpleITK directly to manipulate arrays
   mask_path = "/data/TCIA/RIDER-102/tumor_segmentation.nii.gz"
   sitk_mask = sitk.ReadImage(mask_path)
   
   # Convert to native PyTorch Tensor geometry
   mask_tensor = torch.tensor(sitk.GetArrayFromImage(sitk_mask), dtype=torch.float32)
   spacing = sitk_mask.GetSpacing()
   
   # fastrad strictly maps (z, y, x) configurations natively
   mask = Mask(tensor=mask_tensor, spacing=(spacing[2], spacing[1], spacing[0]))

Step 2: Configure the Extractor Blueprint
-----------------------------------------
The ``FeatureSettings`` object dictates the mathematical limits and hardware routing for your pipeline. The `bin_width` parameter is clinically critical: it defines how raw Hounsfield Units (HU) are quantized into discrete matrix bins for textural features like the GLCM. The IBSI commonly utilizes widths of `25.0` for CT algorithms.

.. code-block:: python

   from fastrad import FeatureSettings

   settings = FeatureSettings(
       # Define the algorithmic scope
       feature_classes=["firstorder", "shape", "glcm", "glrlm", "glszm"],
       
       # IBSI-compliant mathematical binning rule
       bin_width=25.0,
       
       # The Core Upgrade: Dispatch to CUDA natively
       device="cuda" 
   )

Step 3: Execute the Extraction
------------------------------
Initialize the orchestrator and extract your features. If your configured machine lacks CUDA capabilities, ``fastrad`` dynamically detects the hardware restrictions and resolves gracefully to your CPU without destroying your configuration.

.. code-block:: python

   from fastrad import FeatureExtractor

   extractor = FeatureExtractor(settings)
   
   # Extraction occurs instantaneously across the physical PyTorch graph
   features = extractor.extract(image, mask)

   print(f"Extraction Successful! Gathered {len(features)} dimensional vectors.")

Step 4: Consume the Feature Output
----------------------------------
The ``features`` variable is a standard Python dictionary identically mirroring PyRadiomics outputs, making it extremely easy to plug natively into pandas, scikit-learn classifiers, or PyTorch neural networks.

.. code-block:: python

   # Display physical characteristics
   print(f"Maximum 3D Diameter: {features['shape:Maximum3DDiameter']:.2f} mm")
   print(f"Tumor Volume:        {features['shape:MeshVolume']:.2f} mm^3")
   
   # Display complex spatial textural signatures
   print(f"GLCM Contrast:       {features['glcm:Contrast']:.4f}")
   print(f"GLSZM Area Emphasis: {features['glszm:SmallAreaEmphasis']:.4f}")

Next Steps
----------
The above example processed a single volume safely. However, standard clinical research evaluates thousands of patients identically. 

* To learn about iterating over batch arrays and resolving hardware Out-Of-Memory limitations, read our exhaustive :ref:`user-guide`.
* To investigate the core 100+ IBSI vector rules mathematically extracted internally, see the formal Mathematical Documentation.

Advanced: Hardware-Accelerated Voxel-wise Feature Extraction
-----------------------------------------------------------
If your pipeline requires dense feature maps fed directly into downstream neural networks (like CNNs or Transformers), you can utilize our hardware-accelerated memory-strided patch view extractor without altering your mathematical blueprints:

.. code-block:: python

   from fastrad import DenseFeatureExtractor

   dense_extractor = DenseFeatureExtractor(settings)
   
   # Extracts dense sliding window feature maps 
   # Using a kernel size of 3x3x3 voxels and stride of 1.
   dense_features = dense_extractor.extract_dense(image, mask, kernel_size=3, stride=1)
   
   # Output maps are full natively tracked PyTorch Tensors on your hardware configuration
   print(f"Dense Feature Map Shape (Energy): {dense_features['firstorder:energy'].shape}")
