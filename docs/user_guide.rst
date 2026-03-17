.. _user-guide:

User Guide
==========

The ``fastrad`` User Guide provides a comprehensive deep-dive into the internal workings of the library. It is designed for clinical researchers, machine learning engineers, and data scientists looking to seamlessly integrate GPU-accelerated radiomics pipelines into production environments.

Core Philosophy: PyTorch over Sequences
---------------------------------------

Standard radiomics libraries (e.g., PyRadiomics) utilize the versatile ``SimpleITK`` library to handle spatial definitions. However, these libraries compute features by defining structural loops that iterate consecutively over three-dimensional pixels. This sequential iteration becomes a severe constraint when analyzing high-resolution scans.

``fastrad`` was architected utilizing a contrasting philosophy: **Transform iteration into multidimensional matrix algebra.** By strictly backing the library with PyTorch tensors (``torch.Tensor``), extraction becomes fully vectorised.

.. code-block:: text

   Legacy Radiomics Pipeline           fastrad Pipeline (GPU/CPU)
   =========================           ==========================
   
      [ Load DICOM ]                       [ Load DICOM ]
            ↓                                    ↓  (SimpleITK -> torch.Tensor)
      [ Define ROI ]                       [ Isolate ROI & Crop ]
            ↓                                    ↓  (Allocate to VRAM)
     For z in volume:                   [ Tensor Convolution & Shift ]
       For y in volume:                          ↓  (Parallel Compute)
         For x in volume:                 [ Aggregate Statistics ]
           Extract()                             ↓

By offloading calculations simultaneously across hardware cores (either CPU threads or thousands of CUDA cores), the spatial matrix computations finish nearly instantly.

.. note::  
   The primary tradeoff for this massive speedup is RAM consumption. Sequential systems retain small memory allocations. ``fastrad`` requires caching the entire 3D Region of Interest concurrently matrix-side. Always ensure your hardware possesses adequate Video RAM (VRAM) when scaling to massive tumor bounds.

Object-Orientated Architecture
------------------------------

``fastrad`` exposes three foundational classes to orchestrate the internal pipeline dynamically.

1. **MedicalImage**
   The primary object storing the physical radiomics scan natively as a 32-bit floating-point tensor (``torch.float32``).
   
   - **Initialization**: You can construct it natively from a directory containing DICOM slices using ``MedicalImage.from_dicom("/path/to/series")``. 
   - **Spacing**: Physical dimensions (e.g., $z=1.0, y=0.5, x=0.5$ mm) are permanently attached and passed dynamically through textural filters.

2. **Mask**
   Represents your Region of Interest (ROI). 
   
   - **Validation Enforcement**: The mask is mechanically validated prior to feature evaluation to ensure the internal geometric bounds perfectly align with the targeted ``MedicalImage``. 
   - **Boolean Logic**: Values intrinsically $> 0.5$ evaluate to an absolute internal boolean logic mask. It natively rejects blank segmentations, mitigating silent pipeline failures dynamically.

3. **FeatureSettings**
   The configuration blueprint initializing the extractor logic. 
   
   - ``feature_classes`` (List[str]): Array mapping to strictly compute elements. Defaults to all 7 IBSI-compliant modules.
   - ``bin_width`` (float): Discretization resolution. The standard clinical pipeline recommends ``25.0`` for Hounsfield Units (CT scans).
   - ``device`` (str): Pass ``"cpu"`` for standard parallel execution, ``"cuda"`` for GPU VRAM offloading (requires NVIDIA hardware), or ``"auto"`` for dynamic physical detection routines.


Extraction Workflow Execution
-----------------------------

To actually compute features, you instantiate a ``FeatureExtractor`` with your settings blueprint, and run ``.extract()``.

.. code-block:: python

   extractor = FeatureExtractor(settings)
   features_dict = extractor.extract(image, mask)

This singular API call executes entirely safely against local hardware environments. Edge cases and warnings are heavily anticipated:

VRAM Protection and Fallbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Safely deploying against clinical workstations requires respecting systemic graphical constraints. 

If you target ``device="cuda"`` on an entry-level localized workstation, generating massive textural arrays (like the Gray Level Run Length Matrix for a 150mm tumor bound) may exceed explicit GPU memory allocations, triggering a ``torch.cuda.OutOfMemoryError``. 

Instead of abruptly aborting the evaluation script:
1. ``fastrad`` automatically intercepts the CUDA exception payload.
2. The specific matrix process cache uniquely clears (``torch.cuda.empty_cache()``).
3. The failed feature isolates dynamically back to system multi-threaded CPU environments securely.
4. The extraction pipeline cleanly resumes normal GPU extraction execution for the remainder of the configured feature classes.

Handling Anisotropy
^^^^^^^^^^^^^^^^^^^

When applying spatial queries on scans strictly violating uniform pixel arrays (e.g., MRI sequences scaled $3.0 \times 0.5 \times 0.5$ mm), the geometric shift definitions $\delta$ skew the target calculations fundamentally. 

.. warning::
   ``fastrad`` actively warns users when spacing boundaries exhibit heavy geometric warping parameters natively ($>1e-3$ deviation). It does *not* abort extraction natively, fulfilling raw evaluation rules, but researchers are recommended to utilize spatial resampling logic parameters upstream securely.

Single Voxel Limitations
^^^^^^^^^^^^^^^^^^^^^^^^

If an analysis queries an explicit region containing merely a singular internal voxel, ``fastrad`` securely isolates the structural exception. Standard first-order attributes fundamentally map appropriately against arrays of size 1 (i.e. returning identity boundaries natively). However, multi-dimensional textural analyses are explicitly illogical without spatial neighbors. The pipeline actively registers standard warnings preventing obscure ``NaN`` accumulations without interrupting dynamic batches explicitly.
