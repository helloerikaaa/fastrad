Introduction
============

**FastRadiomics (fastrad)** is a modern, high-performance Python library designed for extracting radiomics features from medical images.

Built as a direct, highly optimized alternative to the widely used `PyRadiomics` library, `fastrad` achieves **strict computational parity** with PyRadiomics while enabling dramatically faster execution times. By implementing image feature extraction natively in PyTorch, `fastrad` bypasses the limitations of traditional CPU-bound scalar operations.

Why fastrad?
------------

Traditional radiomics libraries, while scientifically robust and compliant with the Image Biomarker Standardisation Initiative (IBSI), often struggle to scale when processing large multi-modal datasets or high-resolution volumetric scans (like 512x512xN CT scans). This bottleneck can bring modern AI and machine learning pipelines to a halt.

**fastrad** solves this by treating radiomics feature extraction as a hardware-accelerated tensor problem:
 
- **Full Parity:** Every single feature has been rigorously tested against PyRadiomics to ensure identical values extracted at high precision. You can swap `fastrad` into your existing scientific pipelines without degrading experimental validity.
- **Hardware Acceleration:** Under the hood, `fastrad` leverages PyTorch block processing. This allows it to natively route extraction workloads across multi-core CPUs, CUDA-enabled NVIDIA GPUs, and even Apple Silicon (MPS).
- **Scalable Matrix Construction:** Texture matrices (GLCM, GLRLM, GLSZM, GLDM, NGTDM) are notoriously slow to compute sequentially. `fastrad` heavily vectorizes these constructions.

Use Cases
---------

- **High-Throughput Research:** Compute features over cohorts of thousands of patients in a fraction of the time.
- **Real-Time Inference:** Embed radiomics extractors seamlessly inside end-to-end PyTorch deep learning training loops or clinical deployment pipelines without facing severe CPU throttling.
- **Reproducible Science:** Lean on our extensive unit-tests and stability analysis tools simulating adversarial 3D affine transformations to guarantee robust feature replication.

*Note: Since statistical textural features demand extremely high 64-bit precision, CUDA or CPU are the primary recommended execution targets for perfect parity, as some PyTorch MPS (Apple Silicon) operations do not natively support FP64 operations yet.*
