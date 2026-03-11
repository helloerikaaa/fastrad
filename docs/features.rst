Features Overview
=================

`fastrad` implements seven of the most widely used radiomics feature classes globally. 

In total, `fastrad` computes over 100+ unique radiomic descriptors per Region of Interest (ROI), fully compliant with the **IBSI (Image Biomarker Standardisation Initiative)** benchmarks and demonstrating stringent parity against the reference software **PyRadiomics**.

First Order Statistics
----------------------
Describes the distribution of voxel intensities within the image region defined by the mask through basic and commonly used metrics.

- Mean, Median, Variance, Skewness, Kurtosis
- Maximum, Minimum, Range, Interquartile Range
- Root Mean Squared (RMS), Energy, Total Energy
- Entropy, Uniformity
- 90th/10th Percentiles, Robust Mean Absolute Deviation

Shape (2D and 3D)
-----------------
Describes the morphological and geometric properties of the Region of Interest (ROI) completely independent of the gray level intensity distribution.

- Maximum 3D Diameter, Volume, Surface Area
- Sphericity, Surface to Volume Ratio
- Elongation, Flatness, Major/Minor Axis Length (via PCA)

Gray Level Co-occurrence Matrix (GLCM)
--------------------------------------
Quantifies the incidence of voxels with the same intensities occurring at specified distances and angles relative to one another in the image matrix.

- Autocorrelation, Joint Average
- Cluster Prominence/Shade/Tendency
- Contrast, Correlation, Inverse Difference Moment (IDM)
- Maximum Probability, Sum Average, Sum Variance

Gray Level Run Length Matrix (GLRLM)
------------------------------------
Quantifies homogeneous runs (consecutive voxels having the same gray level intensity).

- Short/Long Run Emphasis
- Gray Level Non-Uniformity
- Run Length Non-Uniformity
- Run Percentage, Run Variance

Gray Level Size Zone Matrix (GLSZM)
-----------------------------------
Quantifies the number of connected groups (zones) of voxels that share the same gray level intensity. *Note: `fastrad` uses a highly optimized `scipy/cucim` hybrid approach for massive GPU speedups here.*

- Small/Large Area Emphasis
- Gray Level Non-Uniformity
- Zone Percentage, Zone Variance
- Size Zone Non-Uniformity

Gray Level Dependence Matrix (GLDM)
-----------------------------------
Quantifies gray level dependencies in an image defined as the number of connected voxels within distance $\\delta$ that are dependent on the center voxel.

- Small/Large Dependence Emphasis
- Gray Level Non-Uniformity
- Dependence Non-Uniformity
- Dependence Variance, Dependence Entropy

Neighbourhood Gray Tone Difference Matrix (NGTDM)
-------------------------------------------------
Quantifies the difference between a gray value and the average gray value of its neighbors within distance $\\delta$.

- Coarseness
- Contrast
- Busyness
- Complexity
- Strength
