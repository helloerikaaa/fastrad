Features & Mathematical Formulations
====================================

`fastrad` implements seven foundational radiomics feature classes completely compliant with the **IBSI (Image Biomarker Standardisation Initiative)**. While our backend circumvents procedural looping via PyTorch vectorization, the output mathematical equations are identical to legacy packages like **PyRadiomics**.

First Order Statistics
----------------------

First-order statistics describe the distribution of voxel intensities within the image region $X$ defined by the mask through basic and commonly used metrics. Let $N_p$ be the number of voxels, and $X(i)$ be the intensity of the $i$-th voxel.

.. math::
    \text{Mean} = \frac{1}{N_p} \sum_{i=1}^{N_p} X(i)

.. math::
    \text{Variance} = \frac{1}{N_p - 1} \sum_{i=1}^{N_p} (X(i) - \text{Mean})^2

.. math::
    \text{Entropy} = -\sum_{i=1}^{N_g} p(i) \log_2(p(i) + \epsilon)

Where $N_g$ is the number of discrete intensity bins and $p(i)$ is the probability of a voxel having intensity $i$.

*   **Available Features:** Mean, Median, Variance, Skewness, Kurtosis, Maximum, Minimum, Range, Interquartile Range, Root Mean Squared (RMS), Energy, Total Energy, Entropy, Uniformity, 90th/10th Percentiles, Robust Mean Absolute Deviation.

Shape (2D and 3D)
-----------------

Shape descriptors quantify the morphological and geometric properties of the Region of Interest (ROI), completely independent of the actual pixel intensities inside the tumor or organ. 

.. math::
    \text{Sphericity} = \frac{\pi^{\frac{1}{3}} (6V)^{\frac{2}{3}}}{A}

Where $V$ is the Volume (number of voxels $\times$ physical voxel volume) and $A$ is the Surface Area computed via marching-cubes or voxel-faces matching.

*   **Available Features:** Maximum 3D Diameter, Volume, Surface Area, Sphericity, Surface to Volume Ratio, Elongation, Flatness, Major/Minor Axis Length (via PCA constraints).

Gray Level Co-occurrence Matrix (GLCM)
--------------------------------------

A GLCM of size $N_g \times N_g$ describes the second-order joint probability function of an image region. Let $P(i, j)$ be the probability that a voxel with intensity $i$ is separated from a voxel with intensity $j$ by a fixed distance $\delta$ in a defined angle.

.. math::
    \text{Contrast} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_g} (i - j)^2 P(i,j)

.. math::
    \text{Correlation} = \frac{\sum_{i=1}^{N_g} \sum_{j=1}^{N_g} p(i,j) i j - \mu_{x} \mu_{y}}{\sigma_{x}(i) \sigma_{y}(j)}

`fastrad` computes all 13 symmetric 3D angles simultaneously using volumetric shifted tensor concatenations (`torch.roll`), eliminating nested pixel lookups entirely.

*   **Available Features:** Autocorrelation, Joint Average, Cluster Prominence, Cluster Shade, Cluster Tendency, Contrast, Correlation, Inverse Difference Moment (IDM), Maximum Probability, Sum Average, Sum Variance.

Gray Level Size Zone Matrix (GLSZM)
-----------------------------------

A connected component $c$ is defined as a sequence of logically connected voxels exhibiting identical discretized gray-levels. The GLSZM $P(i, j)$ represents the number of zones with gray-level $i$ and voxel size $j$.

.. math::
    \text{Small Area Emphasis (SAE)} = \frac{1}{N_z} \sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \frac{P(i,j)}{j^2}

Where $N_z$ is the total zone count in the volume, and $N_s$ is the maximum zone size limits.

*Note: fastrad uniquely implements this algorithm utilizing a high-performance CuPy / cuCIM topological labeling subroutine when executed on GPUs, avoiding the severe performance trap common to zone labeling.*

*   **Available Features:** Small/Large Area Emphasis, Gray Level Non-Uniformity, Zone Percentage, Zone Variance, Size Zone Non-Uniformity.

Gray Level Dependence Matrix (GLDM)
-----------------------------------

A GLDM $P(i, j)$ tracks occurrences where a center voxel defined as gray level $i$ has precisely $j$ surrounding neighbor voxels within a radius $\delta$ that share an identical gray level (or fall within a strict alpha tolerance boundary).

.. math::
    \text{Dependence Entropy} = -\sum_{i=1}^{N_g}\sum_{j=1}^{N_d} p(i,j) \log_2(p(i,j) + \epsilon)

`fastrad` accelerates 3D dependency aggregations via dynamic spherical boolean masks multiplied across massive 4D pre-allocated tensors.

*   **Available Features:** Small/Large Dependence Emphasis, Gray Level/Dependence Non-Uniformity, Dependence Variance, Dependence Entropy.

Neighbourhood Gray Tone Difference Matrix (NGTDM)
-------------------------------------------------

Quantifies the difference between a specific gray value $i$ and the average gray value of its immediate spatial neighbors $\bar{A}_i$ extending up to distance $\delta$.

.. math::
    s_i = \sum^{N_p}_{k=1} |i - \bar{A}_i| \text{ for } X(k)=i

.. math::
    \text{Coarseness} = \frac{1}{\sum_{i=1}^{N_g} p_i \cdot s_i}

*   **Available Features:** Coarseness, Contrast, Busyness, Complexity, Strength.
