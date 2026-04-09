from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class FeatureSettings:
    """
    Configuration parameters regulating standard feature extraction pipelines.
    
    Attributes:
        feature_classes: List of feature groups to extract (e.g., ``["firstorder", "glcm"]``).
        bin_width: Radiometric discretization width applied before extracting textures.
        device: Target execution device (``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``).
        spacing: Real world physical spacing array `(z, y, x)` propagated downstream.
        force2D: Flag controlling slice-by-slice 2D calculation routing.
        force2Ddimension: Target dimension slice axis when projecting 3D volumes to 2D.
    """
    feature_classes: List[str] = field(default_factory=lambda: ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"])
    bin_width: float = 25.0
    device: str = "auto"
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    force2D: bool = False
    force2Ddimension: int = 0
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    amp: bool = False
    differentiable: bool = False
