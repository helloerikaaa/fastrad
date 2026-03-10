from dataclasses import dataclass, field
from typing import List

@dataclass
class FeatureSettings:
    feature_classes: List[str] = field(default_factory=lambda: ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"])
    bin_width: float = 25.0
    device: str = "auto"
