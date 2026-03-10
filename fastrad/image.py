import torch
from pathlib import Path
from typing import Union, Tuple

class MedicalImage:
    def __init__(self, tensor: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.tensor = tensor
        self.spacing = spacing

    @classmethod
    def from_dicom(cls, path: Union[str, Path]) -> "MedicalImage":
        raise NotImplementedError("DICOM loading not yet implemented")

class Mask:
    def __init__(self, tensor: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.tensor = tensor
        self.spacing = spacing

    @classmethod
    def from_dicom(cls, path: Union[str, Path]) -> "Mask":
        raise NotImplementedError("DICOM mask loading not yet implemented")
