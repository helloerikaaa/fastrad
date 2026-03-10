import torch
from pathlib import Path
from typing import Union, Tuple
from .utils.dicom import parse_dicom_dir

class MedicalImage:
    def __init__(self, tensor: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.tensor = tensor
        self.spacing = spacing

    @classmethod
    def from_dicom(cls, path: Union[str, Path]) -> "MedicalImage":
        tensor, spacing = parse_dicom_dir(path)
        return cls(tensor=tensor, spacing=spacing)

class Mask:
    def __init__(self, tensor: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        # Ensure mask is binary (0 or 1)
        self.tensor = (tensor > 0).to(torch.float32)
        self.spacing = spacing

    @classmethod
    def from_dicom(cls, path: Union[str, Path]) -> "Mask":
        tensor, spacing = parse_dicom_dir(path)
        return cls(tensor=tensor, spacing=spacing)
