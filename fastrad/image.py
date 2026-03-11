import torch
from pathlib import Path
from typing import Union, Tuple
from .utils.dicom import parse_dicom_dir

class MedicalImage:
    """
    Representation of a continuous 3D medical volume (e.g., CT or MRI).
    
    Attributes:
        tensor (torch.Tensor): The 3D image array as a PyTorch FloatTensor.
        spacing (Tuple[float, float, float]): The physical voxel dimensions (z, y, x).
    """
    def __init__(self, tensor: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.tensor = tensor
        self.spacing = spacing

    @classmethod
    def from_dicom(cls, path: Union[str, Path]) -> "MedicalImage":
        """
        Creates a MedicalImage from a directory containing DICOM slices.
        
        Args:
            path: Path to the directory containing the DICOM files.
            
        Returns:
            A new instantiated MedicalImage object with extracted spacing.
        """
        tensor, spacing = parse_dicom_dir(path)
        return cls(tensor=tensor, spacing=spacing)

class Mask:
    """
    Representation of a binary 3D Region of Interest (ROI) mask.
    
    Attributes:
        tensor (torch.Tensor): The 3D binary mask array as a PyTorch FloatTensor.
        spacing (Tuple[float, float, float]): The physical voxel dimensions (z, y, x).
    """
    def __init__(self, tensor: torch.Tensor, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        # Ensure mask is binary (0 or 1)
        self.tensor = (tensor > 0).to(torch.float32)
        self.spacing = spacing

    @classmethod
    def from_dicom(cls, path: Union[str, Path]) -> "Mask":
        """
        Creates a binary Mask from a directory containing DICOM slices.
        Voxels strictly greater than 0 are set to 1.
        
        Args:
            path: Path to the directory containing the DICOM ROIs.
            
        Returns:
            A new instantiated binary Mask object with extracted spacing.
        """
        tensor, spacing = parse_dicom_dir(path)
        return cls(tensor=tensor, spacing=spacing)

def get_binned_image(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, bin_width: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes PyRadiomics-compatible binning.
    
    1. Floor divisions of pixels by bin_width anchored at minimum
    2. Maps the unique present bins within the mask to continuous integers 1..Ng
       (Empty intermediate bins are removed)
    
    Returns:
        binned_image (torch.Tensor): Remapped binned image
        Ng (torch.Tensor): The array of unique raw bin values present in the mask (ivector)
    """
    voxels = image_tensor[mask_tensor > 0.5]
    if voxels.numel() == 0:
        return torch.zeros_like(image_tensor), torch.empty(0, dtype=torch.float64, device=image_tensor.device)
        
    img_min = torch.min(voxels)
    minimum_binned = torch.floor(img_min / bin_width) * bin_width
    
    # Initial raw absolute binning
    raw_binned = torch.floor((image_tensor - minimum_binned) / bin_width) + 1
    raw_binned_voxels = raw_binned[mask_tensor > 0.5]
    
    # Find unique bins actually present in the mask
    unique_bins = torch.unique(raw_binned_voxels)
    Ng = unique_bins.numel()
    
    if Ng == 0:
        return torch.zeros_like(image_tensor), unique_bins.to(torch.float64)
        
    
    # We maintain raw values because NGTDM mathematical filtering requires actual relative offset intervals
    binned_image = raw_binned.to(torch.float32)
    
    return binned_image, unique_bins.to(torch.float64)

