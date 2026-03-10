import os
from pathlib import Path
from typing import Union, Tuple, List
import pydicom
import torch
import numpy as np

def parse_dicom_dir(path: Union[str, Path]) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """
    Reads a directory of DICOM files, sorts them by spatial location,
    and returns a 3D PyTorch tensor (D, H, W) and spacing (z, y, x).
    """
    path = Path(path)
    slices: List[pydicom.FileDataset] = []
    
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, force=True)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception:
                pass
                
    if not slices:
        raise ValueError(f"No DICOM files with pixel data found in {path}")

    # Sort slices by InstanceNumber or SliceLocation
    if hasattr(slices[0], 'InstanceNumber'):
        slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
    elif hasattr(slices[0], 'SliceLocation'):
        slices.sort(key=lambda x: float(getattr(x, 'SliceLocation', 0.0)))
        
    # Extract spacing: (z, y, x)
    # y, x comes from PixelSpacing (row, col)
    # z comes from SpacingBetweenSlices or SliceThickness
    pixel_spacing = getattr(slices[0], 'PixelSpacing', [1.0, 1.0])
    y_spacing = float(pixel_spacing[0])
    x_spacing = float(pixel_spacing[1])
    
    z_spacing = 1.0
    if hasattr(slices[0], 'SpacingBetweenSlices'):
        z_spacing = float(slices[0].SpacingBetweenSlices)
    elif hasattr(slices[0], 'SliceThickness'):
        z_spacing = float(slices[0].SliceThickness)
        
    spacing = (z_spacing, y_spacing, x_spacing)
    
    # Stack pixel arrays into 3D volume (D, H, W)
    # Note: Using float32 for internal representation
    volume_np = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    volume_tensor = torch.from_numpy(volume_np)
    
    return volume_tensor, spacing
