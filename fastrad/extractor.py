import logging
import warnings
import torch
from typing import Dict, Any, Callable
from .settings import FeatureSettings
from .image import MedicalImage, Mask
from .utils.device import resolve_device

logger = logging.getLogger(__name__)

# Filter the isotropic spacing warning so it only appears once per runtime.
warnings.filterwarnings("once", message=".*is not isotropic.*")

from .features import (
    compute_firstorder,
    compute_shape,
    compute_shape2d,
    compute_glcm,
    compute_glrlm,
    compute_glszm,
    compute_gldm,
    compute_ngtdm
)

_FEATURE_MAP = {
    "firstorder": compute_firstorder,
    "shape": compute_shape,
    "shape2d": compute_shape2d,
    "glcm": compute_glcm,
    "glrlm": compute_glrlm,
    "glszm": compute_glszm,
    "gldm": compute_gldm,
    "ngtdm": compute_ngtdm
}

class FeatureExtractor:
    """
    Main orchestration engine for radiomics feature extraction.
    
    The FeatureExtractor consumes a FeatureSettings configuration and executes
    the specified feature class modules against a provided image and mask. 
    It automatically routes tensors to the requested device (CPU, CUDA, MPS)
    and handles OutOfMemory fallbacks gracefully.
    """
    def __init__(self, settings: FeatureSettings):
        """
        Initializes the FeatureExtractor with the given settings.
        
        Args:
            settings (FeatureSettings): Configuration defining which features to compute.
        """
        self.settings = settings
        self.device = resolve_device(settings.device)

    def extract(self, image: MedicalImage, mask: Mask) -> dict[str, float]:
        """
        Executes feature extraction on the given Image and Mask.
        
        Args:
            image (MedicalImage): The baseline medical volume.
            mask (Mask): The binary Region of Interest mask.
            
        Returns:
            dict[str, float]: A dictionary mapping feature names to their computed values.
        """
        # Move tensors to target device
        img_tensor = image.tensor.to(self.device)
        mask_tensor = mask.tensor.to(self.device)
        
        # Ensure image and mask are same shape
        if img_tensor.shape != mask_tensor.shape:
            raise ValueError(f"Image shape {img_tensor.shape} and mask shape {mask_tensor.shape} do not match.")

        # Populate spacing for volume calculations
        self.settings.spacing = image.spacing

        # Robustness Checks
        mask_sum = mask_tensor.sum().item()
        
        # 1. Empty Mask Check
        if mask_sum == 0:
            raise ValueError("Mask contains no positive voxels.")
            
        # 2. Single-Voxel ROI Check
        if mask_sum == 1:
            logger.warning(
                "Mask contains exactly one positive voxel. "
                "Many spatial and textural features cannot be computed validly."
            )
            
        # 3. Non-Isotropic Spacing Check
        sp = image.spacing
        if max(sp) - min(sp) > 1e-3:
            warnings.warn(
                f"Image spacing {sp} is not isotropic. "
                "PyRadiomics guidelines recommend resampling to isotropic spacing "
                "for robust textural feature calculation.",
                UserWarning,
                stacklevel=2
            )

        features = {}
        
        for feature_class in self.settings.feature_classes:
            if feature_class not in _FEATURE_MAP:
                raise ValueError(f"Unknown feature class: {feature_class}")
            
            compute_fn = _FEATURE_MAP[feature_class]
            
            device_type = "cuda" if self.device.type == "cuda" else ("cpu" if self.device.type == "cpu" else None)
            
            try:
                if self.settings.amp and device_type:
                    with torch.autocast(device_type=device_type):
                        class_features = compute_fn(
                            image_tensor=img_tensor,
                            mask_tensor=mask_tensor,
                            settings=self.settings
                        )
                else:
                    class_features = compute_fn(
                        image_tensor=img_tensor,
                        mask_tensor=mask_tensor,
                        settings=self.settings
                    )
            except torch.cuda.OutOfMemoryError:
                if self.device.type == "cuda":
                    logger.warning(
                        f"CUDA OutOfMemoryError caught while extracting {feature_class} features. "
                        f"Falling back to CPU computation for this feature class."
                    )
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Move tensors to CPU explicitly for this computation
                    cpu_img_tensor = img_tensor.cpu()
                    cpu_mask_tensor = mask_tensor.cpu()
                    
                    # Compute on CPU
                    if self.settings.amp:
                        with torch.autocast(device_type="cpu"):
                            class_features = compute_fn(
                                image_tensor=cpu_img_tensor,
                                mask_tensor=cpu_mask_tensor,
                                settings=self.settings
                            )
                    else:
                        class_features = compute_fn(
                            image_tensor=cpu_img_tensor,
                            mask_tensor=cpu_mask_tensor,
                            settings=self.settings
                        )
                    
                    # Free up CPU memory
                    del cpu_img_tensor
                    del cpu_mask_tensor
                else:
                    raise
            
            # Enforce PyRadiomics float backward-compatibility natively
            if not self.settings.differentiable:
                class_features = {k: v.item() if hasattr(v, "item") else v for k, v in class_features.items()}
                
            features.update(class_features)
            
        return features
