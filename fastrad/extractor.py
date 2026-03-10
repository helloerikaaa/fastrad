import torch
from typing import Dict, Any, Callable
from .settings import FeatureSettings
from .image import MedicalImage, Mask
from .utils.device import resolve_device

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
    def __init__(self, settings: FeatureSettings):
        self.settings = settings
        self.device = resolve_device(settings.device)

    def extract(self, image: MedicalImage, mask: Mask) -> dict[str, float]:
        # Move tensors to target device
        img_tensor = image.tensor.to(self.device)
        mask_tensor = mask.tensor.to(self.device)
        
        # Ensure image and mask are same shape
        if img_tensor.shape != mask_tensor.shape:
            raise ValueError(f"Image shape {img_tensor.shape} and mask shape {mask_tensor.shape} do not match.")

        features = {}
        
        for feature_class in self.settings.feature_classes:
            if feature_class not in _FEATURE_MAP:
                raise ValueError(f"Unknown feature class: {feature_class}")
            
            compute_fn = _FEATURE_MAP[feature_class]
            
            class_features = compute_fn(
                image_tensor=img_tensor,
                mask_tensor=mask_tensor,
                settings=self.settings
            )
            
            features.update(class_features)
            
        return features
