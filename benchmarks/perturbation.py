import torch
import numpy as np
import scipy.ndimage

def apply_perturbation(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, 
                      translation: tuple = (2.0, 2.0, 2.0), 
                      rotation_deg: float = 5.0, 
                      noise_std: float = 5.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a synthetic scan-rescan perturbation to an image and mask.
    This includes:
    1. A small 3D affine transformation (translation + rotation).
    2. Additive Gaussian noise to the image.
    
    Returns:
        (perturbed_image, perturbed_mask) as torch.Tensors.
    """
    img_np = image_tensor.cpu().numpy()
    mask_np = mask_tensor.cpu().numpy()
    
    # 1. Affine Transformation (Translation and simple Z-axis rotation)
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    # Rotation matrix around Z axis
    rot_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_t, -sin_t],
        [0.0, sin_t, cos_t]
    ])
    
    # We apply the affine transform using scipy
    # offset controls translation
    shifted_img = scipy.ndimage.affine_transform(
        img_np, 
        matrix=rot_matrix, 
        offset=translation, 
        order=3, # Cubic interpolation for image
        mode='nearest'
    )
    
    shifted_mask = scipy.ndimage.affine_transform(
        mask_np, 
        matrix=rot_matrix, 
        offset=translation, 
        order=0, # Nearest neighbor interpolation for binary mask
        mode='constant',
        cval=0.0
    )
    
    # 2. Additive Gaussian Noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=shifted_img.shape)
    noisy_img = shifted_img + noise
    
    # Mask out noise outside the ROI for clean reading (optional, but standard)
    # noisy_img = noisy_img * (shifted_mask > 0.5)
    
    # Recast to tensor
    perturbed_img_tensor = torch.from_numpy(noisy_img).to(torch.float32).to(image_tensor.device)
    perturbed_mask_tensor = torch.from_numpy(shifted_mask).to(torch.float32).to(mask_tensor.device)
    
    # Ensure binary mask
    perturbed_mask_tensor = (perturbed_mask_tensor > 0.5).float()
    
    return perturbed_img_tensor, perturbed_mask_tensor
