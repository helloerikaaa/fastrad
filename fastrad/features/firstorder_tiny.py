from tinygrad import Tensor, TinyJit
from typing import Dict
from fastrad.settings import FeatureSettings

# We extract the pure compute logic so we can wrap it in a `@TinyJit` decorator later for hardware acceleration.
def compute_firstorder_core(image: Tensor, mask: Tensor) -> list[Tensor]:
    '''
    Executes firstorder stats via minimal graph operations.
    Tinygrad shapes must be static for JIT, so we compute structurally across the volume natively
    instead of extracting a flattened dynamic array.
    '''
    # Mask is binary (0.0 or 1.0)
    valid_mask = mask > 0.5
    N = valid_mask.sum()
    
    # Zeros out the image where mask is false
    valid_voxels = image * valid_mask
    
    # Basic Stats
    mean = valid_voxels.sum() / N
    
    # We use extreme bounds inversely for max/min to avoid analyzing 0s from the background
    maximum = valid_mask.where(image, -1e10).max()
    minimum = valid_mask.where(image, 1e10).min()
    
    rng = maximum - minimum
    
    # Variance and Std
    centered = valid_mask.where(image - mean, 0.0)
    variance = (centered ** 2).sum() / N
    std = variance.sqrt()
    
    # Energy
    energy = (valid_voxels ** 2).sum()
    
    rms = energy / N
    rms = rms.sqrt()
    
    mad = valid_mask.where((image - mean).abs(), 0.0).sum() / N
    
    # Skewness and Kurtosis
    m3 = (centered ** 3).sum() / N
    m4 = (centered ** 4).sum() / N
    
    epsilon = 1e-16
    skewness = m3 / (std ** 3 + epsilon)
    kurtosis = m4 / (std ** 4 + epsilon)
    
    # We return a list of tensors so JIT knows what the static outputs are.
    return [energy, minimum, maximum, mean, rng, mad, rms, std, skewness, kurtosis, variance]

# Applying TinyJit wrapper. The computations will be fused and launched strictly in optimized metal/cuda kernels!
@TinyJit
def _jit_compute(image: Tensor, mask: Tensor) -> list[Tensor]:
    return compute_firstorder_core(image, mask)

def compute(image_tensor, mask_tensor, settings: FeatureSettings) -> Dict[str, float]:
    """
    Adapter bridging generic tensor matrices into Tinygrad
    """
    import numpy as np
    
    # If it is pytorch, cast to numpy first, then init Tinygrad
    if hasattr(image_tensor, "cpu"):
        image_np = image_tensor.cpu().numpy().astype(np.float32)
        mask_np = mask_tensor.cpu().numpy().astype(np.float32)
    else:
        image_np = image_tensor.astype(np.float32)
        mask_np = mask_tensor.astype(np.float32)
        
    img_t = Tensor(image_np)
    mask_t = Tensor(mask_np)
    
    # Trigger JIT
    res = _jit_compute(img_t, mask_t)
    
    # Realize the lazy graphs structurally natively
    Tensor.realize(*res)
    
    voxel_volume = settings.spacing[0] * settings.spacing[1] * settings.spacing[2]
    
    features = {
        "firstorder:energy": res[0].item(),
        "firstorder:total_energy": res[0].item() * voxel_volume,
        "firstorder:minimum": res[1].item(),
        "firstorder:maximum": res[2].item(),
        "firstorder:mean": res[3].item(),
        "firstorder:range": res[4].item(),
        "firstorder:mean_absolute_deviation": res[5].item(),
        "firstorder:root_mean_squared": res[6].item(),
        "firstorder:standard_deviation": res[7].item(),
        "firstorder:skewness": res[8].item(),
        "firstorder:kurtosis": res[9].item(),
        "firstorder:variance": res[10].item(),
    }
    
    return features
