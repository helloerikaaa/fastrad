import torch
from fastrad.settings import FeatureSettings
from fastrad.image import get_binned_image

EPSILON = 1e-16

import numpy as np

def _label_connected_components(mask_np: np.ndarray, 
                                device: torch.device) -> torch.Tensor:
    structure = np.ones((3, 3, 3), dtype=int) if mask_np.ndim == 3 else np.ones((3, 3), dtype=int)
    if device.type == "cuda":
        try:
            from cucim.core.operations.morphology import label as cucim_label
            import cupy as cp
            mask_cp = cp.asarray(mask_np.astype("int32"))
            labeled = cucim_label(mask_cp, structure=structure)
            return torch.from_dlpack(labeled.toDlpack()).to(device)
        except Exception as e:
            print(f"DEBUG: cuCIM dispatch failed: {e}")
            pass  # fall through to scipy
            
    from scipy.ndimage import label as scipy_label
    labeled, _ = scipy_label(mask_np, structure=structure)
    return torch.from_numpy(labeled.astype("int32")).to(device)

def compute(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    device = image_tensor.device
    
    binned_image, ivector = get_binned_image(image_tensor, mask_tensor, settings.bin_width)
    Ng = ivector.numel()
    if Ng == 0:
        return {}
        
    voxels = image_tensor[mask_tensor > 0.5]
        
    M = mask_tensor > 0.5
    img_int = binned_image.to(torch.int64) * M
    
    if int(torch.max(img_int).item()) == 0:
        return {}
        
    import scipy.ndimage
    import numpy as np
    
    zero_tensor = torch.tensor([0], dtype=img_int.dtype, device=device)
    _, inverse_indices = torch.unique(torch.cat((zero_tensor, img_int.view(-1))), return_inverse=True)
    mapped_int = inverse_indices[1:].view(img_int.shape)
        
    img_np = mapped_int.cpu().numpy()
    
    current_max_label = 0
    labels_tensor = torch.zeros(img_int.shape, dtype=torch.int64, device=device)
    
    # Rapid bounding box identification of all distinct gray levels
    slices = scipy.ndimage.find_objects(img_np)
    
    for g_idx, g_slices in enumerate(slices):
        if g_slices is None:
            continue
            
        g = g_idx + 1 # unique contiguous bins are 1-indexed
        crop_img = img_np[g_slices]
        binary_mask = (crop_img == g)
        
        labeled_mask = _label_connected_components(binary_mask, device)
        max_label = int(labeled_mask.max().item())
            
        if max_label > 0:
            mask_pos = labeled_mask > 0
            labeled_mask[mask_pos] += current_max_label
            labels_tensor[g_slices] += labeled_mask
            current_max_label += max_label
            
    labels = labels_tensor            
    valid_labels = labels[M]
    valid_gray = img_int[M]
    
    unique_labels, inverse_indices, counts = torch.unique(valid_labels, return_inverse=True, return_counts=True)
    
    N_c = unique_labels.numel()
    if N_c == 0:
        return {}
        
    comp_gray = torch.zeros(N_c, dtype=img_int.dtype, device=device)
    comp_gray.scatter_(0, inverse_indices, valid_gray)
    
    Nz = int(torch.max(counts).item())
    
    if Nz == 0:
        return {}
        
    g = comp_gray - 1
    s = counts - 1
    
    max_gl = int(torch.max(ivector).item())
    linear_indices = g * Nz + s
    matrix_counts = torch.bincount(linear_indices, minlength=max_gl*Nz).to(torch.float64)
    P_raw = matrix_counts[:max_gl*Nz].view(max_gl, Nz)
    
    valid_idx = (ivector - 1).to(torch.int64)
    P = P_raw[valid_idx, :]
    
    Ns = P.sum()
    if Ns == 0:
        return {}
        
    Np = voxels.numel()
    
    i_grid = ivector.clone().to(device).view(-1, 1)
    j_grid = torch.arange(1, Nz + 1, dtype=torch.float64, device=device).view(1, -1)
    
    pg = torch.sum(P, dim=1).view(-1, 1)
    pz = torch.sum(P, dim=0).view(1, -1)
    
    sae = torch.sum(pz / (j_grid ** 2)) / Ns
    lae = torch.sum(pz * (j_grid ** 2)) / Ns
    
    glnu = torch.sum(pg ** 2) / Ns
    glnun = torch.sum(pg ** 2) / (Ns ** 2)
    
    szn = torch.sum(pz ** 2) / Ns
    sznn = torch.sum(pz ** 2) / (Ns ** 2)
    
    zp = Ns / Np
    
    p_norm = P / Ns
    mu_i = torch.sum(i_grid * p_norm)
    glv = torch.sum(((i_grid - mu_i) ** 2) * p_norm)
    
    mu_j = torch.sum(j_grid * p_norm)
    zv = torch.sum(((j_grid - mu_j) ** 2) * p_norm)
    
    p_log = torch.where(p_norm > EPSILON, torch.log2(p_norm), torch.zeros_like(p_norm))
    ze = -torch.sum(p_norm * p_log)
    
    lglze = torch.sum(pg / (i_grid ** 2)) / Ns
    hglze = torch.sum(pg * (i_grid ** 2)) / Ns
    
    salgle = torch.sum(P / ((i_grid ** 2) * (j_grid ** 2))) / Ns
    sahgle = torch.sum(P * (i_grid ** 2) / (j_grid ** 2)) / Ns
    lalgle = torch.sum(P * (j_grid ** 2) / (i_grid ** 2)) / Ns
    lahgle = torch.sum(P * (i_grid ** 2) * (j_grid ** 2)) / Ns
    
    features = {
        "glszm:small_area_emphasis": sae.item(),
        "glszm:large_area_emphasis": lae.item(),
        "glszm:gray_level_non_uniformity": glnu.item(),
        "glszm:gray_level_non_uniformity_normalized": glnun.item(),
        "glszm:size_zone_non_uniformity": szn.item(),
        "glszm:size_zone_non_uniformity_normalized": sznn.item(),
        "glszm:zone_percentage": zp.item(),
        "glszm:gray_level_variance": glv.item(),
        "glszm:zone_variance": zv.item(),
        "glszm:zone_entropy": ze.item(),
        "glszm:low_gray_level_zone_emphasis": lglze.item(),
        "glszm:high_gray_level_zone_emphasis": hglze.item(),
        "glszm:small_area_low_gray_level_emphasis": salgle.item(),
        "glszm:small_area_high_gray_level_emphasis": sahgle.item(),
        "glszm:large_area_low_gray_level_emphasis": lalgle.item(),
        "glszm:large_area_high_gray_level_emphasis": lahgle.item()
    }
    
    return features
