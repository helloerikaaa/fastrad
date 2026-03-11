import torch
from fastrad.settings import FeatureSettings
from fastrad.image import get_binned_image

EPSILON = 1e-16

def compute(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    device = image_tensor.device
    
    binned_image, Ng = get_binned_image(image_tensor, mask_tensor, settings.bin_width)
    if Ng == 0:
        return {}
        
    img_int = binned_image.to(torch.int64) * (mask_tensor > 0.5)
    
    M = mask_tensor > 0.5
    if not torch.any(M):
        return {}
        
    shifts = [
        (0, 0, 1), (0, 0, -1),
        (0, 1, 0), (0, -1, 0),
        (1, 0, 0), (-1, 0, 0),
        
        (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
        (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
        
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    
    D, H, W = img_int.shape
    dependence = torch.ones((D, H, W), dtype=torch.int64, device=device)
    
    for dz, dy, dx in shifts:
        z1_tgt, z_tgt_end = max(0, dz), min(D, D + dz)
        y1_tgt, y_tgt_end = max(0, dy), min(H, H + dy)
        x1_tgt, x_tgt_end = max(0, dx), min(W, W + dx)
        
        z1_src, z_src_end = max(0, -dz), min(D, D - dz)
        y1_src, y_src_end = max(0, -dy), min(H, H - dy)
        x1_src, x_src_end = max(0, -dx), min(W, W - dx)
        
        mask_match = (img_int[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end] == \
                      img_int[z1_src:z_src_end, y1_src:y_src_end, x1_src:x_src_end]) & \
                     (img_int[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end] > 0)
                     
        dependence[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end] += mask_match.to(torch.int64)
        
    valid_gray = img_int[M]
    valid_dep = dependence[M]
    
    Ng = int(torch.max(valid_gray).item())
    Nd = int(torch.max(valid_dep).item())
    
    if Ng == 0 or Nd == 0:
        return {}
        
    g = valid_gray - 1
    d = valid_dep - 1
    
    linear_indices = g * Nd + d
    matrix_counts = torch.bincount(linear_indices, minlength=Ng*Nd).to(torch.float64)
    P = matrix_counts[:Ng*Nd].view(Ng, Nd)
    
    Ns = P.sum()
    if Ns == 0:
        return {}
        
    i_grid = torch.arange(1, Ng + 1, dtype=torch.float64, device=device).view(-1, 1)
    j_grid = torch.arange(1, Nd + 1, dtype=torch.float64, device=device).view(1, -1)
    
    pg = torch.sum(P, dim=1).view(-1, 1)
    pd = torch.sum(P, dim=0).view(1, -1)
    
    sde = torch.sum(P / (j_grid ** 2)) / Ns
    lde = torch.sum(P * (j_grid ** 2)) / Ns
    
    glnu = torch.sum(pg ** 2) / Ns
    glnun = torch.sum(pg ** 2) / (Ns ** 2)
    
    dnu = torch.sum(pd ** 2) / Ns
    dnun = torch.sum(pd ** 2) / (Ns ** 2)
    
    p_norm = P / Ns
    mu_i = torch.sum(i_grid * p_norm)
    glv = torch.sum(((i_grid - mu_i) ** 2) * p_norm)
    
    mu_j = torch.sum(j_grid * p_norm)
    dv = torch.sum(((j_grid - mu_j) ** 2) * p_norm)
    
    p_log = torch.where(p_norm > EPSILON, torch.log2(p_norm), torch.zeros_like(p_norm))
    de = -torch.sum(p_norm * p_log)
    
    lgle = torch.sum(pg / (i_grid ** 2)) / Ns
    hgle = torch.sum(pg * (i_grid ** 2)) / Ns
    
    sdlgle = torch.sum(P / ((i_grid ** 2) * (j_grid ** 2))) / Ns
    sdhgle = torch.sum(P * (i_grid ** 2) / (j_grid ** 2)) / Ns
    ldlgle = torch.sum(P * (j_grid ** 2) / (i_grid ** 2)) / Ns
    ldhgle = torch.sum(P * (i_grid ** 2) * (j_grid ** 2)) / Ns
    
    features = {
        "gldm:small_dependence_emphasis": sde.item(),
        "gldm:large_dependence_emphasis": lde.item(),
        "gldm:gray_level_non_uniformity": glnu.item(),
        "gldm:gray_level_non_uniformity_normalized": glnun.item(),
        "gldm:dependence_non_uniformity": dnu.item(),
        "gldm:dependence_non_uniformity_normalized": dnun.item(),
        "gldm:dependence_variance": dv.item(),
        "gldm:gray_level_variance": glv.item(),
        "gldm:dependence_entropy": de.item(),
        "gldm:low_gray_level_emphasis": lgle.item(),
        "gldm:high_gray_level_emphasis": hgle.item(),
        "gldm:small_dependence_low_gray_level_emphasis": sdlgle.item(),
        "gldm:small_dependence_high_gray_level_emphasis": sdhgle.item(),
        "gldm:large_dependence_low_gray_level_emphasis": ldlgle.item(),
        "gldm:large_dependence_high_gray_level_emphasis": ldhgle.item()
    }
    
    return features
