import torch
from fastrad.settings import FeatureSettings

EPSILON = 1e-16

def compute(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    device = image_tensor.device
    
    voxels = image_tensor[mask_tensor > 0.5]
    if voxels.numel() == 0:
        return {}
        
    bin_width = settings.bin_width
    img_min = torch.min(image_tensor)
    minimum_binned = torch.floor(img_min / bin_width) * bin_width
    
    binned_image = torch.floor((image_tensor - minimum_binned) / bin_width) + 1
    
    M = mask_tensor > 0.5
    img_int = binned_image.to(torch.int64) * M
    
    if int(torch.max(img_int).item()) == 0:
        return {}
        
    D, H, W = img_int.shape
    
    labels = torch.arange(1, M.numel() + 1, device=device).view(M.shape)
    labels = labels * M
    
    shifts = [
        (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1),
        (1, 0, 0), (1, 0, 1), (1, 0, -1),
        (1, 1, 0), (1, 1, 1), (1, 1, -1),
        (1, -1, 0), (1, -1, 1), (1, -1, -1)
    ]
    
    changed = True
    while changed:
        old_labels = labels.clone()
        
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
                         
            labels_tgt = labels[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end]
            labels_src = labels[z1_src:z_src_end, y1_src:y_src_end, x1_src:x_src_end]
            
            max_vals = torch.where(mask_match, torch.max(labels_tgt, labels_src), labels_tgt)
            max_vals_src = torch.where(mask_match, torch.max(labels_tgt, labels_src), labels_src)
            
            labels[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end] = torch.max(labels_tgt, max_vals)
            labels[z1_src:z_src_end, y1_src:y_src_end, x1_src:x_src_end] = torch.max(labels_src, max_vals_src)
            
        changed = not torch.equal(labels, old_labels)
        
    valid_labels = labels[M]
    valid_gray = img_int[M]
    
    unique_labels, inverse_indices, counts = torch.unique(valid_labels, return_inverse=True, return_counts=True)
    
    N_c = unique_labels.numel()
    if N_c == 0:
        return {}
        
    comp_gray = torch.zeros(N_c, dtype=img_int.dtype, device=device)
    comp_gray.scatter_(0, inverse_indices, valid_gray)
    
    Ng = int(torch.max(comp_gray).item())
    Nz = int(torch.max(counts).item())
    
    if Ng == 0 or Nz == 0:
        return {}
        
    g = comp_gray - 1
    s = counts - 1
    
    linear_indices = g * Nz + s
    matrix_counts = torch.bincount(linear_indices, minlength=Ng*Nz).to(torch.float64)
    P = matrix_counts[:Ng*Nz].view(Ng, Nz)
    
    Ns = P.sum()
    if Ns == 0:
        return {}
        
    Np = voxels.numel()
    
    i_grid = torch.arange(1, Ng + 1, dtype=torch.float64, device=device).view(-1, 1)
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
