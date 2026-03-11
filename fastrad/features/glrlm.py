import torch
from fastrad.settings import FeatureSettings
from fastrad.image import get_binned_image

EPSILON = 1e-16

def compute(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    device = image_tensor.device
    
    binned_image, Ng = get_binned_image(image_tensor, mask_tensor, settings.bin_width)
    if Ng == 0:
        return {}
        
    voxels = image_tensor[mask_tensor > 0.5]
        
    angles = [
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, -1),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, -1, -1)
    ]
    
    img_int = binned_image.to(torch.int64)
    M = mask_tensor > 0.5
    D, H, W = img_int.shape
    
    angle_results = []
    max_length = 0
    
    for a_idx, (dz, dy, dx) in enumerate(angles):
        z, y, x = torch.nonzero(M, as_tuple=True)
        
        zb, yb, xb = z - dz, y - dy, x - dx
        
        valid_back = (zb >= 0) & (zb < D) & (yb >= 0) & (yb < H) & (xb >= 0) & (xb < W)
        
        is_start = torch.ones_like(z, dtype=torch.bool)
        
        z_valid_back = zb[valid_back]
        y_valid_back = yb[valid_back]
        x_valid_back = xb[valid_back]
        
        match_back = M[z_valid_back, y_valid_back, x_valid_back] & \
                     (img_int[z_valid_back, y_valid_back, x_valid_back] == img_int[z[valid_back], y[valid_back], x[valid_back]])
                     
        is_start[valid_back] = ~match_back
        
        start_z = z[is_start]
        start_y = y[is_start]
        start_x = x[is_start]
        
        lengths = torch.ones_like(start_z)
        gray_levels = img_int[start_z, start_y, start_x]
        
        curr_z, curr_y, curr_x = start_z, start_y, start_x
        active = torch.ones_like(start_z, dtype=torch.bool)
        
        while active.any():
            curr_z = curr_z + dz
            curr_y = curr_y + dy
            curr_x = curr_x + dx
            
            in_bounds = (curr_z >= 0) & (curr_z < D) & (curr_y >= 0) & (curr_y < H) & (curr_x >= 0) & (curr_x < W)
            active = active & in_bounds
            
            if not active.any():
                break
                
            curr_z_act = curr_z[active]
            curr_y_act = curr_y[active]
            curr_x_act = curr_x[active]
            
            match_fwd = M[curr_z_act, curr_y_act, curr_x_act] & \
                        (img_int[curr_z_act, curr_y_act, curr_x_act] == gray_levels[active])
                        
            active_flat = active.clone()
            active_flat[active] = match_fwd
            active = active_flat
            
            lengths[active] += 1

        if lengths.numel() > 0:
            m_len = torch.max(lengths).item()
            if m_len > max_length:
                max_length = int(m_len)
                
        angle_results.append((gray_levels, lengths))
        
    Nr = max_length
    if Nr == 0:
        return {}
        
    N_a = len(angles)
    P_glrlm = torch.zeros((Ng, Nr, N_a), dtype=torch.float64, device=device)
    
    for a_idx, (gray_levels, lengths) in enumerate(angle_results):
        if gray_levels.numel() == 0:
            continue
            
        g = gray_levels - 1
        l = lengths - 1
        
        linear_indices = g * Nr + l
        counts = torch.bincount(linear_indices, minlength=Ng*Nr).to(torch.float64)
        counts = counts[:Ng*Nr].view(Ng, Nr)
        
        P_glrlm[:, :, a_idx] = counts
        
    sums = torch.sum(P_glrlm, dim=(0, 1))
    valid_angles = sums > 0
    if not torch.any(valid_angles):
        return {}
        
    P_glrlm = P_glrlm[:, :, valid_angles]
    sums = sums[valid_angles]
    
    N_a = P_glrlm.shape[2]
    
    Np = voxels.numel()
    
    P = P_glrlm
    Ns = sums
    
    i_grid = torch.arange(1, Ng + 1, dtype=torch.float64, device=device).view(-1, 1, 1)
    j_grid = torch.arange(1, Nr + 1, dtype=torch.float64, device=device).view(1, -1, 1)
    
    pg = torch.sum(P, dim=1)
    pr = torch.sum(P, dim=0)
    
    sre = torch.mean(torch.sum(pr / (j_grid.view(-1, 1) ** 2), dim=0) / Ns)
    
    lre = torch.mean(torch.sum(pr * (j_grid.view(-1, 1) ** 2), dim=0) / Ns)
    
    glnu = torch.mean(torch.sum(pg ** 2, dim=0) / Ns)
    
    glnun = torch.mean(torch.sum(pg ** 2, dim=0) / (Ns ** 2))
    
    rlnu = torch.mean(torch.sum(pr ** 2, dim=0) / Ns)
    
    rlnun = torch.mean(torch.sum(pr ** 2, dim=0) / (Ns ** 2))
    
    rp = torch.mean(Ns / Np)
    
    p_norm = P / Ns.view(1, 1, -1)
    mu_i = torch.sum(i_grid * p_norm, dim=(0, 1))
    glv = torch.mean(torch.sum(((i_grid - mu_i.view(1, 1, -1)) ** 2) * p_norm, dim=(0, 1)))
    
    mu_j = torch.sum(j_grid * p_norm, dim=(0, 1))
    rv = torch.mean(torch.sum(((j_grid - mu_j.view(1, 1, -1)) ** 2) * p_norm, dim=(0, 1)))
    
    p_log = torch.where(p_norm > EPSILON, torch.log2(p_norm), torch.zeros_like(p_norm))
    re_feat = torch.mean(-torch.sum(p_norm * p_log, dim=(0, 1)))
    
    lglre = torch.mean(torch.sum(pg / (i_grid.squeeze(-1) ** 2), dim=0) / Ns)
    
    hglre = torch.mean(torch.sum(pg * (i_grid.squeeze(-1) ** 2), dim=0) / Ns)
    
    srlgle = torch.mean(torch.sum(P / ((i_grid ** 2) * (j_grid ** 2)), dim=(0, 1)) / Ns)
    
    srhgle = torch.mean(torch.sum(P * (i_grid ** 2) / (j_grid ** 2), dim=(0, 1)) / Ns)
    
    lrlgle = torch.mean(torch.sum(P * (j_grid ** 2) / (i_grid ** 2), dim=(0, 1)) / Ns)
    
    lrhgle = torch.mean(torch.sum(P * (i_grid ** 2) * (j_grid ** 2), dim=(0, 1)) / Ns)
    
    features = {
        "glrlm:short_run_emphasis": sre.item(),
        "glrlm:long_run_emphasis": lre.item(),
        "glrlm:gray_level_non_uniformity": glnu.item(),
        "glrlm:gray_level_non_uniformity_normalized": glnun.item(),
        "glrlm:run_length_non_uniformity": rlnu.item(),
        "glrlm:run_length_non_uniformity_normalized": rlnun.item(),
        "glrlm:run_percentage": rp.item(),
        "glrlm:gray_level_variance": glv.item(),
        "glrlm:run_variance": rv.item(),
        "glrlm:run_entropy": re_feat.item(),
        "glrlm:low_gray_level_run_emphasis": lglre.item(),
        "glrlm:high_gray_level_run_emphasis": hglre.item(),
        "glrlm:short_run_low_gray_level_emphasis": srlgle.item(),
        "glrlm:short_run_high_gray_level_emphasis": srhgle.item(),
        "glrlm:long_run_low_gray_level_emphasis": lrlgle.item(),
        "glrlm:long_run_high_gray_level_emphasis": lrhgle.item()
    }
    
    return features
