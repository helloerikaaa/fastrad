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
    
    neighbor_sums = torch.zeros((D, H, W), dtype=torch.float64, device=device)
    neighbor_counts = torch.zeros((D, H, W), dtype=torch.int64, device=device)
    
    img_float = img_int.to(torch.float64)
    
    for dz, dy, dx in shifts:
        z1_tgt, z_tgt_end = max(0, dz), min(D, D + dz)
        y1_tgt, y_tgt_end = max(0, dy), min(H, H + dy)
        x1_tgt, x_tgt_end = max(0, dx), min(W, W + dx)
        
        z1_src, z_src_end = max(0, -dz), min(D, D - dz)
        y1_src, y_src_end = max(0, -dy), min(H, H - dy)
        x1_src, x_src_end = max(0, -dx), min(W, W - dx)
        
        mask_src = M[z1_src:z_src_end, y1_src:y_src_end, x1_src:x_src_end]
        
        neighbor_sums[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end] += \
            img_float[z1_src:z_src_end, y1_src:y_src_end, x1_src:x_src_end] * mask_src
            
        neighbor_counts[z1_tgt:z_tgt_end, y1_tgt:y_tgt_end, x1_tgt:x_tgt_end] += mask_src.to(torch.int64)
        
    s_i_voxel = torch.zeros_like(img_float)
    valid_neighbors = neighbor_counts > 0
    s_i_voxel[valid_neighbors] = torch.abs(img_float[valid_neighbors] - (neighbor_sums[valid_neighbors] / neighbor_counts[valid_neighbors].to(torch.float64)))
    
    valid_gray = img_int[M & (neighbor_counts > 0)]
    s_i_valid = s_i_voxel[M & (neighbor_counts > 0)]
    
    if valid_gray.numel() == 0:
        return {}
        
    Ng = int(torch.max(valid_gray).item())
    
    if Ng == 0:
        return {}
        
    n_i = torch.bincount(valid_gray, minlength=Ng+1).to(torch.float64)
    s_i = torch.bincount(valid_gray, weights=s_i_valid, minlength=Ng+1).to(torch.float64)
    
    valid_mask = n_i > 0
    ivector = torch.arange(0, Ng + 1, dtype=torch.float64, device=device)
    
    n_i = n_i[valid_mask]
    s_i = s_i[valid_mask]
    ivector = ivector[valid_mask]
    
    # Wait, does pyradiomics compute Nvp as sum of voxels with AT LEAST ONE neighbor?
    # Actually, as per PyRadiomics comment: "Nvp is the sum of n_i (i.e. the number of voxels with a valid region; at least 1 neighbor)."
    # Let's count Nvp as sum of voxels with at least 1 neighbor.
    # The Matrix 1 output we saw `[[ 1 0 1 ]]` means `n_i` is NOT voxels with >0 neighbors, 
    # but `p_i` uses `Nvp`! Oh, for Matrix 1, Nvp was 1... but W=0.
    # If Nvp = sum(n_i with >0 neighbors), for Matrix 1 it would be 0, dividing by 0.
    # Actually, PyRadiomics calculates W for each voxel. If W=0, the voxel might just be excluded from N_i if strictly reading the comment.
    # But Matrix 1 had `1` in col 0. Let's just use Nvp = sum(n_i), which is M.sum().
    
    # Wait, the comment says `Nvp is the sum of n_i (i.e. the number of voxels with a valid region; at least 1 neighbor)`.
    # And then `p_i = n_i / Nvp`.
    # Let's see if we should use `M.sum()` or `n_neighbors.sum()`.
    # Let's use `M.sum()`.
    Nvp = torch.sum(n_i)
    if Nvp == 0:
        return {}
        
    p_i = n_i / Nvp
    Ngp = p_i.numel()
    
    # Coarseness
    sum_coarse = torch.sum(p_i * s_i)
    coarseness = 1.0 / sum_coarse if sum_coarse > 0 else torch.tensor(1e6, dtype=torch.float64, device=device)
    
    # Contrast
    i_mat = ivector.view(-1, 1) - ivector.view(1, -1)
    pi_pj = p_i.view(-1, 1) * p_i.view(1, -1)
    
    if Ngp > 1:
        contrast = torch.sum(pi_pj * (i_mat ** 2)) * torch.sum(s_i) / (Nvp * Ngp * (Ngp - 1))
    else:
        contrast = torch.tensor(0.0, dtype=torch.float64, device=device)
        
    # Busyness
    # PyRadiomics definition: sum(p_i * s_i) / sum_i(sum_j(|i*p_i - j*p_j|)) where p_i != 0 and p_j != 0
    pi_pj_mask = (p_i.view(-1, 1) > 0) & (p_i.view(1, -1) > 0)
    
    i_pi = ivector * p_i
    ipi_jpj = torch.abs(i_pi.view(-1, 1) - i_pi.view(1, -1))
    
    # Apply mask where p_i != 0 and p_j != 0
    ipi_jpj = torch.where(pi_pj_mask, ipi_jpj, torch.zeros_like(ipi_jpj))
    
    sum_absdiff = torch.sum(ipi_jpj)
    
    busyness = torch.sum(p_i * s_i) / sum_absdiff if sum_absdiff > 0 else torch.tensor(0.0, dtype=torch.float64, device=device)
    
    # Complexity
    i_minus_j = torch.abs(i_mat)
    num = p_i.view(-1, 1) * s_i.view(-1, 1) + p_i.view(1, -1) * s_i.view(1, -1)
    den = p_i.view(-1, 1) + p_i.view(1, -1)
    
    # avoiding divide by zero
    mask_den = den > 0
    complexity = torch.sum(torch.where(mask_den, i_minus_j * num / den, torch.zeros_like(den))) / Nvp
    
    # Strength
    num_str = (p_i.view(-1, 1) + p_i.view(1, -1)) * (i_mat ** 2)
    sum_str = torch.sum(num_str)
    sum_s_i = torch.sum(s_i)
    strength = sum_str / sum_s_i if sum_s_i > 0 else torch.tensor(0.0, dtype=torch.float64, device=device)
    
    features = {
        "ngtdm:coarseness": coarseness.item(),
        "ngtdm:contrast": contrast.item(),
        "ngtdm:busyness": busyness.item(),
        "ngtdm:complexity": complexity.item(),
        "ngtdm:strength": strength.item()
    }
    
    return features
