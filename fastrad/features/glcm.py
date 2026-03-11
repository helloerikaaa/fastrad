import torch
import math
from fastrad.settings import FeatureSettings
from fastrad.image import get_binned_image

EPSILON = 1e-16

def compute(image_tensor: torch.Tensor, mask_tensor: torch.Tensor, settings: FeatureSettings) -> dict[str, float]:
    device = image_tensor.device
    
    binned_image, ivector = get_binned_image(image_tensor, mask_tensor, settings.bin_width)
    Ng = ivector.numel()
    if Ng == 0:
        return {}
    
    max_gl = int(torch.max(ivector).item())
        
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
    
    P_glcm_raw = torch.zeros((max_gl, max_gl, len(angles)), dtype=torch.float64, device=device)
    
    img_int = binned_image.to(torch.int64) * (mask_tensor > 0.5)
    
    for a_idx, (dz, dy, dx) in enumerate(angles):
        z, y, x = torch.nonzero(mask_tensor > 0.5, as_tuple=True)
        
        z_shift = z + dz
        y_shift = y + dy
        x_shift = x + dx
        
        D, H, W = image_tensor.shape
        valid = (z_shift >= 0) & (z_shift < D) & \
                (y_shift >= 0) & (y_shift < H) & \
                (x_shift >= 0) & (x_shift < W)
                
        z_v, y_v, x_v = z[valid], y[valid], x[valid]
        zs_v, ys_v, xs_v = z_shift[valid], y_shift[valid], x_shift[valid]
        
        mask_valid = mask_tensor[zs_v, ys_v, xs_v] > 0.5
        
        z_v = z_v[mask_valid]
        y_v = y_v[mask_valid]
        x_v = x_v[mask_valid]
        zs_v = zs_v[mask_valid]
        ys_v = ys_v[mask_valid]
        xs_v = xs_v[mask_valid]
        
        if z_v.numel() == 0:
            continue
            
        val1 = img_int[z_v, y_v, x_v] - 1
        val2 = img_int[zs_v, ys_v, xs_v] - 1
        
        linear_indices = val1 * max_gl + val2
        counts = torch.bincount(linear_indices, minlength=max_gl*max_gl).to(torch.float64)
        counts = counts[:max_gl*max_gl].view(max_gl, max_gl)
        counts_sym = counts + counts.T
        P_glcm_raw[:, :, a_idx] = counts_sym
        
    valid_idx = (ivector - 1).to(torch.int64)
    P_glcm = P_glcm_raw[valid_idx][:, valid_idx]
    
    sums = torch.sum(P_glcm, dim=(0, 1))
    valid_angles = sums > 0
    if not torch.any(valid_angles):
        return {}
        
    P_glcm = P_glcm[:, :, valid_angles]
    sums = sums[valid_angles]
    
    p = P_glcm / sums.view(1, 1, -1)
    
    N_a = p.shape[2]
    
    px = torch.sum(p, dim=1)
    py = torch.sum(p, dim=0)
    
    i_grid = ivector.view(-1, 1, 1)
    j_grid = ivector.view(1, -1, 1)
    
    i_val = ivector.view(-1, 1)
    
    ux = torch.sum(i_grid * p, dim=(0, 1))
    uy = torch.sum(j_grid * p, dim=(0, 1))
    
    i_plus_j = i_grid + j_grid
    i_minus_j = torch.abs(i_grid - j_grid)
    
    diff_ij = i_grid - j_grid
    
    contrast = torch.mean(torch.sum((diff_ij ** 2) * p, dim=(0,1)))
    
    diff_avg = torch.mean(torch.sum(i_minus_j * p, dim=(0,1)))
    
    ux_m = torch.sum(i_val * px, dim=0)
    uy_m = torch.sum(i_val * py, dim=0)
    
    var_x = torch.sum(((i_val - ux_m.view(1, -1)) ** 2) * px, dim=0)
    var_y = torch.sum(((i_val - uy_m.view(1, -1)) ** 2) * py, dim=0)
    
    sigmax = torch.sqrt(var_x)
    sigmay = torch.sqrt(var_y)
    
    corm = torch.sum((i_grid - ux_m.view(1, 1, -1)) * (j_grid - uy_m.view(1, 1, -1)) * p, dim=(0, 1))
    corr = corm / (sigmax * sigmay + EPSILON)
    corr[(sigmax * sigmay) == 0] = 1.0
    correlation = torch.mean(corr)
    
    joint_average = torch.mean(ux)
    
    joint_energy = torch.mean(torch.sum(p ** 2, dim=(0, 1)))
    
    p_log = torch.where(p > EPSILON, torch.log2(p), torch.zeros_like(p))
    joint_entropy = torch.mean(-torch.sum(p * p_log, dim=(0, 1)))
    
    # Needs to match PyRadiomics exactly. i_grid is (Ng, 1, 1). j_grid is (1, Ng, 1). p is (Ng, Ng, N_a).
    ac_per_angle = torch.sum(i_grid * j_grid * p, dim=(0, 1))
    autocorrelation = torch.mean(ac_per_angle)
    
    cluster_prominence = torch.mean(torch.sum(((i_grid + j_grid - ux.view(1, 1, -1) - uy.view(1, 1, -1)) ** 4) * p, dim=(0, 1)))
    
    cluster_shade = torch.mean(torch.sum(((i_grid + j_grid - ux.view(1, 1, -1) - uy.view(1, 1, -1)) ** 3) * p, dim=(0, 1)))
    
    cluster_tendency = torch.mean(torch.sum(((i_grid + j_grid - ux.view(1, 1, -1) - uy.view(1, 1, -1)) ** 2) * p, dim=(0, 1)))
    
    max_prob = torch.mean(torch.amax(p, dim=(0, 1)))
    
    sum_squares = torch.mean(torch.sum(((i_grid - ux.view(1, 1, -1)) ** 2) * p, dim=(0, 1)))
    
    id_ = torch.mean(torch.sum(p / (1 + i_minus_j), dim=(0, 1)))
    
    idm = torch.mean(torch.sum(p / (1 + i_minus_j ** 2), dim=(0, 1)))
    
    idn = torch.mean(torch.sum(p / (1 + i_minus_j / max_gl), dim=(0, 1)))
    
    idmn = torch.mean(torch.sum(p / (1 + (i_minus_j ** 2) / (max_gl ** 2)), dim=(0, 1)))
    
    mask_ij = i_minus_j > 0
    inv_var = torch.mean(torch.sum(torch.where(mask_ij, p / (i_minus_j ** 2), torch.zeros_like(p)), dim=(0, 1)))
    
    sum_average = torch.mean(torch.sum(i_plus_j * p, dim=(0, 1)))
    
    p_x_plus_y = torch.zeros((2 * max_gl + 1, N_a), dtype=torch.float64, device=device)
    p_x_minus_y = torch.zeros((max_gl + 1, N_a), dtype=torch.float64, device=device)
    
    flat_p = p.reshape(-1, N_a)
    flat_plus = i_plus_j.expand(-1, -1, N_a).to(torch.int64).reshape(-1, N_a)
    flat_minus = i_minus_j.expand(-1, -1, N_a).to(torch.int64).reshape(-1, N_a)
    
    p_x_plus_y.scatter_add_(0, flat_plus, flat_p)
    p_x_minus_y.scatter_add_(0, flat_minus, flat_p)
    
    diff_avg_a = torch.sum(i_minus_j * p, dim=(0,1))
    
    k_minus = torch.arange(0, max_gl + 1, dtype=torch.float64, device=device).view(-1, 1)
    diff_var = torch.mean(torch.sum(((k_minus - diff_avg_a.view(1, -1)) ** 2) * p_x_minus_y, dim=0))
    
    p_x_minus_y_log = torch.where(p_x_minus_y > EPSILON, torch.log2(p_x_minus_y), torch.zeros_like(p_x_minus_y))
    diff_entropy = torch.mean(-torch.sum(p_x_minus_y * p_x_minus_y_log, dim=0))
    
    p_x_plus_y_log = torch.where(p_x_plus_y > EPSILON, torch.log2(p_x_plus_y), torch.zeros_like(p_x_plus_y))
    sum_entropy = torch.mean(-torch.sum(p_x_plus_y * p_x_plus_y_log, dim=0))
    
    px_py = px.view(Ng, 1, -1) * py.view(1, Ng, -1)
    px_py_log = torch.where(px_py > EPSILON, torch.log2(px_py), torch.zeros_like(px_py))
    
    HXY1 = -torch.sum(p * px_py_log, dim=(0, 1))
    HXY2 = -torch.sum(px_py * px_py_log, dim=(0, 1))
    
    px_log = torch.where(px > EPSILON, torch.log2(px), torch.zeros_like(px))
    py_log = torch.where(py > EPSILON, torch.log2(py), torch.zeros_like(py))
    HX = -torch.sum(px * px_log, dim=0)
    HY = -torch.sum(py * py_log, dim=0)
    
    HXY = -torch.sum(p * p_log, dim=(0, 1))
    
    div = torch.max(HX, HY)
    imc1_vals = torch.where(div > EPSILON, (HXY - HXY1) / div, torch.zeros_like(HXY))
    imc1 = torch.mean(imc1_vals)
    imc2 = torch.mean(torch.sqrt(torch.clamp(1 - torch.exp(-2.0 * (HXY2 - HXY)), min=0.0)))
    
    mcc_list = []
    for a in range(N_a):
        p_a = p[:,:,a]
        px_a = px[:,a].view(-1, 1)
        py_a = py[:,a].view(1, -1)
        
        mask_x = px_a.squeeze() > EPSILON
        
        valid_idx = mask_x.nonzero().squeeze(-1)
        if valid_idx.numel() < 2:
            mcc_list.append(1.0)
            continue
            
        p_valid = p_a[valid_idx][:, valid_idx]
        px_valid = px_a[valid_idx]
        
        A = p_valid / px_valid
        Q = torch.matmul(A, A)
        
        try:
            evs = torch.linalg.eigvals(Q).real
            evs = torch.sort(evs, descending=True).values
            if len(evs) >= 2:
                mcc_list.append(math.sqrt(evs[1].item()))
            else:
                mcc_list.append(1.0)
        except:
            mcc_list.append(1.0)
            
    mcc = sum(mcc_list) / len(mcc_list) if mcc_list else 1.0
    
    features = {
        "glcm:contrast": contrast.item(),
        "glcm:difference_average": diff_avg.item(),
        "glcm:correlation": correlation.item(),
        "glcm:joint_average": joint_average.item(),
        "glcm:joint_energy": joint_energy.item(),
        "glcm:joint_entropy": joint_entropy.item(),
        "glcm:autocorrelation": autocorrelation.item(),
        "glcm:cluster_prominence": cluster_prominence.item(),
        "glcm:cluster_shade": cluster_shade.item(),
        "glcm:cluster_tendency": cluster_tendency.item(),
        "glcm:maximum_probability": max_prob.item(),
        "glcm:sum_squares": sum_squares.item(),
        "glcm:id": id_.item(),
        "glcm:idm": idm.item(),
        "glcm:idn": idn.item(),
        "glcm:idmn": idmn.item(),
        "glcm:inverse_variance": inv_var.item(),
        "glcm:sum_average": sum_average.item(),
        "glcm:difference_variance": diff_var.item(),
        "glcm:difference_entropy": diff_entropy.item(),
        "glcm:sum_entropy": sum_entropy.item(),
        "glcm:imc1": imc1.item(),
        "glcm:imc2": imc2.item(),
        "glcm:mcc": mcc,
    }
    
    return features
