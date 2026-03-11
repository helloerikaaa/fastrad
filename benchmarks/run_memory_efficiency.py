import tracemalloc
import torch
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor
from pathlib import Path
import os
import gc

def create_spherical_mask(image_shape, radius_mm: float, spacing: tuple[float, float, float]):
    D, H, W = image_shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))
    
    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_sq = ((z - center_z) / r_z)**2 + ((y - center_y) / r_y)**2 + ((x - center_x) / r_x)**2
    mask = torch.zeros(image_shape, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask

import subprocess
import sys

def measure_peak_ram_subprocess(lib_name: str, radius: float) -> float:
    script = f'''
import os
import sys
import gc
import resource
import torch
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor

def create_spherical_mask(image_shape, radius_mm: float, spacing: tuple):
    D, H, W = image_shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))
    
    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_sq = ((z - center_z) / r_z)**2 + ((y - center_y) / r_y)**2 + ((x - center_x) / r_x)**2
    mask = torch.zeros(image_shape, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask

project_root = "{Path(__file__).parent.parent.absolute()}"
img_path = os.path.join(project_root, "tests", "fixtures", "tcia", "lung1_image.nrrd")
img_sitk = sitk.ReadImage(str(img_path))
img_t = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).float()
spacing = img_sitk.GetSpacing()[::-1]

classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
mask_t = create_spherical_mask(img_t.shape, {radius}, spacing)

gc.collect()

if "{lib_name}" == "pyrad":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyrad_ext = featureextractor.RadiomicsFeatureExtractor()
        pyrad_ext.settings['binWidth'] = 25.0
        m_sitk = sitk.GetImageFromArray(mask_t.numpy().astype(np.uint8))
        m_sitk.CopyInformation(img_sitk)
        pyrad_ext.execute(img_sitk, m_sitk)
else:
    fastrad_img = MedicalImage(img_t, spacing=spacing)
    f_mask = Mask(mask_t, spacing=spacing)
    fastrad_settings = FeatureSettings(feature_classes=classes, bin_width=25.0, device="cpu")
    f_ext = FeatureExtractor(fastrad_settings)
    f_ext.extract(fastrad_img, f_mask)

maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform == "darwin":
    print(maxrss / (1024 * 1024))
else:
    print(maxrss / 1024)
'''
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0

def run():
    print("Running Memory Efficiency Benchmark...")
    project_root = Path(__file__).parent.parent
    img_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
    
    if not img_path.exists():
        return "Error: TCIA image not found."
        
    md = []
    md.append("## Section 4: Memory Efficiency\n")
    
    # 4.1 Peak RAM vs ROI size
    print("  -> Profiling CPU RAM scaling against ROI size (4.1)...")
    md.append("### 4.1 Peak RAM vs ROI Size (CPU)\n")
    md.append("| Radius (mm) | Voxel Count | PyRadiomics RAM (MB) | fastrad CPU RAM (MB) | Memory Reduction |")
    md.append("|---|---|---|---|---|")
    
    img_sitk = sitk.ReadImage(str(img_path))
    img_t = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).float()
    spacing = img_sitk.GetSpacing()[::-1]
    
    for r in [5, 10, 15, 20, 25, 30]:
        mask_t = create_spherical_mask(img_t.shape, r, spacing)
        n_voxels = int(mask_t.sum().item())
        
        pyrad_mem = measure_peak_ram_subprocess("pyrad", r)
        fastrad_mem = measure_peak_ram_subprocess("fastrad", r)
        
        reduction = pyrad_mem / fastrad_mem if fastrad_mem > 0 else 0
        md.append(f"| {r} | {n_voxels} | {pyrad_mem:.2f} | {fastrad_mem:.2f} | {reduction:.2f}x |")
        
    md.append("\n")
    
    # 4.2 GPU VRAM Profile
    print("  -> Profiling GPU VRAM per feature class (4.2)...")
    md.append("### 4.2 GPU VRAM Profile\n")
    if torch.cuda.is_available():
        mask_t = create_spherical_mask(img_t.shape, 15.0, spacing)
        f_mask = Mask(mask_t, spacing=spacing)
        
        fastrad_img = MedicalImage(img_t, spacing=spacing)
        fastrad_img = fastrad_img.to("cuda")
        f_mask = f_mask.to("cuda")
        
        md.append("| Feature Class | Peak VRAM Allocated (MB) |")
        md.append("|---|---|")
        
        def profile_vram(classes_to_run):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            f_ext = FeatureExtractor(FeatureSettings(feature_classes=classes_to_run, bin_width=25.0, device="cuda"))
            f_ext.extract(fastrad_img, f_mask)
            torch.cuda.synchronize()
            
            peak_bytes = torch.cuda.max_memory_allocated()
            return peak_bytes / (1024**2)
            
        for cls in classes:
            vram = profile_vram([cls])
            md.append(f"| {cls} | {vram:.2f} |")
            
        total_vram = profile_vram(classes)
        md.append(f"| **FULL PIPELINE** | **{total_vram:.2f}** |")
    else:
        md.append("*GPU VRAM profiling skipped (CUDA not available).*")
        
    md.append("\n\n")
    return "\n".join(md)

if __name__ == "__main__":
    print(run())
