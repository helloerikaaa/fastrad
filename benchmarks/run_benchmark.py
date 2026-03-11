import time
import torch
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor

import argparse
from pathlib import Path

def create_spherical_mask(image_tensor: torch.Tensor, radius_mm: float, spacing: tuple[float, float, float]) -> torch.Tensor:
    """Create a spherical mask in the center of the image, sized in millimeters."""
    D, H, W = image_tensor.shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    
    # Calculate radius in voxels for each dimension
    r_z = max(1, int(radius_mm / spacing[0]))
    r_y = max(1, int(radius_mm / spacing[1]))
    r_x = max(1, int(radius_mm / spacing[2]))
    
    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    
    # Ellipsoid distance
    dist_sq = ((z - center_z) / r_z)**2 + ((y - center_y) / r_y)**2 + ((x - center_x) / r_x)**2
    
    mask = torch.zeros_like(image_tensor, dtype=torch.float32)
    mask[dist_sq <= 1.0] = 1.0
    return mask

def run_benchmark(image_dir: str, args):
    print(f"Loading DICOM series from {image_dir}...")
    
    # Load via fastrad
    fastrad_image = MedicalImage.from_dicom(image_dir)
    img_t = fastrad_image.tensor
    spacing = fastrad_image.spacing
    
    print(f"Image shape: {img_t.shape}, Spacing: {spacing}")
    print("Generating 15mm radius synthetic tumor mask...")
    mask_t = create_spherical_mask(img_t, radius_mm=15.0, spacing=spacing)
    fastrad_mask = Mask(mask_t, spacing=spacing)
    
    # Convert exactly to what PyRadiomics expects
    image_vol = img_t.numpy()
    mask_vol = mask_t.numpy().astype(np.uint8)
    
    sitk_image = sitk.GetImageFromArray(image_vol)
    sitk_image.SetSpacing(spacing[::-1])  # PyRadiomics spacing is (X,Y,Z)
    sitk_mask = sitk.GetImageFromArray(mask_vol)
    sitk_mask.SetSpacing(spacing[::-1])
    
    feature_classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    
    print("\n--- PyRadiomics Benchmark ---")
    pyrad_times = {}
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['binWidth'] = 25.0
    
    for cls in feature_classes:
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName(cls)
        t0 = time.time()
        res = extractor.execute(sitk_image, sitk_mask)
        t1 = time.time()
        pyrad_times[cls] = t1 - t0
        print(f"PyRadiomics {cls:<15}: {pyrad_times[cls]:.4f} seconds")
        
    pyrad_total = sum(pyrad_times.values())
    print(f"PyRadiomics TOTAL       : {pyrad_total:.4f} seconds")
    
    # 2. Fastrad setup (1 Thread)
    print("\n--- Fastrad Benchmark (CPU, 1 Thread) ---")
    torch.set_num_threads(1)
    
    fastrad_times_1t = {}
    
    for cls in feature_classes:
        settings = FeatureSettings(feature_classes=[cls], bin_width=25.0, device="cpu")
        extractor = FeatureExtractor(settings)
        
        t0 = time.time()
        res = extractor.extract(fastrad_image, fastrad_mask)
        t1 = time.time()
        fastrad_times_1t[cls] = t1 - t0
        print(f"Fastrad CPU (1t) {cls:<10}: {fastrad_times_1t[cls]:.4f} seconds")
        
    fastrad_total_1t = sum(fastrad_times_1t.values())
    print(f"Fastrad CPU (1t) TOTAL       : {fastrad_total_1t:.4f} seconds")
    
    speedup_cpu_1t = pyrad_total / fastrad_total_1t
    print(f"\nOverall Speedup (CPU, 1t)   : {speedup_cpu_1t:.2f}x")
    
    # 3. Fastrad setup (N Threads)
    # Determine thread count
    num_threads = args.threads if hasattr(args, 'threads') and args.threads else torch.get_num_threads()
    print(f"\n--- Fastrad Benchmark (CPU, {num_threads} Threads) ---")
    torch.set_num_threads(num_threads)
    
    fastrad_times_nt = {}
    
    for cls in feature_classes:
        settings = FeatureSettings(feature_classes=[cls], bin_width=25.0, device="cpu")
        extractor = FeatureExtractor(settings)
        
        t0 = time.time()
        res = extractor.extract(fastrad_image, fastrad_mask)
        t1 = time.time()
        fastrad_times_nt[cls] = t1 - t0
        print(f"Fastrad CPU ({num_threads}t) {cls:<10}: {fastrad_times_nt[cls]:.4f} seconds")
        
    fastrad_total_nt = sum(fastrad_times_nt.values())
    print(f"Fastrad CPU ({num_threads}t) TOTAL       : {fastrad_total_nt:.4f} seconds")
    
    speedup_cpu_nt = pyrad_total / fastrad_total_nt
    print(f"\nOverall Speedup (CPU, {num_threads}t)   : {speedup_cpu_nt:.2f}x")
    
    # Reset to default
    torch.set_num_threads(num_threads)
    
    # 4. Fastrad setup (CUDA)
    if torch.cuda.is_available():
        device_str = "cuda"
        print(f"\n--- Fastrad Benchmark ({device_str.upper()}) ---")
        
        fastrad_gpu_times = {}
        for cls in feature_classes:
            settings = FeatureSettings(feature_classes=[cls], bin_width=25.0, device=device_str)
            extractor = FeatureExtractor(settings)
            
            # warmup pass
            _ = extractor.extract(fastrad_image, fastrad_mask)
            torch.cuda.synchronize()
                
            t0 = time.time()
            res = extractor.extract(fastrad_image, fastrad_mask)
            torch.cuda.synchronize()
            t1 = time.time()
            fastrad_gpu_times[cls] = t1 - t0
            print(f"Fastrad GPU {cls:<15}: {fastrad_gpu_times[cls]:.4f} seconds")
            
        fastrad_gpu_total = sum(fastrad_gpu_times.values())
        print(f"Fastrad GPU TOTAL       : {fastrad_gpu_total:.4f} seconds")
        speedup_gpu = pyrad_total / fastrad_gpu_total
        print(f"\nOverall Speedup (GPU)   : {speedup_gpu:.2f}x")
    else:
        print("\nSkipping GPU benchmark (no CUDA available). Note: MPS does not support float64.")

if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    parser = argparse.ArgumentParser(description="Run fastrad benchmark on a clinical DICOM dataset.")
    parser.add_argument("--image-dir", type=str, help="Path to DICOM series directory",
                        default=str(Path(project_root) / "tests" / "fixtures" / "tcia" / "images"))
    parser.add_argument("--threads", type=int, help="Number of CPU threads to use for PyTorch multi-threading benchmark (default: max available)", default=0)
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir) or not os.listdir(args.image_dir):
        print(f"Error: Could not find DICOM files in {args.image_dir}")
        print("Please run 'python benchmarks/download_tcia_sample.py' first.")
        sys.exit(1)
        
    run_benchmark(args.image_dir, args)
