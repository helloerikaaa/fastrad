import time
import torch
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor

def generate_large_fixture(size=64):
    np.random.seed(42)
    image_vol = np.random.randint(0, 255, size=(size, size, size), dtype=np.uint16)
    mask_vol = np.zeros((size, size, size), dtype=np.uint8)
    
    # Create a sphere mask
    center = size // 2
    radius = size // 4
    z, y, x = np.ogrid[:size, :size, :size]
    dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
    mask_vol[dist_sq <= radius**2] = 1
    
    return image_vol, mask_vol

def run_benchmark():
    print("Generating benchmark data (64x64x64 volume)...")
    image_vol, mask_vol = generate_large_fixture(64)
    
    # 1. PyRadiomics setup
    sitk_image = sitk.GetImageFromArray(image_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask = sitk.GetImageFromArray(mask_vol)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))
    
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
    
    # 2. Fastrad setup
    print("\n--- Fastrad Benchmark (CPU) ---")
    img_t = torch.from_numpy(image_vol.astype(np.float32))
    mask_t = torch.from_numpy(mask_vol.astype(np.float32))
    
    fastrad_times = {}
    
    # Since fastrad has extractor designed for MedicalImage, we can modify it or bypass it
    # We will test the extractor.py interface directly if it supports MedicalImage objects
    # But MedicalImage loads from DICOM path. We can mock it or use the components directly.
    # To be precise, let's use the individual feature modules directly or mock MedicalImage.
    from fastrad.features import firstorder, shape, glcm, glrlm, glszm, gldm, ngtdm
    modules = {
        'firstorder': firstorder,
        'shape': shape,
        'glcm': glcm,
        'glrlm': glrlm,
        'glszm': glszm,
        'gldm': gldm,
        'ngtdm': ngtdm
    }
    
    for cls in feature_classes:
        module = modules[cls]
        settings = FeatureSettings(feature_classes=[cls], bin_width=25.0, device="cpu", spacing=(1.0, 1.0, 1.0))
        t0 = time.time()
        res = module.compute(img_t, mask_t, settings)
        t1 = time.time()
        fastrad_times[cls] = t1 - t0
        print(f"Fastrad CPU {cls:<15}: {fastrad_times[cls]:.4f} seconds")
        
    fastrad_total = sum(fastrad_times.values())
    print(f"Fastrad CPU TOTAL       : {fastrad_total:.4f} seconds")
    
    speedup_cpu = pyrad_total / fastrad_total
    print(f"\nOverall Speedup (CPU)   : {speedup_cpu:.2f}x")
    
    # 3. Fastrad setup (CUDA)
    if torch.cuda.is_available():
        device_str = "cuda"
        print(f"\n--- Fastrad Benchmark ({device_str.upper()}) ---")
        img_device = img_t.to(device_str)
        mask_device = mask_t.to(device_str)
        
        fastrad_gpu_times = {}
        for cls in feature_classes:
            module = modules[cls]
            settings = FeatureSettings(feature_classes=[cls], bin_width=25.0, device=device_str, spacing=(1.0, 1.0, 1.0))
            
            # warmup pass
            _ = module.compute(img_device, mask_device, settings)
            torch.cuda.synchronize()
                
            t0 = time.time()
            res = module.compute(img_device, mask_device, settings)
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
    run_benchmark()
