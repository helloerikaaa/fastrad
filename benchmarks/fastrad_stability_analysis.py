import os
import time
import torch
import radiomics
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor
from benchmarks.perturbation import apply_perturbation
import SimpleITK as sitk

def run_stability_analysis():
    print("--- Feature Stability Analysis (Test-Retest Simulation) ---")
    
    # Ensure pyradiomics doesn't print to stdout excessively
    radiomics.setVerbosity(60)
    
    # 1. Load Baseline Data
    # We use the dataset downloaded in Milestone 22
    tcia_path = "tests/fixtures/tcia/images"
    if not os.path.exists(tcia_path):
        print(f"Error: {tcia_path} not found. Please run download_tcia_sample.py first.")
        return
        
    print(f"Loading baseline DICOM series from {tcia_path}...")
    baseline_image = MedicalImage.from_dicom(tcia_path)
    
    # Create a synthetic mask for the baseline (same as run_benchmark.py)
    D, H, W = baseline_image.tensor.shape
    z, y, x = torch.meshgrid(
        torch.arange(D, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    
    cy, cx, cz = H // 2, W // 2, D // 2
    sz, sy, sx = baseline_image.spacing
    radius_mm = 15.0
    
    dist_sq = ((z - cz) * sz)**2 + ((y - cy) * sy)**2 + ((x - cx) * sx)**2
    mask_tensor = (dist_sq <= radius_mm**2).float()
    baseline_mask = Mask(tensor=mask_tensor, spacing=baseline_image.spacing)
    
    # 2. Generate Perturbed Data
    print("Applying syntactic affine transformation and noise (simulated rescan)...")
    perturbed_img_tensor, perturbed_mask_tensor = apply_perturbation(
        baseline_image.tensor, 
        baseline_mask.tensor,
        translation=(1.5, -1.0, 0.5), # Slight shift
        rotation_deg=2.0,             # Slight rotation
        noise_std=3.0                 # Noise
    )
    
    perturbed_image = MedicalImage(tensor=perturbed_img_tensor, spacing=baseline_image.spacing)
    perturbed_mask = Mask(tensor=perturbed_mask_tensor, spacing=baseline_image.spacing)
    
    # Ensure there is still an ROI
    if perturbed_mask.tensor.sum() == 0:
        print("Error: Perturbation shifted the mask entirely out of bounds.")
        return
        
    # Define features to test
    feature_classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    
    # 3. Fastrad Extraction
    print("Extracting features with fastrad...")
    fastrad_settings = FeatureSettings(feature_classes=feature_classes, bin_width=25.0, device="cpu")
    # Force single thread for predictable comparison if needed, though stability shouldn't care
    torch.set_num_threads(8)
    
    fastrad_extractor = FeatureExtractor(fastrad_settings)
    f_base = fastrad_extractor.extract(baseline_image, baseline_mask)
    f_pert = fastrad_extractor.extract(perturbed_image, perturbed_mask)
    
    # 4. PyRadiomics Extraction
    print("Extracting features with PyRadiomics...")
    pyrad_settings = {'binWidth': 25.0, 'interpolator': sitk.sitkNearestNeighbor, 'resampledPixelSpacing': None}
    pyrad_extractor = featureextractor.RadiomicsFeatureExtractor(**pyrad_settings)
    pyrad_extractor.disableAllFeatures()
    for fc in feature_classes:
        if fc == 'shape':
            pyrad_extractor.enableFeatureClassByName('shape')
            pyrad_extractor.enableFeatureClassByName('shape2D')
        else:
            pyrad_extractor.enableFeatureClassByName(fc)
            
    # Convert baseline to SITK
    sitk_img_base = sitk.GetImageFromArray(baseline_image.tensor.numpy())
    sitk_img_base.SetSpacing(list(reversed(baseline_image.spacing)))
    sitk_msk_base = sitk.GetImageFromArray(baseline_mask.tensor.numpy().astype('uint8'))
    sitk_msk_base.SetSpacing(list(reversed(baseline_image.spacing)))
    
    # Convert perturbed to SITK
    sitk_img_pert = sitk.GetImageFromArray(perturbed_image.tensor.numpy())
    sitk_img_pert.SetSpacing(list(reversed(baseline_image.spacing)))
    sitk_msk_pert = sitk.GetImageFromArray(perturbed_mask.tensor.numpy().astype('uint8'))
    sitk_msk_pert.SetSpacing(list(reversed(baseline_image.spacing)))

    p_base_raw = pyrad_extractor.execute(sitk_img_base, sitk_msk_base)
    p_pert_raw = pyrad_extractor.execute(sitk_img_pert, sitk_msk_pert)
    
    # Clean pyradiomics keys
    p_base = {}
    for k, v in p_base_raw.items():
        if not k.startswith("original_"): continue
        parts = k.split("_")
        fc = parts[1]
        fn = "_".join(parts[2:]).lower()
        if fc == "shape2D": fc = "shape2d"
        p_base[f"{fc}:{fn}"] = float(v)
        
    p_pert = {}
    for k, v in p_pert_raw.items():
        if not k.startswith("original_"): continue
        parts = k.split("_")
        fc = parts[1]
        fn = "_".join(parts[2:]).lower()
        if fc == "shape2D": fc = "shape2d"
        p_pert[f"{fc}:{fn}"] = float(v)
        
    # 5. Stability Comparison
    print("\n--- Stability Analysis Results ---")
    print(f"{'Feature':<45} | {'fastrad Δ %':<15} | {'PyRad Δ %':<15} | {'Parity (Abs Diff)':<15}")
    print("-" * 95)
    
    parity_failures = 0
    total_features = 0
    
    for key in f_base.keys():
        if key not in p_base:
            continue
            
        total_features += 1
            
        f_val_base = f_base[key]
        f_val_pert = f_pert[key]
        
        p_val_base = p_base[key]
        p_val_pert = p_pert[key]
        
        # Calculate percent delta relative to baseline
        # Avoid division by zero
        f_delta_pct = 0.0 if f_val_base == 0 else ((f_val_pert - f_val_base) / abs(f_val_base)) * 100.0
        p_delta_pct = 0.0 if p_val_base == 0 else ((p_val_pert - p_val_base) / abs(p_val_base)) * 100.0
        
        # We assert stability parity if the delta calculated by fastrad is the SAME delta calculated by PyRad
        # Tolerance allows for minor floating point accumulation differences in mathematically equivalent formulas
        abs_diff = abs(f_delta_pct - p_delta_pct)
        
        if abs_diff > 1.0: # If the stability shift differs by more than 1 percentage point
            parity_failures += 1
            print(f"{key:<45} | {f_delta_pct:>14.4f}% | {p_delta_pct:>14.4f}% | **FAIL ({abs_diff:.4f})**")
        else:
            # Only print a randomly sampled few to keep output clean, unless it fails
            if total_features % 10 == 0:
                 print(f"{key:<45} | {f_delta_pct:>14.4f}% | {p_delta_pct:>14.4f}% | {abs_diff:.4f}")
                 
    print("-" * 95)
    print(f"Total Features Compared: {total_features}")
    print(f"Parity Failures: {parity_failures}")
    print("")
    if parity_failures == 0:
        print("✅ Stability Parity Confirmed: fastrad reacts identically to PyRadiomics under synthetic physical deformation.")
    else:
        print("❌ Stability mismatch detected for some features.")

if __name__ == "__main__":
    run_stability_analysis()
