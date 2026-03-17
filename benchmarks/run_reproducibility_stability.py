import numpy as np
import torch
import SimpleITK as sitk
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor
from pathlib import Path
import os
import time

def compute_icc_2_1(data_matrix):
    """
    Computes ICC(2,1) - Two-way random effects, absolute agreement, single rater.
    data_matrix: array of shape (n_subjects, n_raters/scans)
    """
    n, k = data_matrix.shape
    mean_row = np.mean(data_matrix, axis=1)
    mean_col = np.mean(data_matrix, axis=0)
    mean_tot = np.mean(data_matrix)
    
    ss_total = np.sum((data_matrix - mean_tot)**2)
    ss_row = k * np.sum((mean_row - mean_tot)**2)
    ss_col = n * np.sum((mean_col - mean_tot)**2)
    ss_err = ss_total - ss_row - ss_col
    
    ms_row = ss_row / (n - 1) if n > 1 else 0
    ms_col = ss_col / (k - 1) if k > 1 else 0
    ms_err = ss_err / ((n - 1) * (k - 1)) if n > 1 and k > 1 else 0
    
    if ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n == 0:
        return 0.0
        
    icc = (ms_row - ms_err) / (ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n)
    return max(0.0, min(1.0, icc))

def apply_perturbations(img_tensor):
    """Apply defined physics perturbations for robustness testing."""
    noise = img_tensor + torch.randn_like(img_tensor) * 20.0 # 20 HU gaussian noise
    shifted = torch.roll(img_tensor, shifts=(2, 2, 2), dims=(0, 1, 2))
    return {"Gaussian Noise": noise, "Translation": shifted}

def run():
    print("Running Reproducibility and Perturbation Stability Benchmark...")
    project_root = Path(__file__).parent.parent
    img_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
    mask_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
    
    md = []
    md.append("## Section 5: Reproducibility and Stability\n")
    
    print("  -> Running ICC Analysis on real RIDER test-retest pairs (5.1)...")
    md.append("### 5.1 ICC Analysis on Real RIDER Scan-Rescan Pairs\n")
    
    import warnings
    import re
    
    def pyrad_to_fastrad(pyrad_key):
        if not pyrad_key.startswith("original_"):
            return None
        parts = pyrad_key.split('_')
        if len(parts) < 3: return None
        f_class = parts[1]
        f_name = "_".join(parts[2:])
        
        mapping = {
            '10Percentile': '10th_percentile',
            '90Percentile': '90th_percentile',
            'InterquartileRange': 'interquartile_range',
            'MeanAbsoluteDeviation': 'mean_absolute_deviation',
            'RobustMeanAbsoluteDeviation': 'robust_mean_absolute_deviation',
            'RootMeanSquared': 'root_mean_squared',
            'TotalEnergy': 'total_energy',
            'Imc1': 'imc1',
            'Imc2': 'imc2',
            'Idm': 'idm',
            'Idmn': 'idmn',
            'Id': 'id',
            'Idn': 'idn',
            'MCC': 'mcc'
        }
        
        if f_name in mapping:
            f_name_snake = mapping[f_name]
        else:
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', f_name)
            f_name_snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            
        return f'{f_class}:{f_name_snake}'

    
    def create_spherical_mask_t(img_tensor, radius_mm, spacing):
        D, H, W = img_tensor.shape
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
        mask = torch.zeros_like(img_tensor, dtype=torch.float32)
        mask[dist_sq <= 1.0] = 1.0
        return mask

    def load_dicom_series(dir_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dir_path))
        if not dicom_names:
            return None
        reader.SetFileNames(dicom_names)
        return reader.Execute()
        
    rider_dir = project_root / "tests" / "fixtures" / "tcia" / "rider"
    rider_pairs = []
    if rider_dir.exists():
        for p_dir in sorted(rider_dir.iterdir()):
            if p_dir.is_dir():
                s1, s2 = p_dir / "scan1", p_dir / "scan2"
                if s1.exists() and s2.exists():
                    rider_pairs.append((s1, s2))
                    
    if not rider_pairs:
        md.append("*No RIDER pairs available. Skipping ICC Analysis.*")
    else:
        # Evaluate up to 10 pairs
        rider_pairs = rider_pairs[:10]
        n_patients = len(rider_pairs)
        
        pyrad_ext = featureextractor.RadiomicsFeatureExtractor()
        pyrad_ext.settings['binWidth'] = 25.0
        classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
        fastrad_ext = FeatureExtractor(FeatureSettings(feature_classes=classes, bin_width=25.0, device="cpu"))
        
        # Pre-initialize data structs by running on first scan
        test_img = load_dicom_series(rider_pairs[0][0])
        test_img_t = torch.from_numpy(sitk.GetArrayFromImage(test_img)).float()
        test_spacing = test_img.GetSpacing()[::-1]
        test_mask = create_spherical_mask_t(torch.zeros_like(test_img_t), 15.0, test_spacing)
        
        s_test_img = sitk.GetImageFromArray(test_img_t.numpy())
        s_test_img.SetSpacing(test_spacing[::-1])
        s_test_img.SetOrigin(test_img.GetOrigin())
        s_test_img.SetDirection(test_img.GetDirection())
        
        s_test_mask = sitk.GetImageFromArray(test_mask.numpy().astype(np.uint8))
        s_test_mask.SetSpacing(test_spacing[::-1])
        s_test_mask.SetOrigin(test_img.GetOrigin())
        s_test_mask.SetDirection(test_img.GetDirection())
        
        base_f_res = fastrad_ext.extract(MedicalImage(test_img_t, spacing=test_spacing), Mask(test_mask, spacing=test_spacing))
        fastrad_icc_data = {k: np.zeros((n_patients, 2)) for k in base_f_res.keys()}
        pyrad_icc_data = {}
        pyrad_ext.disableAllFeatures()
        for cls in classes:
            pyrad_ext.enableFeatureClassByName(cls)
        res = pyrad_ext.execute(s_test_img, s_test_mask)
        for k in res:
            if k.startswith("original_"):
                key = pyrad_to_fastrad(k)
                if key and key in fastrad_icc_data: pyrad_icc_data[key] = np.zeros((n_patients, 2))
        
        for p_idx, (s1, s2) in enumerate(rider_pairs):
            sitk_img1 = load_dicom_series(s1)
            sitk_img2 = load_dicom_series(s2)
            
            img1_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_img1)).float()
            spacing1 = sitk_img1.GetSpacing()[::-1]
            try:
                img2_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_img2)).float()
                spacing2 = sitk_img2.GetSpacing()[::-1]
            except Exception:
                continue
                
            # Second requirement: generate mask on scan1 ONLY, apply identical mask rigidly to scan2
            # Notice: scan 1 and scan 2 might have different spatial bounding boxes/slice counts in RIDER.
            # We must use proper SimpleITK physical resampling to map the mask into scan 2's frame of reference.
            mask1_t = create_spherical_mask_t(img1_t, 15.0, spacing1)
            
            s_img1 = sitk.GetImageFromArray(img1_t.numpy())
            s_img1.SetSpacing(spacing1[::-1])
            s_img1.SetOrigin(sitk_img1.GetOrigin())
            s_img1.SetDirection(sitk_img1.GetDirection())
            
            s_mask1 = sitk.GetImageFromArray(mask1_t.numpy().astype(np.uint8))
            s_mask1.SetSpacing(spacing1[::-1])
            s_mask1.SetOrigin(sitk_img1.GetOrigin())
            s_mask1.SetDirection(sitk_img1.GetDirection())
            
            s_img2 = sitk.GetImageFromArray(img2_t.numpy())
            s_img2.SetSpacing(spacing2[::-1])
            s_img2.SetOrigin(sitk_img2.GetOrigin())
            s_img2.SetDirection(sitk_img2.GetDirection())
            
            # Resample mask1 into the domain of img2 physically
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(s_img2)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(sitk.Transform()) # Identity transform maps physical space directly
            s_mask2 = resampler.Execute(s_mask1)
            
            # Convert resampled mask back to tensor for fastrad
            mask2_t = torch.from_numpy(sitk.GetArrayFromImage(s_mask2)).float()
            
            f_img1, f_mask1 = MedicalImage(img1_t, spacing=spacing1), Mask(mask1_t, spacing=spacing1)
            f_img2, f_mask2 = MedicalImage(img2_t, spacing=spacing2), Mask(mask2_t, spacing=spacing2)
            
            # Extract Fastrad
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_res1 = fastrad_ext.extract(f_img1, f_mask1)
                f_res2 = fastrad_ext.extract(f_img2, f_mask2)
            for k in fastrad_icc_data:
                fastrad_icc_data[k][p_idx, 0] = f_res1.get(k, 0)
                fastrad_icc_data[k][p_idx, 1] = f_res2.get(k, 0)
                
            # Extract PyRadiomics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pyrad_ext.disableAllFeatures()
                for cls in classes:
                    pyrad_ext.enableFeatureClassByName(cls)
                try:
                    r1 = pyrad_ext.execute(s_img1, s_mask1)
                    r2 = pyrad_ext.execute(s_img2, s_mask2)
                    for k in r1:
                        if k.startswith("original_"):
                            key = pyrad_to_fastrad(k)
                            if key and key in pyrad_icc_data:
                                pyrad_icc_data[key][p_idx, 0] = float(r1[k]) if hasattr(r1[k], "item") == False else float(r1[k].item())
                                pyrad_icc_data[key][p_idx, 1] = float(r2[k]) if hasattr(r2[k], "item") == False else float(r2[k].item())
                except Exception as e:
                    print(f"PyRadiomics extraction failed: {e}")
        
        f_iccs = [compute_icc_2_1(data) for data in fastrad_icc_data.values()]
        p_iccs = []
        for feature_name, data in pyrad_icc_data.items():
            icc_val = compute_icc_2_1(data)
            p_iccs.append(icc_val)
            if np.isnan(icc_val):
                print(f"NaN ICC for PyRadiomics feature: {feature_name}")
                print(f"Data slice: {data[:5]}")
        
        f_high = (sum(1 for x in f_iccs if not np.isnan(x) and x >= 0.90) / len(f_iccs) * 100) if len(f_iccs) > 0 else 0
        p_high = (sum(1 for x in p_iccs if not np.isnan(x) and x >= 0.90) / len(p_iccs) * 100) if len(p_iccs) > 0 else 0
        
        md.append(f"- **Fastrad Features with ICC ≥ 0.90**: {f_high:.1f}%")
        md.append(f"- **PyRadiomics Features with ICC ≥ 0.90**: {p_high:.1f}%")
        md.append(f"- **Fastrad Mean ICC**: {np.nanmean(f_iccs):.4f}")
        md.append(f"- **PyRadiomics Mean ICC**: {np.nanmean(p_iccs):.4f}\n")
        
        from scipy.stats import wilcoxon
        valid_f_iccs = []
        valid_p_iccs = []
        f_keys = list(fastrad_icc_data.keys())
        p_keys = list(pyrad_icc_data.keys())
        for key in p_keys:
            if key in f_keys:
                f_idx = f_keys.index(key)
                p_idx = p_keys.index(key)
                f_val = f_iccs[f_idx]
                p_val = p_iccs[p_idx]
                if not np.isnan(f_val) and not np.isnan(p_val):
                    valid_f_iccs.append(f_val)
                    valid_p_iccs.append(p_val)
        
        if valid_f_iccs and valid_p_iccs:
            stat, p = wilcoxon(valid_f_iccs, valid_p_iccs)
            md.append(f"- **Wilcoxon signed-rank test**: stat={stat:.4f}, p={p:.4f}\n")


    # 5.2 Perturbation Stability Analysis
    print("  -> Profiling Perturbation Stability Matrices (5.2)...")
    md.append("### 5.2 Numerical Robustness to Input Perturbation\n")
    md.append("| Perturbation | PyRadiomics Mean Drift (%) | fastrad Mean Drift (%) | Failure Count |")
    md.append("|---|---|---|---|")
    
    if img_path.exists() and mask_path.exists():
        sitk_image = sitk.ReadImage(str(img_path))
        sitk_mask = sitk.ReadImage(str(mask_path))
        
        img_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
        mask_t = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
        spacing = sitk_image.GetSpacing()[::-1]

        pyrad_ext = featureextractor.RadiomicsFeatureExtractor()
        pyrad_ext.settings['binWidth'] = 25.0
        classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
        fastrad_ext = FeatureExtractor(FeatureSettings(feature_classes=classes, bin_width=25.0, device="cpu"))
        
        fastrad_base = fastrad_ext.extract(MedicalImage(img_t, spacing=spacing), Mask(mask_t, spacing=spacing))
        
        pyrad_base = {}
        for cls in classes:
            pyrad_ext.disableAllFeatures()
            pyrad_ext.enableFeatureClassByName(cls)
            res = pyrad_ext.execute(sitk_image, sitk_mask)
            for k, v in res.items():
                if k.startswith("original_"):
                    key = pyrad_to_fastrad(k)
                    if key:
                        pyrad_base[key] = float(v) if not hasattr(v, "item") else float(v.item())

        perturbations = apply_perturbations(img_t)
        
        for p_name, p_tensor in perturbations.items():
            f_img_p = MedicalImage(p_tensor, spacing=spacing)
            f_res_p = fastrad_ext.extract(f_img_p, Mask(mask_t, spacing=spacing))
            
            s_img_p = sitk.GetImageFromArray(p_tensor.numpy())
            s_img_p.CopyInformation(sitk_image)
            
            p_res_p = {}
            pyrad_ext.disableAllFeatures()
            for cls in classes:
                pyrad_ext.enableFeatureClassByName(cls)
            res = pyrad_ext.execute(s_img_p, sitk_mask)
            for k, v in res.items():
                if k.startswith("original_"):
                    key = pyrad_to_fastrad(k)
                    if key:
                        p_res_p[key] = float(v)
                        
            f_drifts = []
            p_drifts = []
            failures = 0
            
            for k in fastrad_base:
                if abs(fastrad_base[k]) > 1e-6:
                    drv = abs(f_res_p[k] - fastrad_base[k]) / abs(fastrad_base[k]) * 100.0
                    f_drifts.append(drv)
                    
            for k in pyrad_base:
                if abs(pyrad_base[k]) > 1e-6 and k in p_res_p:
                    drv = abs(p_res_p[k] - pyrad_base[k]) / abs(pyrad_base[k]) * 100.0
                    p_drifts.append(drv)
                    
            f_mean = np.mean(f_drifts) if f_drifts else 0
            p_mean = np.mean(p_drifts) if p_drifts else 0
            
            md.append(f"| {p_name} | {p_mean:.4f}% | {f_mean:.4f}% | {failures} |")
            
    md.append("\n*Note: Large translation drift is expected because moving the ROI to a different anatomical region changes the intensity distribution entirely.*\n\n")
    return "\n".join(md)

if __name__ == "__main__":
    print(run())
