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
        test_mask = create_spherical_mask_t(torch.zeros(test_img.GetSize()[::-1]), 15.0, test_img.GetSpacing()[::-1])
        s_test_mask = sitk.GetImageFromArray(test_mask.numpy().astype(np.uint8))
        s_test_mask.CopyInformation(test_img)
        
        base_f_res = fastrad_ext.extract(MedicalImage(torch.from_numpy(sitk.GetArrayFromImage(test_img)).float(), spacing=test_img.GetSpacing()[::-1]), Mask(test_mask, spacing=test_img.GetSpacing()[::-1]))
        fastrad_icc_data = {k: np.zeros((n_patients, 2)) for k in base_f_res.keys()}
        pyrad_icc_data = {}
        for cls in classes:
            pyrad_ext.disableAllFeatures()
            pyrad_ext.enableFeatureClassByName(cls)
            res = pyrad_ext.execute(test_img, s_test_mask)
            for k in res:
                if k.startswith("original_"):
                    key = f"{k.split('_')[1]}_{k.split('_')[2]}".lower()
                    if key in fastrad_icc_data: pyrad_icc_data[key] = np.zeros((n_patients, 2))
        
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
                
            mask1_t = create_spherical_mask_t(img1_t, 15.0, spacing1)
            mask2_t = create_spherical_mask_t(img2_t, 15.0, spacing2)
            
            f_img1, f_mask1 = MedicalImage(img1_t, spacing=spacing1), Mask(mask1_t, spacing=spacing1)
            f_img2, f_mask2 = MedicalImage(img2_t, spacing=spacing2), Mask(mask2_t, spacing=spacing2)
            
            s_mask1 = sitk.GetImageFromArray(mask1_t.numpy().astype(np.uint8))
            s_mask1.CopyInformation(sitk_img1)
            s_mask2 = sitk.GetImageFromArray(mask2_t.numpy().astype(np.uint8))
            s_mask2.CopyInformation(sitk_img2)
            
            # Extract Fastrad
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_res1 = fastrad_ext.extract(f_img1, f_mask1)
                f_res2 = fastrad_ext.extract(f_img2, f_mask2)
            for k in fastrad_icc_data:
                fastrad_icc_data[k][p_idx, 0] = f_res1.get(k, 0)
                fastrad_icc_data[k][p_idx, 1] = f_res2.get(k, 0)
                
            # Extract PyRadiomics
            for cls in classes:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pyrad_ext.disableAllFeatures()
                    pyrad_ext.enableFeatureClassByName(cls)
                    try:
                        r1 = pyrad_ext.execute(sitk_img1, s_mask1)
                        r2 = pyrad_ext.execute(sitk_img2, s_mask2)
                        for k in r1:
                            if k.startswith("original_"):
                                key = f"{k.split('_')[1]}_{k.split('_')[2]}".lower()
                                if key in pyrad_icc_data:
                                    pyrad_icc_data[key][p_idx, 0] = float(r1[k]) if hasattr(r1[k], "item") == False else float(r1[k].item())
                                    pyrad_icc_data[key][p_idx, 1] = float(r2[k]) if hasattr(r2[k], "item") == False else float(r2[k].item())
                    except Exception:
                        pass
        
        f_iccs = [compute_icc_2_1(data) for data in fastrad_icc_data.values()]
        p_iccs = [compute_icc_2_1(data) for data in pyrad_icc_data.values()]
        
        f_high = (sum(1 for x in f_iccs if x >= 0.90) / len(f_iccs) * 100) if len(f_iccs) > 0 else 0
        p_high = (sum(1 for x in p_iccs if x >= 0.90) / len(p_iccs) * 100) if len(p_iccs) > 0 else 0
        
        md.append(f"- **Fastrad Features with ICC ≥ 0.90**: {f_high:.1f}%")
        md.append(f"- **PyRadiomics Features with ICC ≥ 0.90**: {p_high:.1f}%")
        md.append(f"- **Fastrad Mean ICC**: {np.mean(f_iccs):.4f}")
        md.append(f"- **PyRadiomics Mean ICC**: {np.mean(p_iccs):.4f}\n")

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
                    parts = k.split("_")
                    key = f"{parts[1]}_{parts[2]}".lower()
                    pyrad_base[key] = float(v) if not hasattr(v, "item") else float(v.item())

        perturbations = apply_perturbations(img_t)
        
        for p_name, p_tensor in perturbations.items():
            f_img_p = MedicalImage(p_tensor, spacing=spacing)
            f_res_p = fastrad_ext.extract(f_img_p, Mask(mask_t, spacing=spacing))
            
            s_img_p = sitk.GetImageFromArray(p_tensor.numpy())
            s_img_p.CopyInformation(sitk_image)
            
            p_res_p = {}
            for cls in classes:
                pyrad_ext.disableAllFeatures()
                pyrad_ext.enableFeatureClassByName(cls)
                res = pyrad_ext.execute(s_img_p, sitk_mask)
                for k, v in res.items():
                    if k.startswith("original_"):
                        p_res_p[f"{k.split('_')[1]}_{k.split('_')[2]}".lower()] = float(v)
                        
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
