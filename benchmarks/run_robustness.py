import torch
import SimpleITK as sitk
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor
import warnings
import numpy as np

def create_edge_case(t_shape, case_type):
    mask = torch.zeros(t_shape, dtype=torch.float32)
    img = torch.rand(t_shape, dtype=torch.float32) * 1000
    spacing = (1.0, 1.0, 1.0)
    
    if case_type == "empty":
        pass
    elif case_type == "single":
        mask[t_shape[0]//2, t_shape[1]//2, t_shape[2]//2] = 1.0
    elif case_type == "small":
        z, y, x = t_shape[0]//2, t_shape[1]//2, t_shape[2]//2
        mask[z, y, x:x+2] = 1.0
        mask[z, y+1, x:x+2] = 1.0
        # 4 voxels
    elif case_type == "non-isotropic":
        mask[t_shape[0]//2-1:t_shape[0]//2+2, t_shape[1]//2-1:t_shape[1]//2+2, t_shape[2]//2-1:t_shape[2]//2+2] = 1.0
        spacing = (5.0, 0.5, 0.5)
        
    return img, mask, spacing

def extract_safe(extractor_func, *args):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = extractor_func(*args)
            if len(w) > 0:
                warning_msgs = [str(warn.message) for warn in w]
                return f"Graceful Completion (Warnings: {len(warning_msgs)})"
            return "Graceful Completion"
    except Exception as e:
        err_type = type(e).__name__
        return f"{err_type}"

def run():
    print("Running Robustness Edge Cases Benchmark...")
    md = []
    md.append("## Section 6: Robustness\n")
    md.append("### 6.1 Edge Case Handling\n")
    md.append("| Edge Case | Expected Behaviour | fastrad Behaviour | PyRadiomics Behaviour |")
    md.append("|---|---|---|---|")
    
    cases = {
        "Empty Mask": ("empty", "ValueError / Exception"),
        "Single-voxel ROI": ("single", "Exception / Graceful"),
        "Very Small ROI (< 8 voxels)": ("small", "Exception / Graceful"),
        "Non-isotropic Spacing": ("non-isotropic", "UserWarning")
    }
    
    pyrad_ext = featureextractor.RadiomicsFeatureExtractor()
    pyrad_ext.settings['binWidth'] = 25.0
    fastrad_settings = FeatureSettings(feature_classes=['firstorder', 'glcm'], bin_width=25.0, device="cpu")
    
    for case_name, (case_id, expected) in cases.items():
        img_t, mask_t, spacing = create_edge_case((10, 10, 10), case_id)
        
        # Fastrad
        f_img = MedicalImage(img_t, spacing=spacing)
        f_mask = Mask(mask_t, spacing=spacing)
        f_ext = FeatureExtractor(fastrad_settings)
        f_res = extract_safe(f_ext.extract, f_img, f_mask)
        
        # PyRadiomics
        s_img = sitk.GetImageFromArray(img_t.numpy())
        s_img.SetSpacing((spacing[2], spacing[1], spacing[0]))
        s_mask = sitk.GetImageFromArray(mask_t.numpy().astype(np.uint8))
        s_mask.SetSpacing((spacing[2], spacing[1], spacing[0]))
        
        # Needs to catch pyradiomics specific logger errors which might just throw or just log
        p_res = extract_safe(pyrad_ext.execute, s_img, s_mask)
        
        md.append(f"| {case_name} | {expected} | {f_res} | {p_res} |")
        
    md.append("\n\n")
    return "\n".join(md)

if __name__ == "__main__":
    print(run())
