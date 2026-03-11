import time
import torch
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor
from pathlib import Path
import os
import sys

def run():
    print("Running Numerical Parity Benchmark with PyRadiomics on Clinical Data...")
    project_root = Path(__file__).parent.parent
    img_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_image.nrrd"
    mask_path = project_root / "tests" / "fixtures" / "tcia" / "lung1_label.nrrd"
    
    if not (img_path.exists() and mask_path.exists()):
        return "Error: TCIA clinical nrrd files not found."

    # PyRadiomics
    sitk_image = sitk.ReadImage(str(img_path))
    sitk_mask = sitk.ReadImage(str(mask_path))
    
    # fastrad
    image_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
    mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
    spacing = sitk_image.GetSpacing()[::-1] # (Z, Y, X)
    
    fastrad_image = MedicalImage(image_tensor, spacing=spacing)
    fastrad_mask = Mask(mask_tensor, spacing=spacing)
    
    feature_classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    
    pyrad_extractor = featureextractor.RadiomicsFeatureExtractor()
    pyrad_extractor.settings['binWidth'] = 25.0
    
    fastrad_settings = FeatureSettings(feature_classes=feature_classes, bin_width=25.0, device="cpu")
    fastrad_extractor = FeatureExtractor(fastrad_settings)
    
    fastrad_features = fastrad_extractor.extract(fastrad_image, fastrad_mask)
    
    pyrad_features = {}
    for cls in feature_classes:
        pyrad_extractor.disableAllFeatures()
        pyrad_extractor.enableFeatureClassByName(cls)
        res = pyrad_extractor.execute(sitk_image, sitk_mask)
        # Process PyRadiomics dict into identical keys
        for k, v in res.items():
            if not k.startswith("original_"): continue
            parts = k.split("_")
            c_name = parts[1]
            f_name = parts[2]
            key = f"{c_name}_{f_name}".lower()
            if hasattr(v, "item"):
                v = v.item()
            pyrad_features[key] = float(v)
            
    # Compute differences
    md = []
    md.append("## Section 2: Numerical Parity with PyRadiomics (TCIA Clinical Image)\n")
    md.append("### 2.1 Feature-level Agreement Table\n")
    md.append("| Feature Class | Mean Abs Diff | Max Abs Diff | Features Within 1e-4 | Features Outside 1e-4 |")
    md.append("|---|---|---|---|---|")
    
    diff_stats = {}
    
    for k, fastrad_val in fastrad_features.items():
        if k in pyrad_features:
            pyrad_val = pyrad_features[k]
            abs_diff = abs(fastrad_val - pyrad_val)
            
            c_name = k.split("_")[0] if "_" in k else k.split(":")[0]
            
            if c_name not in diff_stats:
                diff_stats[c_name] = {"diffs": [], "within": 0, "outside": 0, "outside_list": []}
                
            diff_stats[c_name]["diffs"].append(abs_diff)
            if abs_diff <= 1e-4:
                diff_stats[c_name]["within"] += 1
            else:
                diff_stats[c_name]["outside"] += 1
                diff_stats[c_name]["outside_list"].append((k, abs_diff, fastrad_val, pyrad_val))
                
    for cls in feature_classes:
        if cls in diff_stats:
            stats = diff_stats[cls]
            mean_diff = sum(stats["diffs"]) / len(stats["diffs"])
            max_diff = max(stats["diffs"])
            within = stats["within"]
            outside = stats["outside"]
            
            md.append(f"| {cls} | {mean_diff:.2e} | {max_diff:.2e} | {within} | {outside} |")
            
    md.append("\n**Outlier Analysis:**\n")
    has_outliers = False
    for cls in feature_classes:
        if cls in diff_stats and diff_stats[cls]["outside"] > 0:
            has_outliers = True
            for out_k, out_diff, f_val, p_val in diff_stats[cls]["outside_list"]:
                md.append(f"- `{out_k}`: diff={out_diff:.2e} (fastrad={f_val:.4e}, PyRadiomics={p_val:.4e})")
                
    if not has_outliers:
        md.append("All features across all classes are strictly within the designated `1e-4` parity tolerance.\n")

    return "\n".join(md) + "\n\n"

if __name__ == "__main__":
    print(run())
