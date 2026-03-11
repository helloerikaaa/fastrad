import os
import sys
import tracemalloc
import time
import torch
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureExtractor, FeatureSettings

import sys
import os
import numpy as np
import torch
import SimpleITK as sitk
import pydicom

def generate_synthetic_tumor_mask(dicom_dir, mask_path, radius_mm=15.0):
    import SimpleITK as sitk
    import pydicom
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Just make a basic binary mask block in the center
    mask_np = np.zeros(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)
    z, y, x = mask_np.shape
    mask_np[z//2-10:z//2+10, y//2-30:y//2+30, x//2-30:x//2+30] = 1
    
    mask_sitk = sitk.GetImageFromArray(mask_np)
    mask_sitk.CopyInformation(image)
    sitk.WriteImage(mask_sitk, mask_path)

def profile_pyradiomics(image_path, mask_path):
    print("Profiling PyRadiomics Memory...")
    tracemalloc.start()
    
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")
    
    start_time = time.time()
    res = extractor.execute(image_path, mask_path)
    end_time = time.time()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"PyRadiomics Peak Memory: {peak / 10**6:.2f} MB")
    print(f"PyRadiomics Time: {end_time - start_time:.2f} s")
    return peak / 10**6

def profile_fastrad(image_path, mask_path):
    print("Profiling fastrad Memory (CPU 1 Thread)...")
    torch.set_num_threads(1)
    
    tracemalloc.start()
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_path)
    reader.SetFileNames(dicom_names)
    sitk_image = reader.Execute()
    image_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_image)).float()
    
    sitk_mask = sitk.ReadImage(mask_path)
    mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(sitk_mask)).float()
    
    image = MedicalImage(image_tensor)
    mask = Mask(mask_tensor)
    
    settings = FeatureSettings(
        feature_classes=["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
        bin_width=25.0,
        device="cpu"
    )
    
    extractor = FeatureExtractor(settings)
    
    start_time = time.time()
    res = extractor.extract(image, mask)
    end_time = time.time()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"fastrad Peak Memory: {peak / 10**6:.2f} MB")
    print(f"fastrad Time: {end_time - start_time:.2f} s")
    return peak / 10**6

if __name__ == "__main__":
    dicom_dir = os.path.join("tests", "fixtures", "tcia", "images")
    mask_path = os.path.join("tests", "fixtures", "tcia", "mask.nrrd")
    
    if not os.path.exists(mask_path):
        import pydicom
        generate_synthetic_tumor_mask(dicom_dir, mask_path, radius_mm=15.0)
    
    peak_fastrad = profile_fastrad(dicom_dir, mask_path)
    peak_pyrad = profile_pyradiomics(dicom_dir, mask_path)
    
    print(f"\nMemory Efficiency: fastrad uses {peak_pyrad / peak_fastrad:.2f}x less RAM than PyRadiomics.")
