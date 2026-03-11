import time
import os
import torch
import radiomics
from radiomics import featureextractor
from fastrad import MedicalImage, Mask, FeatureExtractor, FeatureSettings

import SimpleITK as sitk

def run_bench():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "../tests/fixtures/image")
    mask_dir = os.path.join(base_dir, "../tests/fixtures/mask")

    # PyRadiomics
    t0 = time.time()
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('glszm')
    extractor.settings['binWidth'] = 25.0
    
    import sys
    sys.path.insert(0, os.path.join(base_dir, '../tests'))
    import make_fixtures
    sitk_image = sitk.GetImageFromArray(make_fixtures.img_vol)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask = sitk.GetImageFromArray(make_fixtures.mask_vol)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))
    
    pyrad_result = extractor.execute(sitk_image, sitk_mask)
    t1 = time.time()
    print(f"PyRadiomics CPU time: {t1-t0:.4f}s")

    # fastrad
    t0 = time.time()
    image = MedicalImage.from_dicom(img_dir)
    mask = Mask.from_dicom(mask_dir)
    settings = FeatureSettings(feature_classes=["gldm", "ngtdm", "glszm"], bin_width=25.0, device="cpu")
    fastrad_ext = FeatureExtractor(settings)
    fastrad_features = fastrad_ext.extract(image, mask)
    t1 = time.time()
    print(f"Fastrad CPU time: {t1-t0:.4f}s")

run_bench()
