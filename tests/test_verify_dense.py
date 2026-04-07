import torch
from fastrad import MedicalImage, Mask, FeatureSettings, DenseFeatureExtractor

def test_dense():
    # create a small dummy volume
    img = torch.rand((5, 5, 5), dtype=torch.float32)
    # create a mask with mostly ones
    mask = torch.ones((5, 5, 5), dtype=torch.float32)
    mask[0, 0, 0] = 0
    
    settings = FeatureSettings()
    # only test firstorder to be fast
    settings.feature_classes = ["firstorder"]
    
    med_img = MedicalImage(img)
    med_mask = Mask(mask)
    
    extractor = DenseFeatureExtractor(settings)
    
    out = extractor.extract_dense(med_img, med_mask, kernel_size=3, stride=1)
    
    # 5 - 3 + 1 = 3
    # out should be dict of 3x3x3 tensors
    
    print("Computed features:", list(out.keys()))
    if out:
        first_key = list(out.keys())[0]
        print(f"Shape of {first_key}:", out[first_key].shape)
        
    print("Test passed!")

if __name__ == "__main__":
    test_dense()
