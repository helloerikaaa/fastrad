import time
import torch
import gc
from fastrad import MedicalImage, Mask, FeatureSettings, FeatureExtractor

def generate_cube(size: int):
    """Generate a dummy image and mask of `size x size x size`."""
    image_tensor = torch.randint(0, 255, (size, size, size), dtype=torch.float32)
    # Full ROI
    mask_tensor = torch.ones((size, size, size), dtype=torch.float32)
    return image_tensor, mask_tensor

def profile_memory_usage():
    print("--- GPU Memory Profiling for Feature Classes ---")
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot profile GPU memory.")
        return

    # Warmup
    torch.ones(1).cuda()
    
    feature_classes = ['firstorder', 'shape', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    roi_sizes = [32, 64, 128, 192, 256]
    
    for cls in feature_classes:
        print(f"\nEvaluating Class: {cls}")
        for size in roi_sizes:
            try:
                # Force garbage collection and flush cache
                gc.collect()
                torch.cuda.empty_cache()
                
                # Create fake MedicalImage and Mask components manually using tensors to bypass DICOM loading
                img, msk = generate_cube(size)
                
                # Custom mock for fastrad MedicalImage and Mask
                class MockImage:
                    def __init__(self, t):
                        self.tensor = t
                        self.spacing = (1.0, 1.0, 1.0)
                        
                mock_img = MockImage(img)
                mock_msk = MockImage(msk)
                
                settings = FeatureSettings(feature_classes=[cls], bin_width=25.0, device="cuda")
                extractor = FeatureExtractor(settings)
                
                # Reset max memory stats before run
                torch.cuda.reset_peak_memory_stats()
                
                t0 = time.time()
                _ = extractor.extract(mock_img, mock_msk)
                torch.cuda.synchronize()
                t1 = time.time()
                
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_mb = peak_bytes / (1024 * 1024)
                
                print(f"  ROI {size:3d}^3 -> Peak VRAM: {peak_mb:8.2f} MB  (Time: {t1-t0:6.3f}s)")
                
                # Cleanup
                del mock_img, mock_msk, img, msk, extractor, settings
                gc.collect()
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"  ROI {size:3d}^3 -> OUT OF MEMORY ERROR")
                # Immediately clear cache to survive loop
                torch.cuda.empty_cache()
                break  # Don't try larger sizes for this class
            except Exception as e:
                print(f"  ROI {size:3d}^3 -> ERROR: {e}")
                break

if __name__ == "__main__":
    profile_memory_usage()
