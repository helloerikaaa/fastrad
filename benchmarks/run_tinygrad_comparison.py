import os
os.environ["TINYGRAD_PROFILE"] = "1"

import time
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from fastrad.settings import FeatureSettings
from fastrad.features import firstorder, firstorder_tiny

# Dummy Volume (e.g. 64x64x64 scan)
print("Initializing dummy 64^3 3D volume...")
image_np = np.random.rand(64, 64, 64).astype(np.float32) * 100
mask_np = np.zeros((64, 64, 64), dtype=np.float32)
mask_np[16:48, 16:48, 16:48] = 1.0

img_t = torch.tensor(image_np)
mask_t = torch.tensor(mask_np)

settings = FeatureSettings(device="cpu", compile=False)
settings.spacing = (1.0, 1.0, 1.0)

print("\n--- PyTorch Eager (first pass) ---")
start = time.time()
pt_eager = firstorder.compute(img_t, mask_t, settings)
print(f"Time: {time.time() - start:.4f}s")
print(f"Energy: {pt_eager['firstorder:energy']}")

print("\n--- PyTorch Complied (first pass) ---")
settings.compile = True
start = time.time()
pt_compiled = firstorder.compute(img_t, mask_t, settings)
print(f"Time: {time.time() - start:.4f}s")

print("\n--- PyTorch Compiled (warm run) ---")
start = time.time()
pt_compiled = firstorder.compute(img_t, mask_t, settings)
print(f"Time: {time.time() - start:.4f}s")

print("\n--- Tinygrad @TinyJit (first pass compile) ---")
start = time.time()
tg_lazy = firstorder_tiny.compute(img_t, mask_t, settings)
print(f"Time: {time.time() - start:.4f}s")
print(f"Energy: {tg_lazy['firstorder:energy']}")

print("\n--- Tinygrad @TinyJit (warm run) ---")
start = time.time()
for _ in range(10):
    tg_lazy = firstorder_tiny.compute(img_t, mask_t, settings)
end = time.time()
print(f"Time over 10 iterations: {end - start:.4f}s. Avg: {(end - start)/10:.4f}s")

# Validation check
print("\n--- Correctness ---")
diff_mean = abs(pt_compiled["firstorder:mean"] - tg_lazy["firstorder:mean"])
diff_kurtosis = abs(pt_compiled["firstorder:kurtosis"] - tg_lazy["firstorder:kurtosis"])
print(f"Mean Error: {diff_mean:.6f}")
print(f"Kurtosis Error: {diff_kurtosis:.6f}")
