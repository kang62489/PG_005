"""Throwaway: split zscore_img_segs time into pure TIFF read vs numpy math. Deleted after use."""
import time
from pathlib import Path

import numpy as np
import tifffile

TIFF_PATH = Path("proc_tiffs/2025_12_15-0013_BIEXP_GAUSS.tif")
# 20 segments x 41 frames, matching the real entry's set_interval_frames=20
ranges = [(200 + 40 * i, 240 + 40 * i) for i in range(20)]

t_read = 0.0
t_math = 0.0
with tifffile.TiffFile(TIFF_PATH) as tif:
    for left, right in ranges:
        t0 = time.perf_counter()
        segment = tif.asarray(key=slice(left, right + 1))
        t_read += time.perf_counter() - t0

        t0 = time.perf_counter()
        spike_idx = segment.shape[0] // 2
        baseline = segment[:spike_idx]
        mean = np.mean(baseline, axis=0)
        std = np.std(baseline, axis=0)
        std[std == 0] = 1
        _ = (segment - mean) / std
        t_math += time.perf_counter() - t0

print(f"read={t_read:.2f}s  math={t_math:.2f}s  total={t_read + t_math:.2f}s")

# Compare: upcast to float32 before doing mean/std/subtract
t_math32 = 0.0
with tifffile.TiffFile(TIFF_PATH) as tif:
    for left, right in ranges:
        segment = tif.asarray(key=slice(left, right + 1))
        t0 = time.perf_counter()
        segment32 = segment.astype(np.float32)
        spike_idx = segment32.shape[0] // 2
        baseline = segment32[:spike_idx]
        mean = np.mean(baseline, axis=0)
        std = np.std(baseline, axis=0)
        std[std == 0] = 1
        _ = (segment32 - mean) / std
        t_math32 += time.perf_counter() - t0

print(f"math (float32 upcast first) ={t_math32:.2f}s")
