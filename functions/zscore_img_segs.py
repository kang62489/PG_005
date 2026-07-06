## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile
from numba import njit, prange


@njit(parallel=True, cache=True)
def _cpu_zscore_segment(segment: np.ndarray) -> np.ndarray:
    n_frames, height, width = segment.shape
    spike_frame_idx = n_frames // 2
    out = np.empty((n_frames, height, width), dtype=np.float32)

    for row in prange(height):
        for col in range(width):
            baseline_sum = 0.0
            baseline_sum_sq = 0.0
            for frame_idx in range(spike_frame_idx):
                value = float(segment[frame_idx, row, col])
                baseline_sum += value
                baseline_sum_sq += value * value

            baseline_mean = baseline_sum / spike_frame_idx
            variance = baseline_sum_sq / spike_frame_idx - baseline_mean * baseline_mean
            baseline_std = np.sqrt(variance) if variance > 0.0 else 1.0

            for frame_idx in range(n_frames):
                out[frame_idx, row, col] = (segment[frame_idx, row, col] - baseline_mean) / baseline_std

    return out


def zscore_img_segs(
    proc_tiff_path: Path,
    lst_img_frame_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    lst_zscore: list[np.ndarray] = []

    with tifffile.TiffFile(proc_tiff_path) as tif:
        for left, right in lst_img_frame_ranges:
            # Source TIFFs are float16 — numpy has no native float16 arithmetic and
            # emulates it in software, making mean/std/normalize ~6x slower than float32.
            segment = tif.asarray(key=slice(left, right + 1)).astype(np.float32)
            lst_zscore.append(_cpu_zscore_segment(segment))

    return lst_zscore
