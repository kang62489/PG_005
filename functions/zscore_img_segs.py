## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile


def zscore_img_segs(
    proc_tiff_path: Path,
    lst_img_frame_ranges: list[tuple[int, int]],
) -> list[np.ndarray]:
    lst_zscore: list[np.ndarray] = []

    with tifffile.TiffFile(proc_tiff_path) as tif:
        for left, right in lst_img_frame_ranges:
            segment = tif.asarray(key=slice(left, right + 1))

            spike_frame_idx = segment.shape[0] // 2
            baseline = segment[:spike_frame_idx]

            baseline_mean = np.mean(baseline, axis=0)
            baseline_std = np.std(baseline, axis=0)
            baseline_std[baseline_std == 0] = 1

            lst_zscore.append((segment - baseline_mean) / baseline_std)

    return lst_zscore
