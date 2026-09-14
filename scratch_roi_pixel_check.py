"""One-off check: pull an 8x8 ROI near frame-center at the spike frame across all
20 individual (un-merged) segments of 2025_12_15-0012, and print sorted per-pixel
values + median for a few pixels in that ROI -- to see whether the median pipeline's
modest peak z-score reflects real per-segment variability/jitter, not a bug.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classes import AbfClip  # noqa: E402
from functions import zscore_img_segs  # noqa: E402

PROC_TIFF_NAME = "2025_12_15-0012_BIEXP_GAUSS.tif"
ABF_NAME = "2025_12_15_0008.abf"
ROI_HALF = 4  # 8x8 ROI


def main() -> None:
    clip = AbfClip(
        proc_tiff_path=PROJECT_ROOT / "proc_tiffs" / PROC_TIFF_NAME,
        raw_abf_path=PROJECT_ROOT / "raw_abfs" / ABF_NAME,
        results_dir=PROJECT_ROOT / "results",
        detrend_mode="BIEXP",
        normalization="GAUSS",
    )
    lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)

    n_frames = lst_zscore[0].shape[0]
    spike_frame_idx = n_frames // 2
    row_c, col_c = 533, 466
    r0, r1 = row_c - ROI_HALF, row_c + ROI_HALF
    c0, c1 = col_c - ROI_HALF, col_c + ROI_HALF

    roi_per_segment = np.stack([seg[spike_frame_idx, r0:r1, c0:c1] for seg in lst_zscore])  # (20, 8, 8)
    print(f"ROI: rows {r0}:{r1}, cols {c0}:{c1}  (frame center = {row_c},{col_c})")
    print(f"roi_per_segment shape: {roi_per_segment.shape}  (n_segments, roi_h, roi_w)\n")

    for pr in range(roi_per_segment.shape[1]):
        for pc in range(roi_per_segment.shape[2]):
            values = roi_per_segment[:, pr, pc]
            sorted_values = np.round(np.sort(values), 2).tolist()
            median_value = float(np.median(values))
            print(f"({r0 + pr},{c0 + pc}) median={median_value:6.2f}  sorted={sorted_values}")


if __name__ == "__main__":
    main()
