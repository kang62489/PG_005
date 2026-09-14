"""Export each raw (un-merged) z-scored segment of 2025_12_15-0012's ALS-corrected
tiff to output/als_segments_0012/, so the individual segments can be inspected
directly against the pipeline's merged *_MED.tif output.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classes import AbfClip  # noqa: E402
from functions import zscore_img_segs  # noqa: E402

PROC_TIFF_NAME = "2025_12_15-0012_BIEXP_ALS.tif"
ABF_NAME = "2025_12_15_0008.abf"

OUT_DIR = PROJECT_ROOT / "output" / "als_segments_0012"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    clip = AbfClip(
        proc_tiff_path=PROJECT_ROOT / "proc_tiffs" / PROC_TIFF_NAME,
        raw_abf_path=PROJECT_ROOT / "raw_abfs" / ABF_NAME,
        results_dir=PROJECT_ROOT / "results",
        detrend_mode="BIEXP",
        normalization="ALS",
    )
    lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)

    for seg_idx, segment in enumerate(lst_zscore):
        tifffile.imwrite(OUT_DIR / f"seg{seg_idx:02d}.tif", segment.astype(np.float32))

    print(f"Exported {len(lst_zscore)} segment(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
