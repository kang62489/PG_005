"""Export each raw (un-merged) z-scored segment of 2025_06_11-0003 and 2025_12_15-0012
(GAUSS-normalized, the pipeline's default) to output/zscore_segments_{stem}/, so the
individual segments can be inspected directly.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from classes import AbfClip  # noqa: E402
from functions import zscore_img_segs  # noqa: E402

RECORDINGS = [
    ("2025_06_11-0003_BIEXP_GAUSS.tif", "2025_06_11_0004.abf"),
    ("2025_12_15-0012_BIEXP_GAUSS.tif", "2025_12_15_0008.abf"),
]


def main() -> None:
    for proc_tiff_name, abf_name in RECORDINGS:
        rec_stem = proc_tiff_name.replace("_BIEXP_GAUSS.tif", "")
        out_dir = PROJECT_ROOT / "output" / f"zscore_segments_{rec_stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        clip = AbfClip(
            proc_tiff_path=PROJECT_ROOT / "proc_tiffs" / proc_tiff_name,
            raw_abf_path=PROJECT_ROOT / "raw_abfs" / abf_name,
            results_dir=PROJECT_ROOT / "results",
            detrend_mode="BIEXP",
            normalization="GAUSS",
        )
        lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)

        for seg_idx, segment in enumerate(lst_zscore):
            tifffile.imwrite(out_dir / f"seg{seg_idx:02d}.tif", segment.astype(np.float32))

        print(f"Exported {len(lst_zscore)} segment(s) to {out_dir}")


if __name__ == "__main__":
    main()
