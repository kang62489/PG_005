"""
Prototype: export a few segments as TIFF before and after z-score normalization,
for visual/numeric comparison. Uses the same 2 test recordings as other prototypes.

"Before" = raw segment pixels straight from the proc TIFF (float16 -> float32 cast only).
"After"  = per-pixel, per-segment baseline z-score via functions.zscore_img_segs.

Scratch/one-off script -- not part of the reviewed codebase.
"""

from pathlib import Path

import numpy as np
import tifffile

from classes import AbfClip
from functions import zscore_img_segs

PROC_TIFFS_DIR = Path("D:/Programs/PG_005/proc_tiffs")
RAW_ABFS_DIR = Path("D:/Programs/PG_005/raw_abfs")
OUT_ROOT = Path("D:/Programs/PG_005/output/zscore_before_after")

N_SEGMENTS_TO_EXPORT = 3

RECORDINGS = [
    {"tiff_stem": "2025_06_11-0003", "abf_name": "2025_06_11_0004.abf"},
    {"tiff_stem": "2025_12_15-0012", "abf_name": "2025_12_15_0008.abf"},
]


def run_recording(tiff_stem: str, abf_name: str, normalization: str) -> None:
    print(f"\n=== {tiff_stem} ({normalization}) ===")
    proc_tiff_path = PROC_TIFFS_DIR / f"{tiff_stem}_BIEXP_{normalization}.tif"
    raw_abf_path = RAW_ABFS_DIR / abf_name
    out_dir = OUT_ROOT / normalization / tiff_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    clip = AbfClip(
        proc_tiff_path=proc_tiff_path,
        raw_abf_path=raw_abf_path,
        results_dir=out_dir,
        detrend_mode="BIEXP",
        normalization=normalization,
    )
    if not clip.lst_img_frame_ranges:
        print("No valid segments -- skipping.")
        return

    n_export = min(N_SEGMENTS_TO_EXPORT, len(clip.lst_img_frame_ranges))
    ranges_to_export = clip.lst_img_frame_ranges[:n_export]

    print(f"Reading {n_export} raw (pre-z-score) segments...")
    with tifffile.TiffFile(clip.proc_tiff_path) as tif:
        for i, (left, right) in enumerate(ranges_to_export):
            raw_segment = tif.asarray(key=slice(left, right + 1)).astype(np.float32)
            out_path = out_dir / f"segment_{i:02d}_before.tif"
            tifffile.imwrite(out_path, raw_segment)
            print(
                f"  wrote {out_path.name}  shape={raw_segment.shape}  "
                f"min={raw_segment.min():.3f}  max={raw_segment.max():.3f}"
            )

    print(f"Z-score normalizing {n_export} segments...")
    lst_zscore = zscore_img_segs(clip.proc_tiff_path, ranges_to_export)
    for i, zscore_segment in enumerate(lst_zscore):
        out_path = out_dir / f"segment_{i:02d}_after.tif"
        tifffile.imwrite(out_path, zscore_segment)
        print(
            f"  wrote {out_path.name}  shape={zscore_segment.shape}  "
            f"min={zscore_segment.min():.3f}  max={zscore_segment.max():.3f}"
        )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for normalization in ("GAUSS", "ALS"):
        for rec in RECORDINGS:
            run_recording(rec["tiff_stem"], rec["abf_name"], normalization)


if __name__ == "__main__":
    main()
