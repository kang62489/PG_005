"""
Prototype: print the base977_otsu thresholds (thresh_dim, thresh_bright)
for the median segment of the two test recordings.

Scratch/one-off script -- not part of the reviewed codebase.
"""

from pathlib import Path

from classes import AbfClip, SpatialCategorizer
from functions import spike_centered_median, zscore_img_segs

PROC_TIFFS_DIR = Path("D:/Programs/PG_005/proc_tiffs")
RAW_ABFS_DIR = Path("D:/Programs/PG_005/raw_abfs")
OUT_ROOT = Path("D:/Programs/PG_005/output/reliability_sigma2")

RECORDINGS = [
    {"tiff_stem": "2025_06_11-0003", "abf_name": "2025_06_11_0004.abf"},
    {"tiff_stem": "2025_12_15-0012", "abf_name": "2025_12_15_0008.abf"},
]


def run_recording(tiff_stem: str, abf_name: str) -> None:
    print(f"\n=== {tiff_stem} ===")
    proc_tiff_path = PROC_TIFFS_DIR / f"{tiff_stem}_BIEXP_GAUSS.tif"
    raw_abf_path = RAW_ABFS_DIR / abf_name
    out_dir = OUT_ROOT / tiff_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    clip = AbfClip(
        proc_tiff_path=proc_tiff_path,
        raw_abf_path=raw_abf_path,
        results_dir=out_dir,
        detrend_mode="BIEXP",
        normalization="GAUSS",
    )
    if not clip.lst_img_frame_ranges:
        print("No valid segments -- skipping.")
        return

    print("Z-score normalizing segments...")
    lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
    median_segment, _ = spike_centered_median(lst_zscore)
    spike_frame_idx = median_segment.shape[0] // 2

    categorizer = SpatialCategorizer.morphological(threshold_method="base977_otsu")
    categorizer.fit(median_segment, spike_frame_idx=spike_frame_idx)
    thresh_dim, thresh_bright = categorizer.thresholds_used
    print(f"thresh_dim (baseline mean + 2*std):    {thresh_dim:.4f}")
    print(f"thresh_bright (Otsu on signal pixels): {thresh_bright:.4f}")


def main() -> None:
    for rec in RECORDINGS:
        run_recording(rec["tiff_stem"], rec["abf_name"])


if __name__ == "__main__":
    main()
