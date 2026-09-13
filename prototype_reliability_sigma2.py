"""
Prototype: per-trial reliability analysis with AREA_PCT_SIGMA_MULT=2.0
instead of the production default (10.0).

Runs each individual per-spike segment (not just the median) through the
existing SpatialCategorizer + RegionAnalyzer detection chain, at a 2-sigma
significance bar, on the same 2 real recordings used in the Session 53
prototype (2025_06_11-0003, 2025_12_15-0012). Exports per-segment plots and
a summary to output/reliability_sigma2/ for comparison against the earlier
output/reliability_prototype/ (10-sigma) run.

Scratch/one-off script -- not part of the reviewed codebase.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import classes.region_analyzer as region_analyzer_module
from classes import AbfClip, RegionAnalyzer, SpatialCategorizer
from functions import spike_centered_median, zscore_img_segs

CMAP_CAT = ListedColormap(["black", "gray", "white"])

SIGMA_MULT = 2.0
region_analyzer_module.AREA_PCT_SIGMA_MULT = SIGMA_MULT

PROC_TIFFS_DIR = Path("D:/Programs/PG_005/proc_tiffs")
RAW_ABFS_DIR = Path("D:/Programs/PG_005/raw_abfs")
OUT_ROOT = Path("D:/Programs/PG_005/output/reliability_sigma2")

RECORDINGS = [
    {"tiff_stem": "2025_06_11-0003", "abf_name": "2025_06_11_0004.abf", "obj": "10X"},
    {"tiff_stem": "2025_12_15-0012", "abf_name": "2025_12_15_0008.abf", "obj": "10X"},
]


def analyze_segment(segment: np.ndarray, obj: str) -> tuple[bool, int, float, np.ndarray]:
    """Run categorize + region analysis on one segment. Returns (detected, n_clusters, critical_B_pct, critical_cat_frame)."""
    spike_frame_idx = segment.shape[0] // 2
    categorizer = SpatialCategorizer.morphological(threshold_method="base977_otsu")
    categorizer.fit(segment, spike_frame_idx=spike_frame_idx)
    analyzer = RegionAnalyzer(np.array(categorizer.categorized_frames), segment, spike_frame_idx, obj=obj)
    results = analyzer.get_results()
    detected = analyzer.significant and results["n_clusters"] > 0
    critical_cat_frame = categorizer.categorized_frames[analyzer.critical_frame_idx]
    return detected, results["n_clusters"], results["critical_frame_area_pct"], critical_cat_frame


def save_segment_plot(cat_frame: np.ndarray, detected: bool, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cat_frame, cmap=CMAP_CAT, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(f"{title}\n{'DETECTED' if detected else 'no'}", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def run_recording(tiff_stem: str, abf_name: str, obj: str) -> None:
    print(f"\n=== {tiff_stem} (sigma={SIGMA_MULT}) ===")
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

    print("Z-score normalizing segments (numba JIT compile on first call takes a bit)...")
    lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
    n_segments = len(lst_zscore)
    print(f"Z-score done. Running categorize+cluster on {n_segments} segments...")

    n_detected = 0
    per_segment_lines = []
    for i, segment in enumerate(lst_zscore):
        detected, n_clusters, b_pct, cat_frame = analyze_segment(segment, obj)
        n_detected += int(detected)
        tag = "DETECTED" if detected else "no"
        save_segment_plot(cat_frame, detected, out_dir / f"segment_{i:02d}_{tag}.png", f"segment {i}")
        per_segment_lines.append(f"  segment {i:02d}: {tag:8s}  n_clusters={n_clusters}  B%={b_pct:.3f}")
        print(f"  segment {i + 1}/{n_segments}: {tag}  (n_clusters={n_clusters}, B%={b_pct:.3f})", flush=True)

    median_segment, zscore_range = spike_centered_median(lst_zscore)
    median_detected, median_n_clusters, median_b_pct, median_cat_frame = analyze_segment(median_segment, obj)
    save_segment_plot(
        median_cat_frame, median_detected, out_dir / f"MEDIAN_{'DETECTED' if median_detected else 'no'}.png", "median"
    )

    reliability_pct = 100.0 * n_detected / n_segments
    summary = (
        f"Recording: {tiff_stem}\n"
        f"AREA_PCT_SIGMA_MULT: {SIGMA_MULT}\n"
        f"Per-trial reliability: {n_detected}/{n_segments} = {reliability_pct:.1f}%\n"
        f"Median: {'DETECTED' if median_detected else 'no'}  "
        f"n_clusters={median_n_clusters}  B%={median_b_pct:.3f}  "
        f"z-score range=[{zscore_range[0]:.2f}, {zscore_range[1]:.2f}]\n\n"
        "Per-segment detail:\n" + "\n".join(per_segment_lines) + "\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for rec in RECORDINGS:
        run_recording(rec["tiff_stem"], rec["abf_name"], rec["obj"])


if __name__ == "__main__":
    main()
