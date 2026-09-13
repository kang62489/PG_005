"""
Prototype: 3-row per-segment PNG for each of the two test recordings, at the
production AREA_PCT_SIGMA_MULT=10.0 default.

Row 1: detected mask (categorized critical frame, black/gray/white)
Row 2: detected-cluster contour overlaid on the raw z-scored tiff segment
Row 3: this segment's own Vm spike waveform, aligned to the spike frame

Exported to output/reliability_contours/<recording>/segment_XX_<tag>.png.

Scratch/one-off script -- not part of the reviewed codebase.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from skimage.measure import find_contours

from classes import AbfClip, RegionAnalyzer, SpatialCategorizer
from functions import zscore_img_segs

PROC_TIFFS_DIR = Path("D:/Programs/PG_005/proc_tiffs")
RAW_ABFS_DIR = Path("D:/Programs/PG_005/raw_abfs")
OUT_ROOT = Path("D:/Programs/PG_005/output/reliability_contours")

RECORDINGS = [
    {"tiff_stem": "2025_06_11-0003", "abf_name": "2025_06_11_0004.abf", "obj": "10X"},
    {"tiff_stem": "2025_12_15-0012", "abf_name": "2025_12_15_0008.abf", "obj": "10X"},
]

CMAP_CAT = ListedColormap(["black", "gray", "white"])


def run_recording(tiff_stem: str, abf_name: str, obj: str) -> None:
    print(f"\n=== {tiff_stem} (production sigma=10) ===")
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
    vm_segments = clip.get_vm_segments()
    frame_duration_ms = clip.ts_imgs * 1000
    n_segments = len(lst_zscore)

    for i, segment in enumerate(lst_zscore):
        spike_frame_idx = segment.shape[0] // 2
        n_frames = segment.shape[0]

        categorizer = SpatialCategorizer.morphological(threshold_method="base977_otsu")
        categorizer.fit(segment, spike_frame_idx=spike_frame_idx)
        analyzer = RegionAnalyzer(np.array(categorizer.categorized_frames), segment, spike_frame_idx, obj=obj)
        results = analyzer.get_results()
        detected = analyzer.significant and results["n_clusters"] > 0
        critical_frame_idx = analyzer.critical_frame_idx
        cat_frame = categorizer.categorized_frames[critical_frame_idx]
        raw_frame = segment[critical_frame_idx]
        label_frame = analyzer.label_frame
        time_ms, vm = vm_segments[i]

        tag = "DETECTED" if detected else "no"
        out_path = out_dir / f"segment_{i:02d}_{tag}.png"

        fig, (ax_mask, ax_contour, ax_vm) = plt.subplots(3, 1, figsize=(5, 12))

        # Row 1: detected mask -- categorized critical frame (production black/gray/white)
        ax_mask.imshow(cat_frame, cmap=CMAP_CAT, vmin=0, vmax=2, interpolation="nearest")
        ax_mask.set_title(f"Detected mask -- frame {critical_frame_idx - spike_frame_idx:+d}", fontsize=10)
        ax_mask.axis("off")

        # Row 2: detected-cluster contour on the raw z-scored tiff segment
        ax_contour.imshow(raw_frame, cmap="gray", interpolation="nearest")
        for contour in find_contours(label_frame >= 0, level=0.5):
            ax_contour.plot(contour[:, 1], contour[:, 0], color="#e74c3c", linewidth=1.5)
        ax_contour.set_title("Detected contour on raw segment", fontsize=10)
        ax_contour.axis("off")

        # Row 3: this segment's aligned Vm spike waveform
        ax_vm.plot(time_ms, vm, color="#3498db", linewidth=1.0)
        for offset in range(-4, 6):
            frame_idx = spike_frame_idx + offset
            if 0 <= frame_idx < n_frames:
                color = "#e74c3c" if offset == 0 else "#888888"
                ax_vm.axvline(offset * frame_duration_ms, color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        ax_vm.set_xlabel("Time relative to spike frame (ms)", fontsize=9)
        ax_vm.set_ylabel("Vm (mV)", fontsize=9)
        ax_vm.set_title("Aligned spike (Vm)", fontsize=10)
        ax_vm.tick_params(labelsize=8)

        fig.suptitle(f"{tiff_stem}  segment {i}\n{tag}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        print(f"  segment {i + 1}/{n_segments}: {tag}", flush=True)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for rec in RECORDINGS:
        run_recording(rec["tiff_stem"], rec["abf_name"], rec["obj"])


if __name__ == "__main__":
    main()
