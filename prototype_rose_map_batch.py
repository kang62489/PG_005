"""Rose-map prototype, batched over several recordings, spike..spike+4.
Scratch script only -- no pipeline edits.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path("D:/Programs/PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

from classes import AbfClip, SpatialCategorizer, SpikeReliabilityChecker  # noqa: E402
from classes.spatial_categorization import CATEGORY_BRIGHT  # noqa: E402
from functions import load_img_segs, spike_centered_median  # noqa: E402

N_SECTORS = 8
SECTOR_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
SECTOR_CENTERS = np.deg2rad(np.arange(0, 360, 45))
SECTOR_EDGES = np.deg2rad(np.arange(-22.5, 360, 45))

RECORDINGS = [
    # (stem, abf_name, obj)
    ("2025_06_11-0002", "2025_06_11_0003.abf", "60X"),
    ("2025_06_11-0003", "2025_06_11_0004.abf", "10X"),
    ("2025_11_13-0017", "2025_11_13_0011.abf", "40X"),
    ("2025_12_15-0012", "2025_12_15_0008.abf", "10X"),
]


def rose_histogram(bright_mask: np.ndarray, row_c: float, col_c: float) -> tuple[np.ndarray, float, float]:
    rows, cols = np.nonzero(bright_mask)
    if rows.size == 0:
        return np.zeros(N_SECTORS), 0.0, 0.0
    dr = rows.astype(np.float64) - row_c
    dc = cols.astype(np.float64) - col_c
    angles = np.arctan2(-dr, dc)
    angles_wrapped = np.mod(angles + np.pi / 8, 2 * np.pi) - np.pi / 8
    counts, _ = np.histogram(angles_wrapped, bins=SECTOR_EDGES)
    shares = counts / counts.sum()
    vec = np.sum(shares * np.exp(1j * SECTOR_CENTERS))
    R = float(np.abs(vec))
    mean_angle = float(np.angle(vec))
    return shares, R, mean_angle


def run_one(stem: str, abf_name: str, obj: str) -> None:
    print(f"\n=== {stem} ({obj}) ===")
    clip = AbfClip(
        proc_tiff_path=PROJECT_ROOT / "proc_tiffs" / f"{stem}_BIEXP_ALS.tif",
        raw_abf_path=PROJECT_ROOT / "raw_abfs" / abf_name,
        results_dir=PROJECT_ROOT / "results",
        detrend_mode="BIEXP",
        normalization="ALS",
    )
    lst_segments = load_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
    spike_frame_idx = lst_segments[0].shape[0] // 2

    checker = SpikeReliabilityChecker(obj)
    seg_results, reliability_pct = checker.check(lst_segments, spike_frame_idx)
    print(f"Reliability: {sum(r['detected'] for r in seg_results)}/{len(seg_results)} ({reliability_pct:.1f}%)")

    segments_for_median = [
        s for s, r in zip(lst_segments, seg_results, strict=True) if r["detected"]
    ] or lst_segments
    median_segment, _ = spike_centered_median(segments_for_median)

    categorizer = SpatialCategorizer.morphological(threshold_method="baseline_n_sigma")
    categorizer.fit(median_segment, spike_frame_idx=spike_frame_idx)
    cat_stack = np.array(categorizer.categorized_frames)

    spike_bright = cat_stack[spike_frame_idx] == CATEGORY_BRIGHT
    rows0, cols0 = np.nonzero(spike_bright)
    if rows0.size == 0:
        print("No bright pixels at spike frame -- skipping.")
        return
    row_c, col_c = float(rows0.mean()), float(cols0.mean())

    frame_idxs = [idx for idx in range(spike_frame_idx, spike_frame_idx + 5) if idx < cat_stack.shape[0]]

    results = []
    for idx in frame_idxs:
        bright = cat_stack[idx] == CATEGORY_BRIGHT
        shares, R, mean_angle = rose_histogram(bright, row_c, col_c)
        results.append((idx, shares, R, mean_angle, int(bright.sum())))
        print(
            f"frame {idx - spike_frame_idx:+d}: n_bright={int(bright.sum()):6d}  R={R:.3f}  "
            f"mean_angle={np.degrees(mean_angle):6.1f} deg  "
            f"shares=" + ", ".join(f"{sl}={s:.2f}" for sl, s in zip(SECTOR_LABELS, shares, strict=True))
        )

    n_cols = len(results)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.4, 1.4, 1.2], hspace=0.5, wspace=0.15)
    cmap_cat = ListedColormap(["black", "white"])
    width = 2 * np.pi / N_SECTORS

    for i, (idx, shares, R, mean_angle, n_bright) in enumerate(results):
        offset = idx - spike_frame_idx
        label = "SPIKE" if offset == 0 else f"spike+{offset}"

        ax0 = fig.add_subplot(gs[0, i])
        ax0.imshow(cat_stack[idx] == CATEGORY_BRIGHT, cmap=cmap_cat, vmin=0, vmax=1, interpolation="nearest")
        ax0.plot(col_c, row_c, marker="x", color="red", markersize=8, markeredgewidth=2)
        ax0.set_title(f"{label}\nn_bright={n_bright}", fontsize=9, fontweight="bold" if offset == 0 else "normal")
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = fig.add_subplot(gs[1, i], projection="polar")
        ax1.bar(SECTOR_CENTERS, shares, width=width, bottom=0.0, edgecolor="k", alpha=0.7)
        ax1.set_theta_zero_location("E")
        ax1.set_theta_direction(1)
        ax1.set_title(f"R={R:.2f}", fontsize=9)
        ax1.set_ylim(0, 0.5)

    ax2 = fig.add_subplot(gs[2, :])
    offsets = [idx - spike_frame_idx for idx, *_ in results]
    share_matrix = np.array([r[1] for r in results])
    for s in range(N_SECTORS):
        ax2.plot(offsets, share_matrix[:, s], marker="o", label=SECTOR_LABELS[s])
    ax2.set_xlabel("Frame offset from spike")
    ax2.set_ylabel("Sector share of bright pixels")
    ax2.legend(ncol=8, fontsize=8)
    ax2.set_title("Per-direction share vs. frame (origin fixed at spike-frame centroid)")

    fig.suptitle(f"{stem} rose-map prototype ({obj})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = PROJECT_ROOT / "output" / "test2" / f"rose_map_prototype_{stem[-4:]}.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for stem, abf_name, obj in RECORDINGS:
        run_one(stem, abf_name, obj)
