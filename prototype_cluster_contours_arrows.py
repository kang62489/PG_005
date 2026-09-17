"""Standalone contour-arrow variant; preserves the original prototype.

Arrows connect sampled white boundary points to exact nearest yellow polyline
points. These are geometric distances, not tracked material motion. Uses the
longest contour per frame, matching the original prototype's displayed edges.
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

PROJECT_ROOT = Path("D:/Programs/PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

from classes import AbfClip, SpatialCategorizer, SpikeReliabilityChecker  # noqa: E402
from classes.region_analyzer import (  # noqa: E402
    CATEGORY_BRIGHT,
    PIXEL_SCALE,
    _run_density_gated_cluster_seeker,
    compute_density_thresh,
    compute_eps_px,
    compute_window_px,
)
from functions import load_img_segs, spike_centered_median  # noqa: E402

STEM = "2025_06_11-0002"
ABF_NAME = "2025_06_11_0003.abf"
OBJ = "60X"
OUT_DIR = PROJECT_ROOT / "output" / "cluster_contours"

N_FRAMES = 5  # spike .. spike+4


def load_stack() -> tuple[np.ndarray, np.ndarray, list[int], list[int], float, float]:
    clip = AbfClip(
        proc_tiff_path=PROJECT_ROOT / "proc_tiffs" / f"{STEM}_BIEXP_ALS.tif",
        raw_abf_path=PROJECT_ROOT / "raw_abfs" / ABF_NAME,
        results_dir=OUT_DIR / "spikes",
        detrend_mode="BIEXP",
        normalization="ALS",
    )
    lst_segments = load_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
    spike_frame_idx = lst_segments[0].shape[0] // 2

    checker = SpikeReliabilityChecker(OBJ)
    seg_results, reliability_pct = checker.check(lst_segments, spike_frame_idx)
    print(f"Reliability: {sum(r['detected'] for r in seg_results)}/{len(seg_results)} ({reliability_pct:.1f}%)")

    segments_for_median = [
        s for s, r in zip(lst_segments, seg_results, strict=True) if r["detected"]
    ] or lst_segments
    median_segment, _ = spike_centered_median(segments_for_median)

    categorizer = SpatialCategorizer.morphological(threshold_method="baseline_n_sigma")
    categorizer.fit(median_segment, spike_frame_idx=spike_frame_idx)
    cat_stack = np.array(categorizer.categorized_frames)

    frame_idxs = [idx for idx in range(spike_frame_idx, spike_frame_idx + N_FRAMES) if idx < cat_stack.shape[0]]
    frame_offsets = [idx - spike_frame_idx for idx in frame_idxs]

    um_per_px = 1.0 / PIXEL_SCALE[OBJ]
    frame_duration_ms = clip.ts_imgs * 1000
    return cat_stack, median_segment, frame_idxs, frame_offsets, um_per_px, frame_duration_ms


def combined_mask(cat_stack: np.ndarray, med_stack: np.ndarray, frame_idx: int) -> np.ndarray:
    """Union of all density-gated clusters on one frame (same method RegionAnalyzer uses).

    Individual cluster identity isn't kept -- the grid-line measurement below only
    needs "is this pixel part of a real hotspot," not which specific release site.
    """
    eps_px = compute_eps_px(OBJ)
    window_px = compute_window_px(OBJ)
    density_thresh = compute_density_thresh(OBJ)
    bright_mask = cat_stack[frame_idx] == CATEGORY_BRIGHT
    label_frame, _centroids, _n_raw = _run_density_gated_cluster_seeker(
        bright_mask, eps_px, window_px, density_thresh, z_frame=med_stack[frame_idx]
    )
    return label_frame >= 0


MAX_ARROWS = 22


def contour_connections(mask_a: np.ndarray, mask_b: np.ndarray) -> list[dict]:
    """Return spaced, exact boundary connections with mask-based growth signs."""
    contours_a = find_contours(mask_a.astype(float), 0.5)
    contours_b = find_contours(mask_b.astype(float), 0.5)
    if not contours_a or not contours_b:
        return []
    contours_a = [max(contours_a, key=len)]
    contours_b = [max(contours_b, key=len)]
    starts = np.concatenate([c[:-1] for c in contours_b])
    vectors = np.concatenate([np.diff(c, axis=0) for c in contours_b])
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    candidates = []
    spacing = max(mask_a.shape) / 12
    for contour in contours_a:
        lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(lengths)]
        if arc[-1] < spacing:
            continue
        for distance in np.arange(spacing / 2, arc[-1], spacing / 2):
            point = np.array([np.interp(distance, arc, contour[:, j]) for j in range(2)])
            fraction = np.clip(
                np.einsum("ij,ij->i", point - starts, vectors)
                / np.maximum(squared_lengths, 1e-12), 0, 1,
            )
            projections = starts + fraction[:, None] * vectors
            distances = np.linalg.norm(projections - point, axis=1)
            index = np.argmin(distances)
            if distances[index] < 3:
                continue
            target = projections[index]
            # Inside the new mask at the old edge means local expansion.
            inside_new = map_coordinates(mask_b.astype(float), point[:, None], order=1)[0] > 0.5
            # Reject connections that leave the gained/lost region along the way.
            path = point[:, None] + (target - point)[:, None] * np.linspace(0.05, 0.95, 25)
            old_values = map_coordinates(mask_a.astype(float), path, order=1)
            new_values = map_coordinates(mask_b.astype(float), path, order=1)
            changed = (new_values > old_values) if inside_new else (old_values > new_values)
            if np.mean(changed) < 0.9:
                continue
            candidates.append({"start": point, "end": target, "grow": inside_new,
                               "distance": float(distances[index])})
    # Prefer clear connections while preventing clustered tails and convergent heads.
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["distance"], reverse=True):
        if any(np.linalg.norm(candidate["start"] - s["start"]) < spacing * 1.1
               or np.linalg.norm(candidate["end"] - s["end"]) < spacing * 0.5 for s in selected):
            continue
        selected.append(candidate)
        if len(selected) == MAX_ARROWS:
            break
    return selected


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat_stack, med_stack, frame_idxs, frame_offsets, _, _ = load_stack()
    masks = [combined_mask(cat_stack, med_stack, idx) for idx in frame_idxs]
    n_pairs = len(masks) - 1
    if n_pairs < 1:
        print("Need at least two frames.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), squeeze=False)
    for ax in axes.flat[n_pairs:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes.flat[:n_pairs]):
        mask_a, mask_b = masks[i:i + 2]
        ax.imshow(med_stack[frame_idxs[i]], cmap="gray")
        for mask, color, style, offset in (
            (mask_a, "white", "-", frame_offsets[i]),
            (mask_b, "yellow", "--", frame_offsets[i + 1]),
        ):
            contours = find_contours(mask.astype(float), 0.5)
            contours = [max(contours, key=len)] if contours else []
            for j, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], color=color, linestyle=style, linewidth=1.2,
                        label=f"spike{offset:+d}" if j == 0 else None, zorder=3)
        connections = contour_connections(mask_a, mask_b)
        for connection in connections:
            point, target = connection["start"], connection["end"]
            color = "#ff5353" if connection["grow"] else "#35baff"
            annotation = ax.annotate(
                "", xy=target[::-1], xytext=point[::-1], zorder=5,
                arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.8,
                            "mutation_scale": 14, "shrinkA": 0, "shrinkB": 0},
            )
            annotation.arrow_patch.set_path_effects([pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
            ax.plot(point[1], point[0], "o", markersize=3, color=color, zorder=6)
        ax.set_title(f"spike{frame_offsets[i]:+d} -> spike{frame_offsets[i + 1]:+d}", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"Pair {i}: {len(connections)} white-to-yellow arrows")
    fig.suptitle(f"{STEM}: contour changes\nWhite -> yellow | red: expansion | blue: contraction", fontsize=18)
    fig.text(0.5, 0.015, "Dots: white-contour origins; arrow tips: yellow contour. Geometric connections, not tracked motion.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_path = OUT_DIR / f"contour_connections_{STEM[-4:]}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
