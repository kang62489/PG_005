"""Cluster-boundary displacement tracking for spike..spike+4 on 2025_06_11-0002.

Per frame: density-gated cluster detection (same method RegionAnalyzer already uses)
gives the frame's combined hotspot mask (union of all its kept clusters -- individual
cluster identity isn't tracked here, since the measurement below only cares about "how
does the bright region's edge move," not which release site is which).

For every pair of consecutive frames (spike+0->+1, +1->+2, +2->+3, +3->+4), boundary
movement is measured directly from the RELATIVE CHANGE BETWEEN THE TWO ACTUAL CONTOURS
(no fixed grid, no arbitrary reference point):
  1. A sparse set of real points is sampled along frame N's own contour.
  2. Each point's LOCAL OUTWARD DIRECTION is found from the contour's own shape nearby
     (tangent between points some distance before/after it along the boundary, rotated
     90 degrees, oriented outward by checking which side is actually background) --
     this is smooth and stable even on a jagged boundary, unlike a per-pixel intensity
     gradient. An earlier "nearest point on the whole other contour" approach was
     dropped because it doesn't respect the boundary's own order and can produce
     crossing arrows between two nearby points.
  3. Frame N+1's mask is measured along that same local direction (signed-distance
     comparison) to get how far the boundary actually moved there -- red = grew
     outward, blue = shrank inward.

This only needs two clean binary masks (from the pipeline's own CAT + density-gated
clustering, not raw noisy intensity), so it sidesteps the texture/correspondence
problems that broke PIV/optical-flow on this data entirely.

Scratch script only -- no pipeline edits.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import find_contours
from skimage.measure import label as sk_label
from skimage.morphology import binary_closing, binary_opening, disk

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


SMOOTH_RADIUS_PX = 20  # morphological closing+opening radius -- removes small notches/holes so the
# boundary measurement matches the "big picture" shape a human eye reads, not every jagged pixel


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological closing then opening -- fills small notches/holes, trims small spikes.

    Without this, a signed-distance measurement at a boundary point can be thrown off by
    a tiny nearby notch in the mask's own jagged edge, even when the overall shape has
    clearly receded much farther everywhere else -- exactly the mismatch you flagged
    between the tiny computed arrows and the obviously large visible gap in the plot.
    """
    footprint = disk(SMOOTH_RADIUS_PX)
    closed = binary_closing(mask, footprint)
    opened = binary_opening(closed, footprint)
    return largest_filled_component(opened)


def largest_filled_component(mask: np.ndarray) -> np.ndarray:
    """Largest connected region, with any internal holes filled.

    Without this, find_contours can return an internal hole's boundary as the
    "longest" contour instead of the actual outer envelope -- what looked like a
    broken/floating yellow loop in the plot was exactly that: a hole's edge, not
    the real boundary.
    """
    labeled = sk_label(mask)
    if labeled.max() == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    largest_label = int(np.argmax(sizes))
    largest_component = labeled == largest_label
    return binary_fill_holes(largest_component)


N_SAMPLES = 20  # number of points sampled along the contour -- keeps the arrow count readable
TANGENT_DELTA = 20  # contour-index offset used to estimate each sample point's local direction
NORMAL_TEST_PX = 5  # how far to step off the boundary to tell which side is actually background


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance transform: positive outside mask, negative inside, zero on the boundary."""
    return distance_transform_edt(~mask) - distance_transform_edt(mask)


def local_outward_normal(contour: np.ndarray, idx: int, mask: np.ndarray) -> np.ndarray | None:
    """Outward direction at contour[idx], from the boundary's own local shape (not a pixel gradient).

    Uses the tangent between two points some distance before/after idx along the ordered
    contour (smooths over small jagged wiggles that would make a per-pixel gradient noisy),
    rotates it 90 degrees to get a perpendicular, then picks whichever of the two possible
    perpendiculars actually steps into the background -- so it's always genuinely outward,
    even on a concave/irregular shape, rather than assumed from a global center point.

    Indices are clipped, not wrapped -- `contour` may be an OPEN fragment (a mask that
    touches the image border gets its boundary split into several open segments by
    find_contours, since the true full outline can't close within the frame), and
    wrapping around would incorrectly jump to the fragment's other end.
    """
    n = len(contour)
    p_prev = contour[max(idx - TANGENT_DELTA, 0)]
    p_next = contour[min(idx + TANGENT_DELTA, n - 1)]
    tangent = p_next - p_prev
    t_norm = np.hypot(*tangent)
    if t_norm < 1e-8:
        return None
    tangent = tangent / t_norm
    normal_a = np.array([-tangent[1], tangent[0]])
    normal_b = -normal_a

    point = contour[idx]
    h, w = mask.shape

    def is_background(n_vec: np.ndarray) -> bool:
        r, c = point + n_vec * NORMAL_TEST_PX
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < h and 0 <= ci < w:
            return not mask[ri, ci]
        return True

    if is_background(normal_a):
        return normal_a
    if is_background(normal_b):
        return normal_b
    return None  # ambiguous (both sides still inside the mask) -- skip this point


def sector_boundary_displacements(mask_a: np.ndarray, mask_b: np.ndarray) -> list[dict]:
    """Individual displacement vectors at a sparse set of REAL points on mask_a's contour.

    Each sampled point is an actual point on the white (frame A) boundary. Its direction
    comes from the boundary's own local shape (local_outward_normal), and its length is
    how far frame B's mask actually extends along that same direction (signed-distance
    comparison) -- so it always shows a real, physically meaningful displacement without
    ever crossing another arrow, since direction always follows the boundary's own local
    order rather than an unconstrained nearest-point search across the whole other
    contour (which an earlier attempt used, and which could match two nearby points to
    swapped targets, producing arrows that visibly crossed each other).

    Returns:
        List of dicts: {row, col, disp_row, disp_col, signed_speed_px} -- (row, col) is
        the real sampled point on mask_a's contour (arrow start), (disp_row, disp_col)
        is the real displacement vector to draw, signed_speed_px is its magnitude
        (positive = grew outward, negative = shrank inward) for growth/shrink coloring.
    """
    contours_a = find_contours(mask_a.astype(float), level=0.5)
    if not contours_a:
        return []
    sdf_b = signed_distance(mask_b)

    # A mask touching the image border gets split into several OPEN fragments (its true
    # full outline can't close within the frame) -- sample from every fragment, weighted
    # by length, instead of only the single longest one, so no part of the boundary
    # (and no part of a border-touching frame's real edge) is silently skipped.
    total_len = sum(len(c) for c in contours_a)
    results = []
    for contour_a in contours_a:
        n_samples_here = max(1, round(N_SAMPLES * len(contour_a) / total_len))
        sample_idx = np.linspace(0, len(contour_a) - 1, n_samples_here, dtype=int)
        for idx in sample_idx:
            row, col = contour_a[idx]
            normal = local_outward_normal(contour_a, idx, mask_a)
            if normal is None:
                continue
            ri, ci = int(np.clip(round(row), 0, mask_a.shape[0] - 1)), int(np.clip(round(col), 0, mask_a.shape[1] - 1))
            # A point on A's boundary now INSIDE B (sdf_b < 0) means B has grown past it --
            # outward displacement is positive there, hence the negation.
            signed_speed = -float(sdf_b[ri, ci])
            results.append({
                "row": float(row),
                "col": float(col),
                "disp_row": float(normal[0] * signed_speed),
                "disp_col": float(normal[1] * signed_speed),
                "signed_speed_px": signed_speed,
            })
    return results


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cat_stack, med_stack, frame_idxs, frame_offsets, um_per_px, frame_duration_ms = load_stack()
    dt_s = frame_duration_ms / 1000

    masks = [smooth_mask(combined_mask(cat_stack, med_stack, idx)) for idx in frame_idxs]
    pairs = list(range(len(frame_offsets) - 1))  # index pairs (i, i+1) into frame_offsets/masks
    print(f"{len(frame_offsets)} frames (spike{frame_offsets[0]:+d}..spike{frame_offsets[-1]:+d}) -> {len(pairs)} consecutive pair(s).")

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]

    for ax, i in zip(axes, pairs, strict=True):
        off_a, off_b = frame_offsets[i], frame_offsets[i + 1]
        mask_a, mask_b = masks[i], masks[i + 1]
        sectors = sector_boundary_displacements(mask_a, mask_b)

        ax.imshow(med_stack[frame_idxs[i]], cmap="gray")
        for j, c in enumerate(find_contours(mask_a.astype(float), level=0.5)):
            ax.plot(c[:, 1], c[:, 0], color="white", linewidth=1.3, label=f"spike{off_a:+d} edge" if j == 0 else None)
        for j, c in enumerate(find_contours(mask_b.astype(float), level=0.5)):
            ax.plot(c[:, 1], c[:, 0], color="yellow", linewidth=1.3, linestyle="--", label=f"spike{off_b:+d} edge" if j == 0 else None)
        ax.legend(fontsize=7, loc="lower right")

        disp_um_all = []
        for s in sectors:
            disp_um = s["signed_speed_px"] * um_per_px
            disp_um_all.append(disp_um)
            color = "tab:red" if s["signed_speed_px"] > 0 else "tab:blue"
            end_row = s["row"] + s["disp_row"]
            end_col = s["col"] + s["disp_col"]
            ax.annotate(
                "", xy=(end_col, end_row), xytext=(s["col"], s["row"]),
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2.2},
            )
            ax.text(
                s["col"], s["row"], f"{disp_um:.0f}",
                color=color, fontsize=8, fontweight="bold", ha="center", va="bottom",
            )

        if disp_um_all:
            median_speed = np.median(np.abs(disp_um_all)) / dt_s
            n_growing = sum(1 for d in disp_um_all if d > 0)
            n_shrinking = sum(1 for d in disp_um_all if d < 0)
            print(
                f"  spike{off_a:+d}->spike{off_b:+d}: {len(disp_um_all)} points, "
                f"{n_growing} growing / {n_shrinking} shrinking, "
                f"median |speed| = {median_speed:.1f} um/s"
            )

        ax.set_title(f"spike{off_a:+d} -> spike{off_b:+d}  (red=grow, blue=shrink)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{STEM}: sampled boundary-point displacement (relative contour change)", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / f"sector_displacement_{STEM[-4:]}.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
