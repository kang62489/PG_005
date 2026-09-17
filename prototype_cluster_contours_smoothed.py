"""Compare Gaussian-smoothed CAT cluster boundaries without rerunning categorization.

Standalone duplicate of the contour-arrow prototype. Uses the saved centered CAT
stack, the existing density-gated cluster selection, and nearest-polyline arrows.
Sigma is in pixels; faint outlines show the unsmoothed longest contours.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import find_contours

from classes.region_analyzer import (
    CATEGORY_BRIGHT,
    _run_density_gated_cluster_seeker,
    compute_density_thresh,
    compute_eps_px,
    compute_window_px,
)

PROJECT_ROOT = Path(__file__).resolve().parent
STEM = "2025_06_11-0002"
OBJ = "60X"
CAT_PATH = PROJECT_ROOT / "output/test2/categorized/2025_06_11-0002_A1S4RC1_BIEXP_ALS_CAT.tif"
OUT_DIR = PROJECT_ROOT / "output/cluster_contours"
SIGMAS = (0, 3, 6)


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



def smooth_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Round mask details, with nearest padding to avoid artificial border loss."""
    if sigma == 0:
        return mask.copy()
    return gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest") >= 0.5


def main_contour(mask: np.ndarray) -> np.ndarray:
    """Match the original prototype's longest-contour selection."""
    contours = find_contours(mask.astype(float), 0.5)
    return max(contours, key=len) if contours else np.empty((0, 2))


def draw_pair(ax, original, smoothed, pair: int, sigma: float) -> None:
    """Draw a pair with original outlines and arrows on smoothed outlines."""
    ax.set_facecolor("#202020")
    height, width = original[0].shape
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    for j, (color, style) in enumerate((("white", "-"), ("yellow", "--"))):
        raw = main_contour(original[j])
        curve = main_contour(smoothed[j])
        if sigma:
            ax.plot(raw[:, 1], raw[:, 0], color=color, linewidth=0.65, alpha=0.22)
        ax.plot(curve[:, 1], curve[:, 0], color=color, linestyle=style, linewidth=1.25,
                label=f"spike+{pair + j}", zorder=3)
        if not len(curve):
            ax.text(0.5, 0.5 - j * 0.07, f"No contour: spike+{pair + j}", color=color,
                    transform=ax.transAxes, ha="center")
    arrows = contour_connections(*smoothed)
    for arrow in arrows:
        point, target = arrow["start"], arrow["end"]
        color = "#ff5353" if arrow["grow"] else "#35baff"
        annotation = ax.annotate(
            "", xy=target[::-1], xytext=point[::-1], zorder=5,
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.6,
                        "mutation_scale": 12, "shrinkA": 0, "shrinkB": 0},
        )
        annotation.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.7, foreground="black"), pe.Normal()])
        ax.plot(point[1], point[0], "o", markersize=2.5, color=color, zorder=6)
    ax.set_title(f"sigma = {sigma} px | spike+{pair} -> spike+{pair + 1}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8)
    print(f"sigma={sigma}, pair={pair}: {len(arrows)} arrows")


def run() -> None:
    cat_stack = tifffile.imread(CAT_PATH)
    # ach_domain_analysis builds symmetric spike-centered segments.
    spike_idx = cat_stack.shape[0] // 2
    if cat_stack.ndim != 3 or spike_idx + 4 >= len(cat_stack):
        message = "Expected a centered CAT stack with at least four frames after the spike."
        raise ValueError(message)
    masks = []
    for frame in cat_stack[spike_idx:spike_idx + 5]:
        labels, _, _ = _run_density_gated_cluster_seeker(
            frame == CATEGORY_BRIGHT, compute_eps_px(OBJ), compute_window_px(OBJ),
            compute_density_thresh(OBJ), z_frame=None,
        )
        masks.append(labels >= 0)
    print(f"CAT: {CAT_PATH.name}; spike index: {spike_idx}; objective: {OBJ}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SIGMAS), 4, figsize=(24, 18), layout="constrained")
    for row, sigma in enumerate(SIGMAS):
        smoothed = [smooth_mask(mask, sigma) for mask in masks]
        for pair in range(4):
            draw_pair(axes[row, pair], masks[pair:pair + 2], smoothed[pair:pair + 2], pair, sigma)
        # Separate larger four-panel export for inspecting each smoothing level.
        detail, detail_axes = plt.subplots(2, 2, figsize=(14, 14), layout="constrained")
        for pair, ax in enumerate(detail_axes.flat):
            draw_pair(ax, masks[pair:pair + 2], smoothed[pair:pair + 2], pair, sigma)
        detail.suptitle(f"{STEM} | Gaussian sigma = {sigma} pixels\n"
                        "White -> yellow | red: expansion | blue: contraction\n"
                        "Faint outlines: original | longest contour per frame | geometric connections", fontsize=15)
        path = OUT_DIR / f"contour_smoothing_sigma{sigma}_{STEM[-4:]}.png"
        detail.savefig(path, dpi=160)
        plt.close(detail)
        print(f"Saved: {path}")
    fig.suptitle(f"{STEM}: smoothing comparison from saved CAT masks\n"
                 "Rows: sigma = 0, 3, 6 pixels | white -> yellow | red: expansion | blue: contraction\n"
                 "Faint outlines: original | longest contour per frame | geometric connections, not tracked motion",
                 fontsize=18)
    path = OUT_DIR / f"contour_smoothing_comparison_{STEM[-4:]}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    run()
