"""MED height-map (3D surface) prototype, spike..spike+4.
Scratch script only -- no pipeline edits.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.figure import Figure

PROJECT_ROOT = Path("D:/Programs/PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

CATEGORY_BRIGHT = 1

MED_DIR = PROJECT_ROOT / "output" / "test4" / "median"
CAT_DIR = PROJECT_ROOT / "output" / "test4" / "categorized"

RECORDINGS = [
    # tag used in the *_MED.tif / *_CAT.tif filenames
    "2025_06_11-0002_A1S4RC1_BIEXP_ALS",
    "2025_06_11-0003_A1S4RC1_BIEXP_ALS",
]


def plot_med_height_map_series(
    med_stack: np.ndarray,
    spike_frame_idx: int,
    n_frames: int,
    title: str,
    bright_mask_stack: np.ndarray | None = None,
    z_label: str = "MED intensity",
    cmap: str = "inferno",
    elev: float = 35.0,
    azim: float = -60.0,
    stride: int = 4,
) -> Figure:
    """3D intensity height-map (surface) panels for spike-aligned MED frames.

    One Axes3D panel per frame in [spike_frame_idx, spike_frame_idx + n_frames),
    clamped to med_stack bounds. z = med_stack's own values as given (caller decides
    raw MED intensity vs. baseline z-score, etc.) -- this function doesn't transform
    them. Shared z-limits and color scale across panels so hill height is directly
    comparable frame-to-frame, not diluted by per-panel auto-scaling.

    If bright_mask_stack is given (same shape as med_stack, True = CAT-bright
    pixel), non-bright pixels are floored to the shared vmin per frame so
    background speckle noise no longer registers as height -- display-only,
    the underlying med_stack values are never modified.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the "3d" projection)

    frame_idxs = [idx for idx in range(spike_frame_idx, spike_frame_idx + n_frames) if idx < med_stack.shape[0]]
    height, width = med_stack.shape[1], med_stack.shape[2]
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    selected = med_stack[frame_idxs]
    vmin, vmax = float(np.nanmin(selected)), float(np.nanmax(selected))

    fig = Figure(figsize=(5 * len(frame_idxs), 5), layout="constrained")
    surf = None
    for i, idx in enumerate(frame_idxs):
        offset = idx - spike_frame_idx
        frame = med_stack[idx]
        if bright_mask_stack is not None:
            frame = np.where(bright_mask_stack[idx], frame, vmin)
        ax = fig.add_subplot(1, len(frame_idxs), i + 1, projection="3d")
        surf = ax.plot_surface(
            x, y, frame, cmap=cmap, vmin=vmin, vmax=vmax,
            rstride=stride, cstride=stride, antialiased=False,
        )
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zlabel(z_label, fontsize=8)
        frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
        ax.set_title(frame_label, fontsize=10, fontweight="bold" if offset == 0 else "normal")

    fig.colorbar(surf, ax=fig.axes[-1], shrink=0.6, label=z_label)
    mask_note = " | non-bright (CAT) pixels floored to vmin" if bright_mask_stack is not None else ""
    fig.suptitle(title + f"\nz = {z_label}, shared color/z-scale across panels{mask_note}", fontsize=13)
    return fig


def run_one(tag: str, use_mask: bool) -> None:
    print(f"\n=== {tag} (mask={use_mask}) ===")
    median_segment = tifffile.imread(MED_DIR / f"{tag}_MED.tif")
    spike_frame_idx = median_segment.shape[0] // 2
    print(f"MED shape: {median_segment.shape}, intensity range: [{median_segment.min():.3f}, {median_segment.max():.3f}]")

    bright_mask_stack = None
    suffix = "_NOMASK"
    if use_mask:
        cat_stack = tifffile.imread(CAT_DIR / f"{tag}_CAT.tif")
        bright_mask_stack = cat_stack == CATEGORY_BRIGHT
        suffix = ""

    fig = plot_med_height_map_series(
        median_segment, spike_frame_idx, n_frames=5, title=f"{tag} MED height-map",
        bright_mask_stack=bright_mask_stack,
    )
    out_path = PROJECT_ROOT / "output" / "test4" / "med_height_map" / f"{tag}_HEIGHTMAP{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for tag in RECORDINGS:
        run_one(tag, use_mask=True)
