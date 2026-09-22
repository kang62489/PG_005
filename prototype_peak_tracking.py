"""Direct replication of Matityahu et al. 2023's peak-location tracking method
(Methods: "Space-time representation of waves and estimation of wave location").

Their procedure, applied here as literally as the data allows:
1. Freehand ROI around the area of significant signal change.
   -> here: union of CAT-bright pixels across all frames of the recording.
2. Cluster ROI pixels into bands perpendicular to a chosen axis.
3. Per frame, average the signal across each band -> collapses 2D frame to 1D.
4. Stack these 1D profiles over frames -> space-time (kymograph) rendition.
5. Per frame, find the band with maximal signal -> approximate wave location.
6. Temporal derivative of that location -> instantaneous velocity.

One unavoidable adaptation: the paper had a known anatomical axis (mediolateral)
to orient the bands. This data has no predetermined axis, so both axes (rows
collapsed / columns collapsed) are tried and shown side by side -- nothing else
is added beyond what the Methods section describes.

Scratch script only, no pipeline edits.
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
OUT_DIR = PROJECT_ROOT / "output" / "test4" / "peak_tracking"

RECORDINGS = [
    "2025_06_11-0002_A1S4RC1_BIEXP_ALS",
]


def band_profile(frame: np.ndarray, roi_mask: np.ndarray, axis: int) -> np.ndarray:
    """Average signal per band perpendicular to `axis`.

    axis=1 -> bands are columns (average over rows), profile indexed by x.
    axis=0 -> bands are rows (average over columns), profile indexed by y.
    NaN where a band has no ROI pixels.
    """
    masked = np.where(roi_mask, frame, np.nan)
    return np.nanmean(masked, axis=axis)


def run_one(tag: str) -> None:
    print(f"\n=== {tag} ===")
    median_segment = tifffile.imread(MED_DIR / f"{tag}_MED.tif")
    cat_stack = tifffile.imread(CAT_DIR / f"{tag}_CAT.tif")
    n_frames = median_segment.shape[0]
    spike_frame_idx = n_frames // 2
    print(f"MED shape: {median_segment.shape}")

    roi_mask = np.any(cat_stack == CATEGORY_BRIGHT, axis=0)
    print(f"ROI (union of CAT-bright across all {n_frames} frames): {roi_mask.sum()} px")

    fig = Figure(figsize=(14, 8), layout="constrained")
    axes = fig.subplots(2, 2)

    for col, (axis, axis_name) in enumerate([(1, "x (columns)"), (0, "y (rows)")]):
        profiles = np.stack([band_profile(median_segment[t], roi_mask, axis) for t in range(n_frames)])
        # profiles: (n_frames, n_bands)
        peak_loc = np.array([
            np.nanargmax(profiles[t]) if not np.all(np.isnan(profiles[t])) else np.nan
            for t in range(n_frames)
        ], dtype=float)
        velocity = np.diff(peak_loc)

        ax_kymo = axes[0, col]
        im = ax_kymo.imshow(
            profiles.T, aspect="auto", cmap="inferno", origin="lower",
            extent=[0, n_frames - 1, 0, profiles.shape[1]],
        )
        ax_kymo.plot(np.arange(n_frames), peak_loc, "co", ms=4, label="peak location")
        ax_kymo.axvline(spike_frame_idx, color="w", ls="--", lw=1, label="spike frame")
        ax_kymo.set_title(f"Space-time, bands perpendicular to {axis_name}")
        ax_kymo.set_xlabel("frame")
        ax_kymo.set_ylabel("band index")
        ax_kymo.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax_kymo, shrink=0.8, label="mean MED intensity")

        ax_vel = axes[1, col]
        ax_vel.plot(np.arange(1, n_frames), velocity, "k.-")
        ax_vel.axhline(0, color="gray", lw=0.8)
        ax_vel.axvline(spike_frame_idx, color="r", ls="--", lw=1)
        ax_vel.set_title("Instantaneous velocity (d(peak_loc)/d(frame))")
        ax_vel.set_xlabel("frame")
        ax_vel.set_ylabel("band/frame")

        print(f"axis={axis_name}: peak_loc={np.round(peak_loc, 1).tolist()}")

    fig.suptitle(f"{tag} -- peak-location tracking (Matityahu et al. 2023 method)", fontsize=13)
    out_path = OUT_DIR / f"{tag}_PEAKTRACK.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for tag in RECORDINGS:
        run_one(tag)
