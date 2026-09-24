"""Spike-aligned TV-L1 optical flow, blurred (production) data, pre-masked.

Scratch script only -- no pipeline edits. Narrowed scope after earlier
iterations ruled out several variants:
  - No-blur reprocessing: dropped -- the whole point was working around TV-L1's
    inability to find texture in a blurred blob, but the production CAT mask
    (and everything downstream of it) is built on blurred data, so comparing
    against it needs blurred data too.
  - Cumulative (direct spike->spike+4) flow: dropped -- redundant once the
    per-step pairs are trusted; added complexity without a clear use.
  - Unmasked display: dropped -- the point of masking is to only look at the
    hotspot; showing the unmasked field was only useful for the earlier
    background-leakage diagnosis, which is now resolved by masking BEFORE
    flow computation instead of after.

Key finding this scope reflects: TV-L1's global regularization pulls flow
estimates from background pixels into the "masked" region if masking is only
applied to the DISPLAY after computing flow on the full frame -- see
premasked_flow(). Masking before computation floors non-bright pixels to
background first, so flow inside the mask only reflects the bright region's
own content.

Pairs analyzed per recording: spike-1->spike, then spike->spike+1 through
spike+3->spike+4 (5 panels total).
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.figure import Figure
from scipy.ndimage import maximum_filter
from skimage.registration import optical_flow_tvl1

PROJECT_ROOT = Path("D:/Programs/PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

CATEGORY_BRIGHT = 1
MED_DIR = PROJECT_ROOT / "output" / "test4" / "median"
CAT_DIR = PROJECT_ROOT / "output" / "test4" / "categorized"
FLOW_DIR = PROJECT_ROOT / "output" / "test5" / "optical_flow"

QUIVER_STEP = 24
QUIVER_SCALE = 0.8  # arrow_length_px = magnitude / QUIVER_SCALE

RECORDINGS = [
    # tag used in the *_MED.tif / *_CAT.tif filenames
    "2025_06_11-0002_A1S4RC1_BIEXP_ALS",
    "2025_06_11-0003_A1S4RC1_BIEXP_ALS",
    "2025_11_13-0017_A1S1RC2_BIEXP_ALS",
    "2025_11_13-0018_A1S1RC2_BIEXP_ALS",
    "2025_12_15-0012_A1S3RC1_BIEXP_ALS",
    "2025_12_15-0013_A1S3RC1_BIEXP_ALS",
]


def mask_frame(frame: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Floor everything outside `keep` to this frame's own min."""
    return np.where(keep, frame, float(frame.min()))


def premasked_flow(med: np.ndarray, cat: np.ndarray, idx_from: int, idx_to: int) -> tuple[np.ndarray, np.ndarray]:
    """TV-L1 flow computed on frames pre-masked to the union of both frames' CAT-bright pixels.

    Masking before computation (rather than after, on the full-frame flow result)
    keeps TV-L1's global regularization from pulling background motion into the
    displayed hotspot region.
    """
    keep = (cat[idx_from] == CATEGORY_BRIGHT) | (cat[idx_to] == CATEGORY_BRIGHT)
    frame_from = mask_frame(med[idx_from], keep)
    frame_to = mask_frame(med[idx_to], keep)
    return optical_flow_tvl1(frame_from, frame_to)


def plot_flow_panels(med: np.ndarray, cat: np.ndarray, spike_frame_idx: int, title: str) -> Figure:
    """Quiver-overlay pre-masked TV-L1 flow: spike-1->spike, then spike->spike+1..spike+3->spike+4.

    Arrows are only drawn at grid points within QUIVER_STEP of a CAT-bright
    pixel in the "from" frame (a max-filter dilation of the mask, not just an
    exact-pixel hit -- otherwise a coarse quiver grid can straddle every bright
    pixel and plot nothing at all).
    """
    pairs = [(spike_frame_idx - 1, spike_frame_idx)] + [
        (spike_frame_idx + i, spike_frame_idx + i + 1) for i in range(4)
    ]
    n_panels = len(pairs)
    fig = Figure(figsize=(6 * n_panels, 6), layout="constrained")
    height, width = med.shape[1], med.shape[2]
    grid_y, grid_x = np.mgrid[0:height:QUIVER_STEP, 0:width:QUIVER_STEP]

    for i, (idx_from, idx_to) in enumerate(pairs):
        v, u = premasked_flow(med, cat, idx_from, idx_to)

        ax = fig.add_subplot(1, n_panels, i + 1)
        ax.imshow(med[idx_from], cmap="gray", origin="upper")

        bright_dilated = maximum_filter(cat[idx_from] == CATEGORY_BRIGHT, size=QUIVER_STEP)
        keep = bright_dilated[grid_y, grid_x]
        plot_x, plot_y = grid_x[keep], grid_y[keep]
        sample_u, sample_v = u[grid_y, grid_x][keep], v[grid_y, grid_x][keep]

        if plot_x.size:
            ax.quiver(
                plot_x, plot_y, sample_u, sample_v,
                color="red", angles="xy", scale_units="xy", scale=QUIVER_SCALE, width=0.003,
                headwidth=2.5, headlength=3, headaxislength=2.5,
            )
        offset_from = idx_from - spike_frame_idx
        offset_to = idx_to - spike_frame_idx
        from_label = "spike" if offset_from == 0 else f"spike{offset_from:+d}"
        to_label = "spike" if offset_to == 0 else f"spike{offset_to:+d}"
        ax.set_title(f"{from_label} -> {to_label}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title + " (pre-masked, masked to CAT-bright)", fontsize=13)
    return fig


def run() -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)

    for tag in RECORDINGS:
        print(f"\n=== {tag} ===")
        med = tifffile.imread(MED_DIR / f"{tag}_MED.tif")
        cat = tifffile.imread(CAT_DIR / f"{tag}_CAT.tif")
        spike_frame_idx = med.shape[0] // 2

        fig = plot_flow_panels(med, cat, spike_frame_idx, title=tag)
        out_path = FLOW_DIR / f"{tag}_FLOW_PREMASKED.png"
        fig.savefig(out_path, dpi=110)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    run()
