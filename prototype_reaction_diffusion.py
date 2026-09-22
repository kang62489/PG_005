"""Reaction-diffusion (diffusion + linear decay) fit prototype, spike..spike+4.

Model: dC/dt = D * laplacian(C) - k * C

Fits D (effective spread rate, px^2/frame) and k (decay rate, 1/frame) via
linear least squares on real MED-stack pixel data -- no correspondence/motion
assumption, no brightness-constancy assumption. Scratch script only, no
pipeline edits.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from matplotlib.figure import Figure
from scipy.ndimage import laplace

PROJECT_ROOT = Path("D:/Programs/PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

CATEGORY_BRIGHT = 1
N_FRAMES = 5  # spike, spike+1, ..., spike+4

MED_DIR = PROJECT_ROOT / "output" / "test4" / "median"
CAT_DIR = PROJECT_ROOT / "output" / "test4" / "categorized"
OUT_DIR = PROJECT_ROOT / "output" / "test4" / "reaction_diffusion"

RECORDINGS = [
    "2025_06_11-0002_A1S4RC1_BIEXP_ALS",
    "2025_06_11-0003_A1S4RC1_BIEXP_ALS",
    "2025_11_13-0017_A1S1RC2_BIEXP_ALS",
    "2025_11_13-0018_A1S1RC2_BIEXP_ALS",
    "2025_12_15-0012_A1S3RC1_BIEXP_ALS",
    "2025_12_15-0013_A1S3RC1_BIEXP_ALS",
]


def fit_diffusion_decay(
    frames: np.ndarray, fit_mask: np.ndarray
) -> tuple[float, float, float]:
    """Fit D, k in dC/dt = D*laplacian(C) - k*C via linear least squares.

    frames: (n_frames, H, W) consecutive frames, dt = 1 frame.
    fit_mask: (H, W) bool, pixels included in the regression.
    Returns (D, k, r_squared).
    """
    dcdt_list = []
    lap_list = []
    c_list = []
    for t in range(frames.shape[0] - 1):
        c_t = frames[t]
        dcdt = frames[t + 1] - c_t
        lap = laplace(c_t, mode="reflect")
        dcdt_list.append(dcdt[fit_mask])
        lap_list.append(lap[fit_mask])
        c_list.append(c_t[fit_mask])

    y = np.concatenate(dcdt_list)
    x_lap = np.concatenate(lap_list)
    x_c = np.concatenate(c_list)

    design = np.stack([x_lap, -x_c], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    d_coeff, k_coeff = coeffs

    y_pred = design @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(d_coeff), float(k_coeff), float(r_squared)


def plot_fit_diagnostic(
    frames: np.ndarray, fit_mask: np.ndarray, d_coeff: float, k_coeff: float, title: str
) -> Figure:
    """Predicted-vs-actual next frame (last consecutive pair) + dC/dt scatter."""
    c_t = frames[-2]
    c_next_actual = frames[-1]
    lap = laplace(c_t, mode="reflect")
    c_next_pred = c_t + d_coeff * lap - k_coeff * c_t

    vmin = min(c_t.min(), c_next_actual.min(), c_next_pred.min())
    vmax = max(c_t.max(), c_next_actual.max(), c_next_pred.max())

    fig = Figure(figsize=(16, 4.5), layout="constrained")
    ax1, ax2, ax3, ax4 = fig.subplots(1, 4)

    ax1.imshow(c_next_actual, cmap="inferno", vmin=vmin, vmax=vmax)
    ax1.set_title("Actual next frame")
    ax1.axis("off")

    ax2.imshow(c_next_pred, cmap="inferno", vmin=vmin, vmax=vmax)
    ax2.set_title("Model-predicted next frame")
    ax2.axis("off")

    resid = c_next_actual - c_next_pred
    resid_lim = np.abs(resid).max()
    im3 = ax3.imshow(resid, cmap="coolwarm", vmin=-resid_lim, vmax=resid_lim)
    ax3.set_title("Residual (actual - predicted)")
    ax3.axis("off")
    fig.colorbar(im3, ax=ax3, shrink=0.7)

    dcdt_actual = (c_next_actual - c_t)[fit_mask]
    dcdt_pred = (d_coeff * lap - k_coeff * c_t)[fit_mask]
    ax4.scatter(dcdt_actual, dcdt_pred, s=2, alpha=0.3)
    lims = [min(dcdt_actual.min(), dcdt_pred.min()), max(dcdt_actual.max(), dcdt_pred.max())]
    ax4.plot(lims, lims, "k--", lw=1)
    ax4.set_xlabel("actual dC/dt")
    ax4.set_ylabel("model dC/dt")
    ax4.set_title("Fit-region dC/dt")

    fig.suptitle(f"{title}\nD={d_coeff:.4f} px^2/frame, k={k_coeff:.4f} /frame", fontsize=12)
    return fig


def run_one(tag: str) -> None:
    print(f"\n=== {tag} ===")
    median_segment = tifffile.imread(MED_DIR / f"{tag}_MED.tif")
    cat_stack = tifffile.imread(CAT_DIR / f"{tag}_CAT.tif")
    spike_frame_idx = median_segment.shape[0] // 2

    frame_idxs = list(range(spike_frame_idx, spike_frame_idx + N_FRAMES))
    frames = median_segment[frame_idxs]
    bright_stack = cat_stack[frame_idxs] == CATEGORY_BRIGHT
    fit_mask = np.any(bright_stack, axis=0)
    print(f"Fit region: {fit_mask.sum()} px (union of bright mask across {N_FRAMES} frames)")

    d_coeff, k_coeff, r_squared = fit_diffusion_decay(frames, fit_mask)
    print(f"D = {d_coeff:.4f} px^2/frame, k = {k_coeff:.4f} /frame, R^2 = {r_squared:.4f}")

    fig = plot_fit_diagnostic(frames, fit_mask, d_coeff, k_coeff, title=tag)
    out_path = OUT_DIR / f"{tag}_RDFIT.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for tag in RECORDINGS:
        run_one(tag)
