# ruff: noqa: INP001
"""
detrend_figures.py  --  Explanation figures for docs/plain.md, step 1 (pixel-wise detrend).

Steps
-----
1. Load a raw TIFF, estimate (tau1, tau2) with the pipeline's sample_tau().
2. Detrend in row bands with the pipeline's biexp_detrend() (per-pixel, so banding is exact).
3. Pick a 100 x 100 px ROI on tissue with the clearest flash.
4. Fig. 1: raw frame vs detrended frame at the ROI's peak (FIG1_RECORDING).
5. Fig. 2: ROI-mean trace before and after pixel-wise detrending, one row per recording.

Usage:
    .venv/Scripts/python.exe docs/plain_01/scripts/detrend_figures.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle
from numba import cuda

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from functions import biexp_detrend, sample_tau  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_DIR = ROOT / "raw_tiffs"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
RECORDINGS = ["2025_11_27-0005.tif", "2025_12_15-0013.tif", "2026_01_08-0012.tif"]
FIG1_RECORDING = "2026_01_08-0012.tif"
FPS = 20.0              # FRAME_RATE_HZ in classes/sp_zone_analyzer.py
ROI = 100               # ROI side (px)
BLOCK = 25              # ROI search grid step (px); ROI = 4 x 4 blocks
SKIP_FRAMES = 40        # ignore the first 2 s when looking for a flash peak
BAND_ROWS = 128         # rows per detrend band (memory)
TRACE_COLOR = "#222222"


# ── STEP 1-2: detrend ─────────────────────────────────────────────────────────
def detrend_stack(img: np.ndarray, use_cuda: bool) -> tuple[np.ndarray, float, float]:
    """Pipeline detrend (raw - bi-exp trend), run in row bands to limit memory."""
    tau1, tau2 = sample_tau(img)
    out = np.empty(img.shape, dtype=np.float32)
    for r0 in range(0, img.shape[1], BAND_ROWS):
        out[:, r0:r0 + BAND_ROWS] = biexp_detrend(img[:, r0:r0 + BAND_ROWS], tau1, tau2, use_cuda)
    return out, tau1, tau2


# ── STEP 3: ROI search ────────────────────────────────────────────────────────
def find_roi(raw_mean: np.ndarray, det: np.ndarray) -> tuple[int, int, int]:
    """Return (y0, x0, peak_frame) of the tissue ROI whose mean trace has the highest peak z."""
    n_frames = det.shape[0]
    nb = det.shape[1] // BLOCK
    k = ROI // BLOCK
    blocks = np.empty((n_frames, nb, nb), dtype=np.float64)
    for i in range(nb):
        band = det[:, i * BLOCK:(i + 1) * BLOCK, :nb * BLOCK]
        blocks[:, i] = band.reshape(n_frames, BLOCK, nb, BLOCK).sum(axis=(1, 3))
    raw_blocks = raw_mean[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean(axis=(1, 3))

    n_win = nb - k + 1
    tissue_level = np.median([raw_blocks[i:i + k, j:j + k].mean() for i in range(n_win) for j in range(n_win)])
    best = (-np.inf, 0, 0, 0)
    for i in range(n_win):
        for j in range(n_win):
            if raw_blocks[i:i + k, j:j + k].mean() < tissue_level:
                continue
            tr = blocks[SKIP_FRAMES:, i:i + k, j:j + k].sum(axis=(1, 2))
            med = np.median(tr)
            noise = 1.4826 * np.median(np.abs(tr - med))
            z = (tr.max() - med) / noise
            if z > best[0]:
                best = (z, i * BLOCK, j * BLOCK, SKIP_FRAMES + int(tr.argmax()))
    return best[1], best[2], best[3]


# ── STEP 4: Fig. 1 ────────────────────────────────────────────────────────────
def plot_basal_removal(name: str, raw_frame: np.ndarray, det_frame: np.ndarray, frame: int) -> Path:
    """Raw frame vs detrended frame at the same moment."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    lo, hi = np.percentile(raw_frame, [1, 99.5])
    im0 = axes[0].imshow(raw_frame, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("A. Raw frame")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Raw intensity (counts)")
    lo, hi = np.percentile(det_frame, [1, 99.5])
    im1 = axes[1].imshow(det_frame, cmap="gray", vmin=lo, vmax=hi)
    axes[1].set_title("B. Detrended frame (raw − trend)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Raw − trend (counts)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{name}  |  frame {frame} ({frame / FPS:.2f} s)")
    path = OUT_DIR / "fig1_basal_removal.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── STEP 5: Fig. 2 ────────────────────────────────────────────────────────────
def plot_roi_traces(rows: list[dict]) -> Path:
    """One row per recording: thumbnail with ROI, ROI-mean trace before and after detrending."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.6 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 2.6, 2.6]}, constrained_layout=True)
    for r, row in enumerate(rows):
        t = np.arange(row["raw_trace"].size) / FPS
        ax_img, ax_pre, ax_post = axes[r]

        lo, hi = np.percentile(row["raw_mean"], [1, 99.5])
        ax_img.imshow(row["raw_mean"], cmap="gray", vmin=lo, vmax=hi)
        ax_img.add_patch(Rectangle((row["x0"], row["y0"]), ROI, ROI, fill=False, edgecolor="#f4a300", lw=2))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(row["name"], fontsize=10)

        ax_pre.plot(t, row["raw_trace"], color=TRACE_COLOR, lw=1)
        ax_pre.set_ylabel("Raw intensity (counts)")

        ax_post.plot(t, row["det_trace"], color=TRACE_COLOR, lw=1)
        ax_post.axhline(0, color="#999999", lw=0.8, ls="--")
        ax_post.set_ylabel("Raw − trend (counts)")

        if r == 0:
            ax_pre.set_title("Before detrending")
            ax_post.set_title("After detrending")
        for ax in (ax_pre, ax_post):
            ax.set_xlim(t[0], t[-1])
            ax.spines[["top", "right"]].set_visible(False)
            if r == len(rows) - 1:
                ax.set_xlabel("Time (s)")
    path = OUT_DIR / "fig2_roi_traces.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """Run steps 1-5 for all recordings."""
    use_cuda = cuda.is_available()
    print(f"CUDA: {use_cuda}")
    rows = []
    for idx, name in enumerate(RECORDINGS):
        print(f"[{idx + 1}/{len(RECORDINGS)}] {name}: loading...")
        img = tifffile.imread(RAW_DIR / name)
        raw_mean = img.mean(axis=0, dtype=np.float64)
        det, tau1, tau2 = detrend_stack(img, use_cuda)
        print(f"  tau1 = {tau1:.1f} frames ({tau1 / FPS:.1f} s), tau2 = {tau2:.1f} frames ({tau2 / FPS:.1f} s)")

        y0, x0, peak = find_roi(raw_mean, det)
        print(f"  ROI y={y0}:{y0 + ROI}, x={x0}:{x0 + ROI}, peak frame {peak} ({peak / FPS:.2f} s)")
        rows.append({
            "name": name, "raw_mean": raw_mean, "y0": y0, "x0": x0,
            "raw_trace": img[:, y0:y0 + ROI, x0:x0 + ROI].mean(axis=(1, 2), dtype=np.float64),
            "det_trace": det[:, y0:y0 + ROI, x0:x0 + ROI].mean(axis=(1, 2), dtype=np.float64),
        })
        if name == FIG1_RECORDING:
            path = plot_basal_removal(name, img[peak].astype(np.float32), det[peak], peak)
            print(f"  saved {path}")
        del img, det

    path = plot_roi_traces(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
