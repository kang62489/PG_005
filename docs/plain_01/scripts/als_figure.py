# ruff: noqa: INP001
"""
als_figure.py  --  Explanation figure for docs/plain.md, section 5 (ALS baseline correction).

Steps
-----
1. Pick 5 random 128 x 128 px ROIs (as the ALS test in controllers/ctrl_als_correct.py; seeded here).
2. Before  : ROI-mean trace of the GAUSS file.
3. Fitted  : ALS baseline of that trace (als_run, same parameters as the stored ALS files).
4. After   : ROI-mean trace of the stored ALS file (pipeline output, per-pixel ALS).
5. Fig. 7  : GAUSS frame with the ROIs + one row of traces per ROI.

Usage:
    .venv/Scripts/python.exe docs/plain_01/scripts/als_figure.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from functions import als_run  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
RECORDING = "2026_01_08-0012"
FRAME = 948                     # overview frame, same as Figs. 1-5
GAUSS_PATH = ROOT / "proc_tiffs" / f"{RECORDING}_BIEXP_GAUSS.tif"
ALS_PATH = ROOT / "proc_tiffs" / f"{RECORDING}_BIEXP_ALS.tif"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
LAM, P, N_ITER = 11, 0.05, 10   # reproduce the stored ALS file (als_correct.py CLI defaults)
ROI_SIZE = 128                  # ctrl_als_correct.ROI_SIZE
N_ROIS = 5                      # ctrl_als_correct.N_ROIS
SEED = 0
FPS = 20.0
TRACE_COLOR = "#222222"
FIT_COLOR = "#f4a300"


# ── STEP 1-4: ROIs and traces ─────────────────────────────────────────────────
def roi_traces() -> list[dict]:
    """Before / fitted / after traces for N_ROIS random ROIs."""
    gauss = tifffile.memmap(GAUSS_PATH)
    als = tifffile.memmap(ALS_PATH)
    n_frames, height, width = gauss.shape
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(N_ROIS):
        y0 = int(rng.integers(0, height - ROI_SIZE + 1))
        x0 = int(rng.integers(0, width - ROI_SIZE + 1))
        sl = (slice(None), slice(y0, y0 + ROI_SIZE), slice(x0, x0 + ROI_SIZE))
        before = gauss[sl].astype(np.float32).mean(axis=(1, 2))
        fitted = als_run(before.reshape(n_frames, 1, 1), LAM, P, N_ITER, False)[:, 0, 0]
        after = als[sl].astype(np.float32).mean(axis=(1, 2))
        rows.append({"y0": y0, "x0": x0, "before": before, "fitted": fitted, "after": after})
    return rows


# ── STEP 5: Fig. 7 ────────────────────────────────────────────────────────────
def plot_als(gauss_frame: np.ndarray, rows: list[dict]) -> Path:
    """GAUSS frame with numbered ROIs (left) + before/fitted and after traces per ROI."""
    fig = plt.figure(figsize=(16, 2.4 * N_ROIS), layout="constrained")
    sub_img, sub_tr = fig.subfigures(1, 2, width_ratios=[1.6, 5.2])
    ax_img = sub_img.add_axes((0.02, 0.48, 0.96, 0.44))           # top of the left column
    cax = sub_img.add_axes((0.2, 0.43, 0.6, 0.012))
    lo, hi = np.percentile(gauss_frame, [1, 99.5])
    im = ax_img.imshow(gauss_frame, cmap="gray", vmin=lo, vmax=hi)
    sub_img.colorbar(im, cax=cax, orientation="horizontal", label="z")
    for k, r in enumerate(rows, 1):
        ax_img.add_patch(Rectangle((r["x0"], r["y0"]), ROI_SIZE, ROI_SIZE, fill=False, edgecolor=FIT_COLOR, lw=1.5))
        ax_img.text(r["x0"] + 6, r["y0"] + 22, str(k), color=FIT_COLOR, fontsize=11, weight="bold")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"{RECORDING}  GAUSS, frame {FRAME} ({FRAME / FPS:.2f} s)\n"
                     f"{N_ROIS} random ROIs, {ROI_SIZE} × {ROI_SIZE} px", fontsize=10)

    t = np.arange(rows[0]["before"].size) / FPS
    gs = sub_tr.add_gridspec(N_ROIS, 2)
    first_pre = first_post = None   # same y range down each column
    for k, r in enumerate(rows):
        ax_pre = sub_tr.add_subplot(gs[k, 0], sharey=first_pre)
        ax_post = sub_tr.add_subplot(gs[k, 1], sharey=first_post)
        first_pre, first_post = first_pre or ax_pre, first_post or ax_post
        ax_pre.plot(t, r["before"], color=TRACE_COLOR, lw=0.8, label="GAUSS (before)")
        ax_pre.plot(t, r["fitted"], color=FIT_COLOR, lw=1.6, label="ALS baseline (fitted)")
        ax_post.plot(t, r["after"], color=TRACE_COLOR, lw=0.8)
        ax_post.axhline(0, color="#999999", lw=0.8, ls="--")
        ax_pre.set_ylabel(f"ROI {k + 1}\nz")
        for ax in (ax_pre, ax_post):
            ax.set_xlim(t[0], t[-1])
            ax.spines[["top", "right"]].set_visible(False)
            if k < N_ROIS - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Time (s)")
        if k == 0:
            ax_pre.set_title("Before + fitted ALS baseline")
            ax_post.set_title("After ALS correction (ALS file)")
            ax_pre.legend(frameon=False, loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(f"ALS baseline correction  |  λ = {LAM}, p = {P}, n_iter = {N_ITER}")
    path = OUT_DIR / "fig7_als_correction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """Run steps 1-5."""
    rows = roi_traces()
    for k, r in enumerate(rows, 1):
        print(f"ROI {k}: y={r['y0']}:{r['y0'] + ROI_SIZE}, x={r['x0']}:{r['x0'] + ROI_SIZE}")
    gauss_frame = tifffile.memmap(GAUSS_PATH)[FRAME].astype(np.float32)
    print(f"saved {plot_als(gauss_frame, rows)}")


if __name__ == "__main__":
    main()
