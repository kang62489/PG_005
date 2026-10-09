# ruff: noqa: INP001
"""
zscore_blur_figures.py  --  Explanation figures for docs/plain.md, steps 2 (z-score) and 3 (blur).

Steps
-----
1. Detrend each recording with the pipeline (sample_tau + biexp_detrend, row bands).
2. Global histogram + left-side Gaussian fit, same binning / seed / fit as fit_hist_sigma();
   checked against fit_hist_sigma() itself.
3. Fig. 3: histogram + fitted Gaussian, one panel per recording.
4. Fig. 4: z-scored frame blurred with different widths (pipeline's gaussian_blur_run).
5. Fig. 5: z-scored frame before / after the pipeline blur (SIGMA = 4).

Usage:
    .venv/Scripts/python.exe docs/plain_01/scripts/zscore_blur_figures.py
"""
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from numba import cuda
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from functions import fit_hist_sigma, gaussian_blur_run, img_zscore_convert  # noqa: E402
from functions.fit_hist import N_HIST_BINS, _cpu_masked_std, _gaussian, histogram_counts  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detrend_figures import detrend_stack  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_DIR = ROOT / "raw_tiffs"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
RECORDINGS = ["2025_11_27-0005.tif", "2025_12_15-0013.tif", "2026_01_08-0012.tif"]
FRAME_RECORDING = "2026_01_08-0012.tif"
FRAME = 948                         # same frame as Fig. 1
FPS = 20.0
PIPELINE_WIDTH = 4.0                # SIGMA in img_proc.py
BLUR_WIDTHS = [1, 2, 4, 6, 8, 16]   # px
HIGHLIGHT = "#f4a300"


def kernel_px(width: float) -> int:
    """Kernel side used by gaussian_blur.py: ceil(6 * width), made odd."""
    size = math.ceil(width * 6)
    return size + 1 if size % 2 == 0 else size


# ── STEP 2: histogram + left-side fit ─────────────────────────────────────────
def histogram_fit(det: np.ndarray) -> dict:
    """Pipeline histogram + left-side Gaussian fit, with the fitted amplitude kept for plotting."""
    values = det.ravel()
    counts, centers = histogram_counts(values, N_HIST_BINS, float(values.min()), float(values.max()), False)
    x_peak = float(centers[int(np.argmax(counts))])
    seed = float(_cpu_masked_std(values, x_peak))
    left = centers <= x_peak
    popt, _ = curve_fit(_gaussian, centers[left], counts[left],
                        p0=[float(counts.max()), x_peak, seed], maxfev=5000)
    mu, sd = fit_hist_sigma(det, cuda_available=False)
    assert np.isclose(mu, popt[1]), "mean differs from fit_hist_sigma()"
    assert np.isclose(sd, abs(popt[2])), "sigma differs from fit_hist_sigma()"
    return {"counts": counts, "centers": centers, "x_peak": x_peak,
            "amp": float(popt[0]), "mu": mu, "sd": sd}


# ── STEP 3: Fig. 3 ────────────────────────────────────────────────────────────
def plot_histograms(fits: list[tuple[str, dict]]) -> Path:
    """Global histogram (log y) + fitted Gaussian; fitted range = peak and left side."""
    fig, axes = plt.subplots(1, len(fits), figsize=(5.2 * len(fits), 4.4), constrained_layout=True)
    for ax, (name, f) in zip(axes, fits, strict=True):
        c, x = f["counts"], f["centers"]
        ax.bar(x, c, width=x[1] - x[0], color="#bbbbbb", edgecolor="none", label="All pixels, all frames")
        ax.axvspan(x[0], f["x_peak"], color="#4c78a8", alpha=0.08, label="Fitted range (peak + left)")
        xs = np.linspace(x[0], x[-1], 4000)
        ax.plot(xs, _gaussian(xs, f["amp"], f["mu"], f["sd"]), color="#4c78a8", lw=2, label="Fitted Gaussian")
        ax.axvline(f["mu"], color="#222222", lw=1, ls="--")
        ax.text(0.97, 0.95, f"μ = {f['mu']:.1f} counts\nσ = {f['sd']:.1f} counts",
                transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(1, c.max() * 5)
        ax.set_xlim(f["mu"] - 8 * f["sd"], f["mu"] + 16 * f["sd"])
        ax.set_xlabel("Detrended value (counts)")
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Number of values")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    path = OUT_DIR / "fig3_histogram_fit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── STEP 4: Fig. 4 ────────────────────────────────────────────────────────────
def plot_width_comparison(z_frame: np.ndarray, blurred: dict[float, np.ndarray]) -> Path:
    """Unblurred + each blur width, same gray range (from the pipeline width panel)."""
    lo, hi = np.percentile(blurred[PIPELINE_WIDTH], [1, 99.5])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), constrained_layout=True)
    panels = [("No blur", z_frame, False)]
    panels += [(f"Blur width {w} px\n(kernel {kernel_px(w)} × {kernel_px(w)} px)", blurred[w], w == PIPELINE_WIDTH)
               for w in BLUR_WIDTHS]
    for ax, (title, img, is_pipeline) in zip(axes.flat, panels, strict=False):
        im = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_xticks([])
        ax.set_yticks([])
        if is_pipeline:
            title += "\n★ used in pipeline"
            for side in ax.spines.values():
                side.set_edgecolor(HIGHLIGHT)
                side.set_linewidth(4)
        ax.set_title(title, fontsize=10, color=HIGHLIGHT if is_pipeline else "#222222")
    axes.flat[-1].axis("off")
    fig.colorbar(im, ax=axes.flat[-1], fraction=0.5, label="z")
    fig.suptitle(f"{FRAME_RECORDING}  |  frame {FRAME} ({FRAME / FPS:.2f} s)  |  same gray range in every panel")
    path = OUT_DIR / "fig4_blur_width_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── STEP 5: Fig. 5 ────────────────────────────────────────────────────────────
def plot_blur_before_after(z_frame: np.ndarray, blurred: np.ndarray) -> Path:
    """Z-scored frame before vs after the pipeline blur, same gray range (as in Fig. 4)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    w = PIPELINE_WIDTH
    lo, hi = np.percentile(blurred, [1, 99.5])
    for ax, img, title in [(axes[0], z_frame, "A. Z-scored frame"),
                           (axes[1], blurred, f"B. After blur (width {w:g} px, kernel {kernel_px(w)} × {kernel_px(w)} px)")]:
        im = ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax, shrink=0.8, label="z")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{FRAME_RECORDING}  |  frame {FRAME} ({FRAME / FPS:.2f} s)  |  same gray range in both panels")
    path = OUT_DIR / "fig5_blur_before_after.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """Run steps 1-5."""
    use_cuda = cuda.is_available()
    fits = []
    for idx, name in enumerate(RECORDINGS):
        print(f"[{idx + 1}/{len(RECORDINGS)}] {name}: loading + detrending...")
        img = tifffile.imread(RAW_DIR / name)
        det, _, _ = detrend_stack(img, use_cuda)
        del img
        f = histogram_fit(det)
        print(f"  mu = {f['mu']:.2f} counts, sigma = {f['sd']:.2f} counts")
        fits.append((name, f))

        if name == FRAME_RECORDING:
            z_frame = img_zscore_convert(det[FRAME:FRAME + 1], f["mu"], f["sd"])
            blurred = {w: gaussian_blur_run(z_frame, float(w), use_cuda)[0] for w in BLUR_WIDTHS}
            print(f"  saved {plot_width_comparison(z_frame[0], blurred)}")
            print(f"  saved {plot_blur_before_after(z_frame[0], blurred[PIPELINE_WIDTH])}")
        del det
    print(f"saved {plot_histograms(fits)}")


if __name__ == "__main__":
    main()
