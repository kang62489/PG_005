"""
tau_estimate.py  --  Estimate bi-exponential time constants (tau1, tau2) from image data.

Runs once per file: fits a bi-exponential model to a random sample of pixels using
scipy curve_fit, then returns the median tau1 and tau2 for use in the full detrend pass.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

_N_PIXEL_SAMPLE = 500
_RNG_SEED = 42


def biexp(t: np.ndarray, A: float, tau1: float, B: float, tau2: float, C: float) -> np.ndarray:
    """Bi-exponential decay model: A*exp(-t/tau1) + B*exp(-t/tau2) + C."""
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2) + C


def make_p0_bounds(img: np.ndarray) -> tuple[list[float], tuple[list[float], list[float]]]:
    """
    Derive initial parameter guess and bounds from the image's mean trace.

    tau1 (slow decay) and tau2 (fast decay) are forced into separate ranges
    to prevent the two exponentials from collapsing onto the same time constant.
    """
    n_frames = img.shape[0]
    mean_tr = img.mean(axis=(1, 2)).astype(np.float64)
    start = float(mean_tr[:10].mean())
    end = float(mean_tr[-10:].mean())
    amp = max(start - end, 1.0)
    p0 = [amp * 0.7, n_frames * 0.4, amp * 0.3, n_frames * 0.08, end]
    lo = [0, n_frames * 0.15, 0, 5, -np.inf]
    hi = [np.inf, n_frames * 2, np.inf, n_frames * 0.25, np.inf]
    return p0, (lo, hi)


def _fit_pixel(
    y: np.ndarray,
    t: np.ndarray,
    p0: list[float],
    bounds: tuple[list[float], list[float]],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit bi-exp to a single pixel trace. Returns (trend, popt) or (zeros, None) on failure."""
    try:
        popt, _ = curve_fit(biexp, t, y, p0=p0, bounds=bounds, maxfev=2000)
    except RuntimeError:
        return np.zeros_like(y), None
    else:
        return biexp(t, *popt), np.asarray(popt)


def sample_tau(img: np.ndarray, n_pixels: int = _N_PIXEL_SAMPLE, seed: int = _RNG_SEED) -> tuple[float, float]:
    """
    Estimate shared tau1, tau2 by fitting bi-exp to a random sample of pixels.

    Returns (median_tau1, median_tau2). Falls back to frame-count-based defaults
    if all fits fail.
    """
    n_frames, H, W = img.shape
    t = np.arange(n_frames, dtype=np.float64)
    p0, bounds = make_p0_bounds(img)

    rng = np.random.default_rng(seed)
    pys = rng.integers(0, H, n_pixels)
    pxs = rng.integers(0, W, n_pixels)

    tau1s: list[float] = []
    tau2s: list[float] = []
    n_failed = 0

    for py, px in zip(pys, pxs, strict=False):
        y = img[:, py, px].astype(np.float64)
        _, popt = _fit_pixel(y, t, p0, bounds)
        if popt is not None:
            tau1s.append(float(popt[1]))
            tau2s.append(float(popt[3]))
        else:
            n_failed += 1

    print(f"  Sampled {n_pixels} pixels  ({n_failed} failed)")
    if tau1s:
        print(
            f"  tau1: median={np.median(tau1s):.1f}  "
            f"IQR [{np.percentile(tau1s, 25):.1f}, {np.percentile(tau1s, 75):.1f}]"
        )
        print(
            f"  tau2: median={np.median(tau2s):.1f}  "
            f"IQR [{np.percentile(tau2s, 25):.1f}, {np.percentile(tau2s, 75):.1f}]"
        )

    tau1 = float(np.median(tau1s)) if tau1s else n_frames * 0.4
    tau2 = float(np.median(tau2s)) if tau2s else n_frames * 0.08
    return tau1, tau2
