"""
run_biexp_detrend.py  --  Per-pixel bi-exponential trend removal
================================================================
Pipeline per file:
  1. Randomly sample N_TAU_SAMPLE pixels, run curve_fit on each
     -> take median tau1, tau2 as shared time constants
  2. Per pixel: fast linear solve for (A, B) with fixed tau1, tau2
     -> subtract bi-exp trend over the full trace
  3. Align pixel means, save as <stem>_BIEXP_CAL.tif
  4. Gaussian blur (sigma=SIGMA), save as <stem>_BIEXP_GAUSS.tif
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from scipy.optimize import curve_fit

from functions import cpu_gaussian_blur

SIGMA = 6.0

# ── Config ────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent / "raw_tiffs"
OUT_DIR = Path(__file__).parent / "proc_tiffs"
OUT_DIR.mkdir(exist_ok=True)

FILES = [
    # "20260323_002.tif",
    # "20260323_005.tif",
    # "20260323_008.tif",
    # "20260323_011.tif",
    # "2024_12_19-0018.tif",
    # "2025_02_27-0036.tif",
    # "2026_03_20-0028.tif",
    # "2026_03_20-0029.tif",
    # "2026_03_20-0040.tif",
    # "2026_03_20-0041.tif",
    "2025_06_11-0002.tif",
    "2025_06_11-0003.tif",
    # "2026_03_20-0040.tif",
    # "2026_03_20-0041.tif",
]

N_TAU_SAMPLE = 500  # pixels used to estimate tau1, tau2
N_WORKERS = min(os.cpu_count() or 4, 8)
RNG_SEED = 42


# ── Core functions ────────────────────────────────────────


def biexp(t: np.ndarray, A: float, tau1: float, B: float, tau2: float, C: float) -> np.ndarray:
    """Bi-exponential decay: A*exp(-t/tau1) + B*exp(-t/tau2) + C."""
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2) + C


def make_p0_bounds(img: np.ndarray) -> tuple[list[float], tuple[list[float], list[float]]]:
    """
    Derive initial guess and bounds from the mean trace.
    tau1 (slow) and tau2 (fast) are forced into separate ranges.
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
    y: np.ndarray, t: np.ndarray, p0: list[float], bounds: tuple[list[float], list[float]]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit bi-exp to one pixel. Returns (trend, popt) or (zeros, None)."""
    try:
        popt, _ = curve_fit(biexp, t, y, p0=p0, bounds=bounds, maxfev=2000)
        return biexp(t, *popt), np.asarray(popt)
    except RuntimeError:
        return np.zeros_like(y), None


def sample_tau(img: np.ndarray, n_sample: int = N_TAU_SAMPLE, seed: int = RNG_SEED) -> tuple[float, float]:
    """
    Estimate shared tau1, tau2 by fitting bi-exp to a random sample of pixels.
    Returns (median_tau1, median_tau2).
    """
    n_frames, H, W = img.shape
    t = np.arange(n_frames, dtype=np.float64)
    p0, bounds = make_p0_bounds(img)

    rng = np.random.default_rng(seed)
    pys = rng.integers(0, H, n_sample)
    pxs = rng.integers(0, W, n_sample)

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

    print(f"  Sampled {n_sample} pixels  ({n_failed} failed)")
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


def _process_chunk(rows: list[int], img: np.ndarray, E: np.ndarray, E_pinv: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Fast linear detrend for one chunk of rows (worker thread)."""
    n_frames = img.shape[0]
    n_cols = img.shape[2]
    out_chunk = np.empty((n_frames, len(rows), n_cols), dtype=np.float32)

    for i, row in enumerate(rows):
        Y = img[:, row, :].astype(np.float64)
        trend = E @ (E_pinv @ Y)                              # (n_frames, n_cols)
        edge_min = np.minimum(trend[0], trend[-1])            # per-pixel baseline
        out_chunk[:, i, :] = (Y - trend + edge_min).astype(np.float32)

    return rows, out_chunk


def detrend_stack(img: np.ndarray, tau1: float, tau2: float, n_workers: int = N_WORKERS) -> np.ndarray:
    """Per-pixel linear solve with fixed tau1, tau2 (row-parallel)."""
    n_frames, H, _ = img.shape
    t = np.arange(n_frames, dtype=np.float64)
    E = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2), np.ones(n_frames)])
    E_pinv = np.linalg.pinv(E)

    out = np.empty_like(img, dtype=np.float32)
    chunk_size = max(1, -(-H // n_workers))
    row_chunks = [list(range(i, min(i + chunk_size, H))) for i in range(0, H, chunk_size)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_process_chunk, rows, img, E, E_pinv) for rows in row_chunks]
        for future in futures:
            rows, out_chunk = future.result()
            for i, row in enumerate(rows):
                out[:, row, :] = out_chunk[:, i, :]

    return out


def align_to_min(stack: np.ndarray) -> np.ndarray:
    """Shift each pixel so its time-mean equals the global minimum mean."""
    means = stack.mean(axis=0)
    offset = means - means.min()
    return stack - offset[None, :, :]


# ── Main ─────────────────────────────────────────────────

for fname in FILES:
    fpath = RAW_DIR / fname
    if not fpath.exists():
        print(f"\n[SKIP] {fname} not found in {RAW_DIR}")
        continue

    t0 = time.time()
    stem = Path(fname).stem
    print(f"\n{'=' * 60}\n  {fname}")

    # Step 1 -- load stack
    img = tifffile.imread(fpath).astype(np.float32)
    n_frames, H, W = img.shape
    print(f"  Shape : ({n_frames}, {H}, {W})   loaded {time.time() - t0:.1f}s")

    # Step 2 -- estimate tau from sampled pixels
    print(f"  Sampling {N_TAU_SAMPLE} pixels for tau estimation...")
    tau1, tau2 = sample_tau(img)
    print(f"  -> using tau1={tau1:.1f}  tau2={tau2:.1f}  ({time.time() - t0:.1f}s)")

    # Step 3 -- fast per-pixel linear detrend
    print(f"  Detrending  (workers={N_WORKERS})...")
    detrended = detrend_stack(img, tau1, tau2)

    # Step 4 -- align + save Cal
    detrended = align_to_min(detrended)
    tifffile.imwrite(OUT_DIR / f"{stem}_BIEXP_CAL.tif", np.clip(detrended, 0, 65535).astype(np.uint16))
    print(f"  Saved  -> {stem}_BIEXP_CAL.tif   ({time.time() - t0:.1f}s total)")

    # Step 5 -- Gaussian blur + save Gauss
    gaussian = cpu_gaussian_blur(detrended, SIGMA)
    tifffile.imwrite(OUT_DIR / f"{stem}_BIEXP_GAUSS.tif", np.clip(gaussian, 0, 65535).astype(np.uint16))
    print(f"  Saved  -> {stem}_BIEXP_GAUSS.tif   ({time.time() - t0:.1f}s total)")

    del img, detrended, gaussian

print(f"\n{'=' * 60}\nAll done!")
