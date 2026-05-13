"""
run_biexp_detrend.py  --  Per-pixel bi-exponential trend removal
================================================================
Pipeline per file:
  1. Randomly sample N_PIXEL_SAMPLE pixels, run curve_fit on each
     -> take median tau1, tau2 as shared time constants
  2. Per pixel: fast linear solve for (A, B, C) with fixed tau1, tau2
     -> subtract bi-exp trend over the full trace  (Numba JIT / CUDA)
  3. Align pixel means, save as <stem>_BIEXP_CAL.tif
  4. Gaussian blur (sigma=SIGMA), save as <stem>_BIEXP_GAUSS.tif

Note: Kept as a standalone archive/reference script.
      Production pipeline functions live in functions/detrend.py,
      functions/tau_estimate.py, and functions/gaussian_blur.py.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import tifffile
from numba import cuda, jit, prange
from scipy.optimize import curve_fit

from functions import gaussian_blur_run

SIGMA = 6.0
N_PIXEL_SAMPLE = 500   # pixels used to estimate tau1, tau2
RNG_SEED = 42

# ── Config ────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent / "raw_tiffs"
OUT_DIR = Path(__file__).parent / "proc_tiffs"
OUT_DIR.mkdir(exist_ok=True)

FILES = [
    # "20260323_002.tif",
    "2025_06_11-0002.tif",
    "2025_06_11-0003.tif",
]


# ── Tau estimation ────────────────────────────────────────


def biexp(t: np.ndarray, A: float, tau1: float, B: float, tau2: float, C: float) -> np.ndarray:
    """Bi-exponential decay: A*exp(-t/tau1) + B*exp(-t/tau2) + C."""
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2) + C


def make_p0_bounds(img: np.ndarray) -> tuple[list[float], tuple[list[float], list[float]]]:
    """
    Derive initial parameter guess (p0) and search bounds from the image mean trace.

    p0 = [A, tau1, B, tau2, C] — starting values for scipy curve_fit.
    tau1 (slow) and tau2 (fast) are forced into separate ranges to prevent
    the two exponentials from collapsing onto the same time constant.
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
    """Fit bi-exp to one pixel. Returns (trend, popt) or (zeros, None) on failure."""
    try:
        popt, _ = curve_fit(biexp, t, y, p0=p0, bounds=bounds, maxfev=2000)
    except RuntimeError:
        return np.zeros_like(y), None
    else:
        return biexp(t, *popt), np.asarray(popt)


def sample_tau(img: np.ndarray, n_pixels: int = N_PIXEL_SAMPLE, seed: int = RNG_SEED) -> tuple[float, float]:
    """
    Estimate shared tau1, tau2 by fitting bi-exp to a random sample of pixels.
    Returns (median_tau1, median_tau2).
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


# ── Bi-exponential detrend (Numba JIT + CUDA) ────────────


@jit(nopython=True, parallel=True)
def _cpu_biexp(img_flat: np.ndarray, basis_pinv: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """
    Numba JIT per-pixel bi-exp detrend on CPU (parallel over pixels).

    Maths:
      basis_matrix  (T, 3): columns are [exp(-t/tau1), exp(-t/tau2), ones]
      basis_pinv    (3, T): Moore-Penrose pseudo-inverse of basis_matrix
      For each pixel trace y (T,):
        coeffs = basis_pinv @ y         # least-squares fit → (3,)
        trend  = basis_matrix @ coeffs  # reconstructed bi-exp baseline → (T,)
        detrended = y - trend + min(trend[0], trend[-1])  # edge normalization
    """
    n_pixels, T = img_flat.shape
    output = np.zeros_like(img_flat)
    for i in prange(n_pixels):
        y = img_flat[i]
        coeffs = np.dot(basis_pinv, y)       # least-squares coefficients (3,)
        trend = np.dot(basis_matrix, coeffs)  # reconstructed baseline (T,)
        edge_min = min(trend[0], trend[T - 1])
        output[i] = y - trend + edge_min
    return output


@cuda.jit
def _gpu_biexp(
    img_flat: np.ndarray,
    basis_pinv: np.ndarray,
    basis_matrix: np.ndarray,
    output: np.ndarray,
) -> None:
    """
    CUDA kernel: one thread per pixel for bi-exp detrend.

    basis_pinv   (3, T): projects pixel trace onto 3-component basis via dot product.
    basis_matrix (T, 3): reconstructs trend from the 3 coefficients.
    Trend computed on-the-fly per frame to avoid large local arrays.
    Edge normalization: shift so the lower endpoint value = 0.
    """
    i = cuda.grid(1)
    if i >= img_flat.shape[0]:
        return

    T = img_flat.shape[1]

    # coeffs = basis_pinv @ y  (manual dot, result size = 3)
    coeffs = cuda.local.array(3, dtype=np.float64)
    for k in range(3):
        s = 0.0
        for t in range(T):
            s += basis_pinv[k, t] * img_flat[i, t]
        coeffs[k] = s

    # Compute trend only at endpoints for edge normalization (avoids storing full trend)
    trend_0 = 0.0
    trend_end = 0.0
    for k in range(3):
        trend_0 += basis_matrix[0, k] * coeffs[k]
        trend_end += basis_matrix[T - 1, k] * coeffs[k]
    edge_min = min(trend_0, trend_end)

    # Subtract trend + apply edge normalization, computing trend per frame on-the-fly
    for t in range(T):
        trend_t = 0.0
        for k in range(3):
            trend_t += basis_matrix[t, k] * coeffs[k]
        output[i, t] = img_flat[i, t] - trend_t + edge_min


def biexp_detrend(img: np.ndarray, tau1: float, tau2: float, cuda_available: bool) -> np.ndarray:
    """
    Bi-exp detrend dispatcher: GPU if available, else CPU Numba JIT.

    Builds basis_matrix / basis_pinv from tau1, tau2, flattens img to
    (n_pixels, T), runs detrend per pixel, reshapes back to (T, H, W).
    """
    n_frames, H, W = img.shape
    t = np.arange(n_frames, dtype=np.float64)

    # basis_matrix (T, 3): columns = [exp(-t/tau1), exp(-t/tau2), constant ones]
    basis_matrix = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2), np.ones(n_frames)])
    # basis_pinv (3, T): pseudo-inverse used as the least-squares projection operator
    basis_pinv = np.linalg.pinv(basis_matrix)

    img_flat = img.reshape(n_frames, -1).T.astype(np.float64)  # (n_pixels, T)

    if cuda_available:
        d_img = cuda.to_device(img_flat)
        d_pinv = cuda.to_device(basis_pinv)
        d_basis = cuda.to_device(basis_matrix)
        d_out = cuda.to_device(np.empty_like(img_flat))
        threads = 256
        blocks = math.ceil(img_flat.shape[0] / threads)
        _gpu_biexp[blocks, threads](d_img, d_pinv, d_basis, d_out)
        cuda.synchronize()
        output_flat = d_out.copy_to_host()
    else:
        output_flat = _cpu_biexp(img_flat, basis_pinv, basis_matrix)

    return output_flat.T.reshape(n_frames, H, W).astype(np.float32)


def align_to_min(stack: np.ndarray) -> np.ndarray:
    """Shift each pixel so its time-mean equals the global minimum mean."""
    means = stack.mean(axis=0)
    offset = means - means.min()
    return stack - offset[None, :, :]


# ── Main ─────────────────────────────────────────────────

try:
    _cuda_ok = cuda.is_available()
except Exception:  # noqa: BLE001
    _cuda_ok = False

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
    print(f"  Sampling {N_PIXEL_SAMPLE} pixels for tau estimation...")
    tau1, tau2 = sample_tau(img)
    print(f"  -> using tau1={tau1:.1f}  tau2={tau2:.1f}  ({time.time() - t0:.1f}s)")

    # Step 3 -- fast per-pixel bi-exp detrend (Numba JIT or CUDA)
    print(f"  Detrending  (cuda={_cuda_ok})...")
    detrended = biexp_detrend(img, tau1, tau2, _cuda_ok)

    # Step 4 -- align pixel means + save CAL
    detrended = align_to_min(detrended)
    tifffile.imwrite(OUT_DIR / f"{stem}_BIEXP_CAL.tif", np.clip(detrended, 0, 65535).astype(np.uint16))
    print(f"  Saved  -> {stem}_BIEXP_CAL.tif   ({time.time() - t0:.1f}s total)")

    # Step 5 -- Gaussian blur + save GAUSS
    gaussian = gaussian_blur_run(detrended, SIGMA, _cuda_ok)
    tifffile.imwrite(OUT_DIR / f"{stem}_BIEXP_GAUSS.tif", np.clip(gaussian, 0, 65535).astype(np.uint16))
    print(f"  Saved  -> {stem}_BIEXP_GAUSS.tif   ({time.time() - t0:.1f}s total)")

    del img, detrended, gaussian

print(f"\n{'=' * 60}\nAll done!")
