"""
run_als_baseline.py  --  Per-pixel ALS baseline estimation
===========================================================
Pipeline per file:
  Input : *_Biexp_Gauss.tif  (T x H x W, uint16)
  1. ALS pixel-wise  ->  slowly-varying F0 per pixel
  2. Save *_Biexp_Baseline.tif  (float32)
  3. Compute dF/F0 = (Gauss - Baseline) / Baseline
  4. Save *_Biexp_dFF0.tif  (float32)

GPU path  : one Numba CUDA thread per pixel, Thomas algorithm per ALS iteration.
CPU path  : same Thomas algorithm vectorised across all pixels with NumPy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tifffile
from numba import cuda

# ── Config ────────────────────────────────────────────────
PROC_DIR = Path(__file__).parent / "proc_tiffs"

FILES = [
    # "2026_03_20-0028_Biexp_Gauss.tif",
    # "2026_03_20-0029_Biexp_Gauss.tif",
    # "2026_03_20-0040_Biexp_Gauss.tif",
    # "2026_03_20-0041_Biexp_Gauss.tif",
    "2025_06_11-0002_Biexp_Gauss.tif",
    "2025_06_11-0003_Biexp_Gauss.tif",
]

LAM = 1e2    # smoothness: larger → smoother baseline
P = 0.05    # asymmetry: small → baseline hugs the lower envelope
N_ITER = 10  # ALS iterations (10 is usually sufficient)
MAX_T = 2048  # compile-time local-array size; must be >= number of frames


# ── GPU kernel ────────────────────────────────────────────


@cuda.jit
def _als_gpu_kernel(data, baseline, lam, p, n_iter) -> None:
    """ALS baseline per pixel using Thomas algorithm (1st-order differences).

    data / baseline shape: (n_pixels, T).
    Each thread handles one pixel independently.
    Local arrays are fixed at MAX_T = 2048 frames.

    System solved each iteration: (W + λ L'L) z = W y
      L'L  main diag : [1, 2, 2, ..., 2, 1]
      L'L  off  diag : -1  (constant)
    """
    px = cuda.grid(1)
    if px >= data.shape[0]:
        return

    T = data.shape[1]

    # Thread-local arrays (stored in GPU local/register memory)
    z = cuda.local.array(2048, dtype=np.float32)   # current baseline estimate
    w = cuda.local.array(2048, dtype=np.float32)   # ALS weights
    c = cuda.local.array(2048, dtype=np.float32)   # Thomas: modified upper diagonal
    d = cuda.local.array(2048, dtype=np.float32)   # Thomas: modified RHS

    for i in range(T):
        z[i] = data[px, i]
        w[i] = 1.0

    for _ in range(n_iter):
        # ── Forward sweep (Thomas algorithm) ─────────────────────────────
        # main[0]: L'L contributes 1 at endpoints, 2 elsewhere
        denom = w[0] + lam * 1.0
        c[0] = -lam / denom
        d[0] = w[0] * data[px, 0] / denom

        for i in range(1, T):
            lll_m = 1.0 if i == T - 1 else 2.0
            # denom = modified main = b[i] - a[i]*c'[i-1]
            #       = (w[i] + λ*lll_m) - (-λ)*c[i-1]
            #       = w[i] + λ*lll_m + λ*c[i-1]   (c[i-1] is negative)
            denom = w[i] + lam * lll_m + lam * c[i - 1]
            if i < T - 1:
                c[i] = -lam / denom
            # d'[i] = (d[i] - a[i]*d'[i-1]) / denom
            #       = (w[i]*y[i] - (-λ)*d[i-1]) / denom
            d[i] = (w[i] * data[px, i] + lam * d[i - 1]) / denom

        # ── Backward substitution ─────────────────────────────────────────
        z[T - 1] = d[T - 1]
        for i in range(T - 2, -1, -1):
            z[i] = d[i] - c[i] * z[i + 1]

        # ── Update weights ────────────────────────────────────────────────
        for i in range(T):
            w[i] = p if data[px, i] > z[i] else (1.0 - p)

    for i in range(T):
        baseline[px, i] = z[i]


# ── GPU driver ────────────────────────────────────────────


def gpu_als_baseline(gauss: np.ndarray, lam: float = LAM, p: float = P, n_iter: int = N_ITER) -> np.ndarray:
    """Run ALS pixel-wise on GPU. Input shape: (T, H, W) float32."""
    T, H, W = gauss.shape
    if T > MAX_T:
        msg = f"T={T} exceeds MAX_T={MAX_T}; increase MAX_T and recompile"
        raise ValueError(msg)

    n_px = H * W
    # Reshape to (n_pixels, T) — one row per pixel, matching kernel layout
    data_px = np.ascontiguousarray(gauss.reshape(T, n_px).T, dtype=np.float32)

    data_gpu = cuda.to_device(data_px)
    baseline_gpu = cuda.device_array_like(data_px)

    threads = 128
    blocks = (n_px + threads - 1) // threads
    _als_gpu_kernel[blocks, threads](data_gpu, baseline_gpu, float(lam), float(p), int(n_iter))
    cuda.synchronize()

    baseline_px = baseline_gpu.copy_to_host()       # (n_px, T)
    return baseline_px.T.reshape(T, H, W)           # (T, H, W)


# ── CPU driver ────────────────────────────────────────────


def cpu_als_baseline(gauss: np.ndarray, lam: float = LAM, p: float = P, n_iter: int = N_ITER) -> np.ndarray:
    """Run ALS pixel-wise on CPU (Thomas algorithm vectorised across pixels).

    Same algorithm as the GPU kernel, but each numpy operation acts on all
    pixels simultaneously.  The only Python loop is over T (frame index).
    """
    T, H, W = gauss.shape
    y = gauss.reshape(T, -1).astype(np.float64)    # (T, n_px)

    lll_m = np.full(T, 2.0)
    lll_m[0] = lll_m[-1] = 1.0

    z = y.copy()
    w = np.ones_like(y)
    c = np.empty_like(y)
    d = np.empty_like(y)

    for _ in range(n_iter):
        # Forward sweep — all pixels in parallel via numpy broadcast
        denom = w[0] + lam * lll_m[0]          # (n_px,)
        c[0] = -lam / denom
        d[0] = w[0] * y[0] / denom

        for i in range(1, T):
            denom = w[i] + lam * lll_m[i] + lam * c[i - 1]
            if i < T - 1:
                c[i] = -lam / denom
            d[i] = (w[i] * y[i] + lam * d[i - 1]) / denom

        # Backward substitution
        z[-1] = d[-1]
        for i in range(T - 2, -1, -1):
            z[i] = d[i] - c[i] * z[i + 1]

        # Update weights
        w = np.where(y > z, p, 1.0 - p)

    return z.reshape(T, H, W).astype(np.float32)


# ── Main ──────────────────────────────────────────────────

USE_GPU = cuda.is_available()
print("GPU detected — using CUDA kernel" if USE_GPU else "No GPU — using CPU fallback")

for fname in FILES:
    fpath = PROC_DIR / fname
    if not fpath.exists():
        print(f"\n[SKIP] {fname} not found in {PROC_DIR}")
        continue

    stem = fpath.stem.replace("_Biexp_Gauss", "")
    t0 = time.time()
    print(f"\n{'=' * 60}\n  {fname}")

    gauss = tifffile.imread(fpath).astype(np.float32)
    T, H, W = gauss.shape
    print(f"  Shape: ({T}, {H}, {W})   loaded {time.time() - t0:.1f}s")

    # ── ALS baseline ─────────────────────────────────────
    print(f"  Running ALS  (lam={LAM:.0e}, p={P}, n_iter={N_ITER})...")
    baseline = gpu_als_baseline(gauss) if USE_GPU else cpu_als_baseline(gauss)
    print(f"  ALS done  ({time.time() - t0:.1f}s)")

    # ── Save baseline ─────────────────────────────────────
    tifffile.imwrite(PROC_DIR / f"{stem}_Biexp_Baseline.tif", baseline.astype(np.float16))
    print(f"  Saved  -> {stem}_Biexp_Baseline.tif")

    # ── Compute and save dF/F0 ───────────────────────────
    # f0 = np.maximum(baseline, 1.0)          # guard against zero baseline
    dff0 = (100 * (gauss - baseline) / baseline).astype(np.float16)
    tifffile.imwrite(PROC_DIR / f"{stem}_Biexp_dFF0.tif", dff0)
    print(f"  Saved  -> {stem}_Biexp_dFF0.tif   ({time.time() - t0:.1f}s total)")

    del gauss, baseline, dff0

print(f"\n{'=' * 60}\nAll done!")
