"""
als_baseline.py  --  Per-pixel ALS baseline estimation (CPU NumPy + CUDA GPU).

Public API
----------
als_baseline_run(stack, lam, p, n_iter, cuda_available)  ->  np.ndarray
"""

import math
import os
import warnings

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# Suppress numba performance warnings and noisy CUDA log messages
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "40")

# ── Defaults ───────────────────────────────────────────────────────────────────

LAM: float = 1e2    # smoothness: larger → smoother baseline
P: float = 0.05     # asymmetry: small → baseline hugs the lower envelope
N_ITER: int = 10    # ALS iterations (10 is usually sufficient)
MAX_T: int = 2048   # compile-time local-array size; must be >= number of frames


# ── GPU kernel ─────────────────────────────────────────────────────────────────


@cuda.jit
def _gpu_als_kernel(data, baseline, lam, p, n_iter) -> None:
    """ALS baseline per pixel using Thomas algorithm (1st-order differences).

    data / baseline shape: (T, n_pixels) — frame index first for coalesced access.
    Each thread handles one pixel independently.
    Local arrays are fixed at MAX_T = 2048 frames.

    System solved each iteration: (W + λ L'L) z = W y
      L'L  main diag : [1, 2, 2, ..., 2, 1]
      L'L  off  diag : -1  (constant)
    """
    px = cuda.grid(1)
    if px >= data.shape[1]:
        return

    T = data.shape[0]

    # Thread-local arrays (stored in GPU local/register memory)
    z = cuda.local.array(2048, dtype=np.float32)   # current baseline estimate
    w = cuda.local.array(2048, dtype=np.float32)   # ALS weights
    c = cuda.local.array(2048, dtype=np.float32)   # Thomas: modified upper diagonal
    d = cuda.local.array(2048, dtype=np.float32)   # Thomas: modified RHS

    for i in range(T):
        z[i] = data[i, px]
        w[i] = 1.0

    for _ in range(n_iter):
        # ── Forward sweep (Thomas algorithm) ─────────────────────────────
        denom = w[0] + lam * 1.0
        c[0] = -lam / denom
        d[0] = w[0] * data[0, px] / denom

        for i in range(1, T):
            lll_m = 1.0 if i == T - 1 else 2.0
            denom = w[i] + lam * lll_m + lam * c[i - 1]
            if i < T - 1:
                c[i] = -lam / denom
            d[i] = (w[i] * data[i, px] + lam * d[i - 1]) / denom

        # ── Backward substitution ─────────────────────────────────────────
        z[T - 1] = d[T - 1]
        for i in range(T - 2, -1, -1):
            z[i] = d[i] - c[i] * z[i + 1]

        # ── Update weights ────────────────────────────────────────────────
        for i in range(T):
            w[i] = p if data[i, px] > z[i] else (1.0 - p)

    for i in range(T):
        baseline[i, px] = z[i]


# ── GPU driver ─────────────────────────────────────────────────────────────────


def _gpu_als(stack: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    """Run ALS pixel-wise on GPU. Input shape: (T, H, W) float32."""
    T, H, W = stack.shape
    if T > MAX_T:
        msg = f"T={T} exceeds MAX_T={MAX_T}; increase MAX_T and recompile"
        raise ValueError(msg)

    n_px = H * W
    # Reshape to (T, n_pixels) — frame index first for coalesced access across threads
    data_px = np.ascontiguousarray(stack.reshape(T, n_px), dtype=np.float32)

    data_gpu = cuda.to_device(data_px)
    baseline_gpu = cuda.device_array_like(data_px)

    threads = 128
    blocks = math.ceil(n_px / threads)
    _gpu_als_kernel[blocks, threads](data_gpu, baseline_gpu, float(lam), float(p), int(n_iter))
    cuda.synchronize()

    baseline_px = baseline_gpu.copy_to_host()   # (T, n_px)
    return baseline_px.reshape(T, H, W)         # (T, H, W)


# ── CPU driver ─────────────────────────────────────────────────────────────────


def _cpu_als(stack: np.ndarray, lam: float, p: float, n_iter: int) -> np.ndarray:
    """Run ALS pixel-wise on CPU (Thomas algorithm vectorised across pixels).

    Same algorithm as the GPU kernel, but each numpy operation acts on all
    pixels simultaneously. The only Python loop is over T (frame index).
    """
    T, H, W = stack.shape
    y = stack.reshape(T, -1).astype(np.float64)    # (T, n_px)

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


# ── Public dispatch ────────────────────────────────────────────────────────────


def als_baseline_run(
    stack: np.ndarray,
    lam: float = LAM,
    p: float = P,
    n_iter: int = N_ITER,
    cuda_available: bool = False,
) -> np.ndarray:
    """Estimate per-pixel ALS baseline of an image stack.

    Routes to GPU (CUDA) or CPU (NumPy) based on `cuda_available`.

    Args:
        stack: Input array of shape (T, H, W).
        lam: Smoothness parameter; larger → smoother baseline.
        p: Asymmetry weight; small → baseline hugs the lower envelope.
        n_iter: Number of ALS iterations.
        cuda_available: Route to GPU implementation if True.

    Returns:
        Baseline array of same shape as input, dtype float32.
    """
    stack_f32 = stack.astype(np.float32)
    if cuda_available:
        return _gpu_als(stack_f32, lam, p, n_iter)
    return _cpu_als(stack_f32, lam, p, n_iter)
