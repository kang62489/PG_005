"""
als_cal.py  --  Per-pixel ALS slow-fluctuation estimation (CPU NumPy + CUDA GPU).

Public API
----------
als_cal_run(stack, lam, p, n_iter, cuda_available)  ->  np.ndarray
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
MAX_T: int = 4096   # compile-time local-array size; must be >= number of frames


# ── GPU kernel ─────────────────────────────────────────────────────────────────


@cuda.jit
def _gpu_als_kernel(data, slow_fluc, lam, p, n_iter) -> None:
    """ALS baseline per pixel using Thomas algorithm (1st-order differences).

    data / slow_fluc shape: (T, n_pixels) — frame index first for coalesced access.
    Each thread handles one pixel independently.
    Local arrays are fixed at MAX_T frames (compile-time constant).

    System solved each iteration: (W + λ L'L) z = W y
      L'L  main diag : [1, 2, 2, ..., 2, 1]
      L'L  off  diag : -1  (constant)

    Optimisations vs. naive version:
      - w[] eliminated: weight computed on the fly from z[] saved from previous iter
      - Edge rows (i=0, i=T-1) special-cased to remove branch from the hot interior loop
      - All scalars cast to float32 at entry to prevent float64 upcast contamination
    """
    px = cuda.grid(1)
    if px >= data.shape[1]:
        return

    T = data.shape[0]

    # Cast scalars to float32 — prevents silent float64 upcast of float32 array ops
    lam32 = np.float32(lam)
    p32   = np.float32(p)
    weight_below   = np.float32(1.0) - p32   # complement weight (data below baseline)
    two_lam  = np.float32(2.0) * lam32  # interior diagonal multiplier

    # 3 local arrays instead of 4 — w[] dropped (computed on the fly)
    z = cuda.local.array(MAX_T, dtype=np.float32)   # baseline estimate
    c = cuda.local.array(MAX_T, dtype=np.float32)   # Thomas: modified upper diagonal
    d = cuda.local.array(MAX_T, dtype=np.float32)   # Thomas: modified RHS

    for i in range(T):
        z[i] = data[i, px]

    for _ in range(n_iter):
        # ── Forward sweep ────────────────────────────────────────────────
        # i = 0  (edge — L'L diagonal = 1)
        weight_first    = p32 if data[0, px] > z[0] else weight_below
        denom = weight_first + lam32
        c[0]  = -lam32 / denom
        d[0]  = weight_first * data[0, px] / denom

        # i = 1 … T-2  (interior — L'L diagonal = 2, no branch needed)
        for i in range(1, T - 1):
            weight_cur = p32 if data[i, px] > z[i] else weight_below
            denom      = weight_cur + two_lam + lam32 * c[i - 1]
            c[i]       = -lam32 / denom
            d[i]       = (weight_cur * data[i, px] + lam32 * d[i - 1]) / denom

        # i = T-1  (edge — L'L diagonal = 1, c[T-1] not needed)
        weight_last       = p32 if data[T - 1, px] > z[T - 1] else weight_below
        denom    = weight_last + lam32 + lam32 * c[T - 2]
        d[T - 1] = (weight_last * data[T - 1, px] + lam32 * d[T - 2]) / denom

        # ── Backward substitution ─────────────────────────────────────────
        z[T - 1] = d[T - 1]
        for i in range(T - 2, -1, -1):
            z[i] = d[i] - c[i] * z[i + 1]

    for i in range(T):
        slow_fluc[i, px] = z[i]


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

    # Flush any pending Numba GPU deallocations before allocating large arrays,
    # to avoid CUDA_ERROR_ILLEGAL_ADDRESS from memory pressure.
    cuda.current_context().deallocations.clear()

    data_gpu = cuda.to_device(data_px)
    slow_fluc_gpu = cuda.device_array_like(data_px)

    threads = 128
    blocks = math.ceil(n_px / threads)
    _gpu_als_kernel[blocks, threads](data_gpu, slow_fluc_gpu, float(lam), float(p), int(n_iter))
    cuda.synchronize()

    slow_fluc_px = slow_fluc_gpu.copy_to_host()   # (T, n_px)
    return slow_fluc_px.reshape(T, H, W)           # (T, H, W)


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


def als_cal_run(
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
