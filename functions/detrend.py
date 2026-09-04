"""
detrend.py  --  Bi-exponential detrend (CPU Numba JIT + CUDA GPU).

Public API
----------
biexp_detrend(img, tau1, tau2, cuda_available)        ->  np.ndarray

Returns the residual (y - trend), not a ratio. Downstream normalization to a
stack-wide z-score happens separately in functions/zscore_normalize.py.
"""

import math
import os
import warnings

import numpy as np
from numba import cuda, jit, prange
from numba.core.errors import NumbaPerformanceWarning

# Suppress numba performance warnings and noisy CUDA log messages
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "40")


# ── Bi-exponential detrend ─────────────────────────────────────────────────────


@jit(nopython=True, parallel=True, cache=True)
def _cpu_biexp(img_flat: np.ndarray, basis_pinv: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    """
    Numba JIT per-pixel bi-exp detrend on CPU (parallel over pixels).

    Maths:
      basis_matrix  (T, 3): columns are [exp(-t/tau1), exp(-t/tau2), ones]
      basis_pinv    (3, T): Moore-Penrose pseudo-inverse of basis_matrix
      For each pixel trace y (T,):
        coeffs   = basis_pinv @ y         # least-squares fit → (3,)
        trend    = basis_matrix @ coeffs  # reconstructed bi-exp baseline F0 → (T,)
        residual = y - trend              # detrended residual (not yet normalized)
    """
    n_pixels, T = img_flat.shape
    output = np.zeros_like(img_flat)
    for i in prange(n_pixels):
        y = img_flat[i]
        coeffs = np.dot(basis_pinv, y)        # least-squares coefficients (3,)
        trend = np.dot(basis_matrix, coeffs)   # reconstructed baseline (T,)
        output[i] = y - trend
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

    basis_pinv   (3, T): projects pixel trace onto 3-component basis.
    basis_matrix (T, 3): reconstructs trend from the 3 coefficients.
    Trend computed on-the-fly per frame to avoid large local arrays.
    Output is the residual (y - trend), not a ratio.
    """
    pixel_idx = cuda.grid(1)
    if pixel_idx >= img_flat.shape[1]:  # img_flat shape: (n_frames, n_pixels)
        return

    n_frames = img_flat.shape[0]

    coeffs = cuda.local.array(3, dtype=np.float32)
    for k in range(3):
        s = np.float32(0.0)
        for frame_idx in range(n_frames):
            s += basis_pinv[k, frame_idx] * img_flat[frame_idx, pixel_idx]
        coeffs[k] = s

    for frame_idx in range(n_frames):
        trend_t = np.float32(0.0)
        for k in range(3):
            trend_t += basis_matrix[frame_idx, k] * coeffs[k]
        output[frame_idx, pixel_idx] = img_flat[frame_idx, pixel_idx] - trend_t


def biexp_detrend(img: np.ndarray, tau1: float, tau2: float, cuda_available: bool) -> np.ndarray:
    """
    Bi-exp detrend dispatcher: GPU if available, else CPU Numba JIT.

    Builds basis_matrix / basis_pinv from tau1, tau2, flattens img to
    (n_pixels, T), runs detrend per pixel, reshapes back to (n_frames, H, W).

    Args:
        img: Input stack, shape (n_frames, H, W).
        tau1: Slow time constant (frames).
        tau2: Fast time constant (frames).
        cuda_available: Route to GPU kernel if True.

    Returns:
        Residual stack (y - trend), shape (n_frames, H, W), float32. Not yet
        normalized — pass through functions.zscore_normalize for that.
    """
    n_frames, H, W = img.shape
    t = np.arange(n_frames, dtype=np.float32)

    # basis_matrix (T, 3): columns = [exp(-t/tau1), exp(-t/tau2), constant ones]
    basis_matrix = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2), np.ones(n_frames, dtype=np.float32)]).astype(np.float32)
    # basis_pinv (3, T): pseudo-inverse — the least-squares projection operator
    basis_pinv = np.linalg.pinv(basis_matrix).astype(np.float32)

    if cuda_available:
        # (n_frames, n_pixels) — kernel reads img_flat[frame_idx, pixel_idx], coalesced across threads
        img_flat_gpu = img.reshape(n_frames, -1).astype(np.float32)
        n_pixels = img_flat_gpu.shape[1]
        d_img = cuda.to_device(img_flat_gpu)
        d_pinv = cuda.to_device(basis_pinv)
        d_basis = cuda.to_device(basis_matrix)
        d_out = cuda.to_device(np.empty_like(img_flat_gpu))
        threads = 256
        blocks = math.ceil(n_pixels / threads)
        _gpu_biexp[blocks, threads](d_img, d_pinv, d_basis, d_out)
        cuda.synchronize()
        output_flat_gpu = d_out.copy_to_host()
        return output_flat_gpu.reshape(n_frames, H, W)
    # (n_pixels, n_frames) — each pixel's trace is contiguous in memory
    img_flat_cpu = img.reshape(n_frames, -1).T.astype(np.float32)
    output_flat_cpu = _cpu_biexp(img_flat_cpu, basis_pinv, basis_matrix)
    return output_flat_cpu.T.reshape(n_frames, H, W)
