"""
detrend.py  --  Moving-average and bi-exponential detrend (CPU Numba JIT + CUDA GPU).

Public API
----------
mov_detrend(stack, cuda_available, window_size=101)  ->  np.ndarray
biexp_detrend(img, tau1, tau2, cuda_available)        ->  np.ndarray
align_to_min(stack)                                   ->  np.ndarray
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
from numba import cuda, jit, prange
from numba.core.errors import NumbaPerformanceWarning

# Suppress numba performance warnings and noisy CUDA log messages
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "40")


# ── Moving-average detrend ─────────────────────────────────────────────────────


@jit(nopython=True, parallel=True)
def _cpu_mov(pixel_data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Numba JIT moving-average detrend on CPU (parallel over pixels).

    Args:
        pixel_data: shape (n_pixels, n_frames), float32
        window_size: width of the centred moving-average window (frames)

    Returns:
        Detrended array, same shape as input.
    """
    n_pixels, n_frames = pixel_data.shape
    output = np.zeros_like(pixel_data, dtype=np.float32)

    for pixel_idx in prange(n_pixels):
        moving_avgs = np.zeros(n_frames, dtype=np.float32)
        for frame_idx in prange(n_frames):
            window_start = max(0, frame_idx - window_size // 2)
            window_end = min(n_frames, frame_idx + window_size // 2)
            moving_avgs[frame_idx] = np.mean(pixel_data[pixel_idx, window_start:window_end])

        # Trend = moving average minus its lower endpoint (edge normalization)
        edge_min = min(moving_avgs[0], moving_avgs[-1])
        for frame_idx in prange(n_frames):
            output[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - (moving_avgs[frame_idx] - edge_min)

    return output


@cuda.jit
def _gpu_mov(pixel_data: np.ndarray, output: np.ndarray, window_size: int) -> None:
    """
    CUDA kernel: one thread per pixel for moving-average detrend.

    Uses a fixed-size local array (2048 frames) for moving averages.
    """
    pixel_idx = cuda.grid(1)
    if pixel_idx >= pixel_data.shape[0]:
        return

    n_frames = pixel_data.shape[1]
    half_window = window_size // 2
    moving_averages = cuda.local.array(2048, dtype=np.float32)

    for frame_idx in range(n_frames):
        window_start = max(0, frame_idx - half_window)
        window_end = min(n_frames, frame_idx + half_window)
        window_sum = 0.0
        for k in range(window_start, window_end):
            window_sum += pixel_data[pixel_idx, k]
        moving_averages[frame_idx] = window_sum / (window_end - window_start)

    edge_min = min(moving_averages[0], moving_averages[n_frames - 1])
    for frame_idx in range(n_frames):
        output[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - (moving_averages[frame_idx] - edge_min)


def mov_detrend(stack: np.ndarray, cuda_available: bool, window_size: int = 101) -> np.ndarray:
    """
    Moving-average detrend of an image stack (GPU if available, else CPU Numba JIT).

    Flattens the stack to (n_pixels, n_frames), detrends each pixel in parallel,
    then reshapes back and aligns pixel time-means to the global minimum.

    Args:
        stack: Input array of shape (n_frames, H, W).
        cuda_available: Route to GPU kernel if True.
        window_size: Width of the centred moving-average window (frames).

    Returns:
        Detrended stack, shape (n_frames, H, W), float32.
    """
    n_frames, height, width = stack.shape
    pixel_data = stack.reshape(n_frames, -1).T.astype(np.float32)  # (n_pixels, n_frames)

    if cuda_available:
        gpu_input = cuda.to_device(pixel_data)
        gpu_output = cuda.to_device(np.zeros_like(pixel_data))
        threads = 256
        blocks = math.ceil(pixel_data.shape[0] / threads)
        _gpu_mov[blocks, threads](gpu_input, gpu_output, window_size)
        cuda.synchronize()
        detrended_pixels = gpu_output.copy_to_host()
    else:
        detrended_pixels = _cpu_mov(pixel_data, window_size)

    detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)

    # Align pixel time-means: shift so the lowest-mean pixel has mean = 0
    pixel_means = detrended_stack.mean(axis=0)
    detrended_stack -= (pixel_means - pixel_means.min())[None, :, :]
    return detrended_stack


# ── Bi-exponential detrend ─────────────────────────────────────────────────────


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
        coeffs = np.dot(basis_pinv, y)        # least-squares coefficients (3,)
        trend = np.dot(basis_matrix, coeffs)   # reconstructed baseline (T,)
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

    basis_pinv   (3, T): projects pixel trace onto 3-component basis.
    basis_matrix (T, 3): reconstructs trend from the 3 coefficients.
    Trend computed on-the-fly per frame to avoid large local arrays.
    Edge normalization: shift so the lower endpoint value = 0.
    """
    i = cuda.grid(1)
    if i >= img_flat.shape[0]:
        return

    T = img_flat.shape[1]

    # coeffs = basis_pinv @ y  (manual dot product, result size = 3)
    coeffs = cuda.local.array(3, dtype=np.float64)
    for k in range(3):
        s = 0.0
        for t in range(T):
            s += basis_pinv[k, t] * img_flat[i, t]
        coeffs[k] = s

    # Compute trend only at endpoints for edge normalization
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
    (n_pixels, T), runs detrend per pixel, reshapes back to (n_frames, H, W).

    Args:
        img: Input stack, shape (n_frames, H, W).
        tau1: Slow time constant (frames).
        tau2: Fast time constant (frames).
        cuda_available: Route to GPU kernel if True.

    Returns:
        Detrended stack, shape (n_frames, H, W), float32.
    """
    n_frames, H, W = img.shape
    t = np.arange(n_frames, dtype=np.float64)

    # basis_matrix (T, 3): columns = [exp(-t/tau1), exp(-t/tau2), constant ones]
    basis_matrix = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2), np.ones(n_frames)])
    # basis_pinv (3, T): pseudo-inverse — the least-squares projection operator
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


# ── Shared utility ─────────────────────────────────────────────────────────────


def align_to_min(stack: np.ndarray) -> np.ndarray:
    """
    Shift each pixel's time series so that the lowest-mean pixel has mean = 0.

    Computes the per-pixel time-mean, finds the global minimum, and subtracts
    the per-pixel offset so all pixels are aligned relative to the quietest one.
    """
    means = stack.mean(axis=0)
    offset = means - means.min()
    return stack - offset[None, :, :]
