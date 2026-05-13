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
    half_window = window_size // 2

    for pixel_idx in prange(n_pixels):
        # Moving average at frame 0 — needed for edge_min
        first_window_right_bound = min(n_frames, half_window)
        first_window_sum = np.float32(0.0)
        for k in range(first_window_right_bound):
            first_window_sum += pixel_data[pixel_idx, k]
        first_window_moving_avg = first_window_sum / np.float32(first_window_right_bound)

        # Moving average at last frame — needed for edge_min
        last_window_left_bound = max(0, n_frames - 1 - half_window)
        last_window_sum = np.float32(0.0)
        for k in range(last_window_left_bound, n_frames):
            last_window_sum += pixel_data[pixel_idx, k]
        last_window_moving_avg = last_window_sum / np.float32(n_frames - last_window_left_bound)

        edge_min = min(first_window_moving_avg, last_window_moving_avg)

        # Sliding-window pass: compute moving avg and write output in one loop
        window_left_bound = 0
        window_right_bound = first_window_right_bound
        running_sum = first_window_sum
        for frame_idx in range(n_frames):
            moving_avg_at_frame = running_sum / np.float32(window_right_bound - window_left_bound)
            output[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - moving_avg_at_frame + edge_min
            next_window_left_bound = max(0, frame_idx + 1 - half_window)
            next_window_right_bound = min(n_frames, frame_idx + 1 + half_window)
            if next_window_right_bound > window_right_bound:
                running_sum += pixel_data[pixel_idx, next_window_right_bound - 1]
            if next_window_left_bound > window_left_bound:
                running_sum -= pixel_data[pixel_idx, window_left_bound]
            window_left_bound = next_window_left_bound
            window_right_bound = next_window_right_bound

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

    # -- Moving average at frame 0 (for edge_min) ---------------------------------
    first_window_right_bound = min(n_frames, half_window)
    first_window_sum = np.float32(0.0)
    for k in range(first_window_right_bound):
        first_window_sum += pixel_data[pixel_idx, k]
    first_window_moving_avg = first_window_sum / np.float32(first_window_right_bound)

    # -- Moving average at last frame (for edge_min) ------------------------------
    last_window_left_bound = max(0, n_frames - 1 - half_window)
    last_window_sum = np.float32(0.0)
    for k in range(last_window_left_bound, n_frames):
        last_window_sum += pixel_data[pixel_idx, k]
    last_window_moving_avg = last_window_sum / np.float32(n_frames - last_window_left_bound)

    edge_min = min(first_window_moving_avg, last_window_moving_avg)

    # -- Sliding-window pass: compute moving avg and write output in one loop -----
    window_left_bound = 0
    window_right_bound = first_window_right_bound
    running_sum = first_window_sum
    for frame_idx in range(n_frames):
        moving_avg_at_frame = running_sum / np.float32(window_right_bound - window_left_bound)
        output[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - moving_avg_at_frame + edge_min
        next_window_left_bound = max(0, frame_idx + 1 - half_window)
        next_window_right_bound = min(n_frames, frame_idx + 1 + half_window)
        if next_window_right_bound > window_right_bound:
            running_sum += pixel_data[pixel_idx, next_window_right_bound - 1]
        if next_window_left_bound > window_left_bound:
            running_sum -= pixel_data[pixel_idx, window_left_bound]
        window_left_bound = next_window_left_bound
        window_right_bound = next_window_right_bound


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

    return align_to_min(detrended_stack)


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

    trend_0 = np.float32(0.0)
    trend_end = np.float32(0.0)
    for k in range(3):
        trend_0 += basis_matrix[0, k] * coeffs[k]
        trend_end += basis_matrix[n_frames - 1, k] * coeffs[k]
    edge_min = min(trend_0, trend_end)

    for frame_idx in range(n_frames):
        trend_t = np.float32(0.0)
        for k in range(3):
            trend_t += basis_matrix[frame_idx, k] * coeffs[k]
        output[frame_idx, pixel_idx] = img_flat[frame_idx, pixel_idx] - trend_t + edge_min


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
        return align_to_min(output_flat_gpu.reshape(n_frames, H, W))
    else:
        # (n_pixels, n_frames) — each pixel's trace is contiguous in memory
        img_flat_cpu = img.reshape(n_frames, -1).T.astype(np.float32)
        output_flat_cpu = _cpu_biexp(img_flat_cpu, basis_pinv, basis_matrix)
        return align_to_min(output_flat_cpu.T.reshape(n_frames, H, W))


# ── Shared utility ─────────────────────────────────────────────────────────────


def align_to_min(stack: np.ndarray, floor: float = 3.0) -> np.ndarray:
    """
    Normalise each pixel's time series to a common baseline of ``floor``.

    Computes the per-pixel time-mean and subtracts it from each pixel's trace,
    then lifts everything by ``floor``.  Every pixel's baseline therefore lands at
    ``floor`` (~3.0), with dF fluctuations preserved around it.  This avoids
    division-by-zero in dF/F0 calculations.
    """
    means = stack.mean(axis=0)
    return (stack - means[None, :, :] + floor).astype(np.float16)
