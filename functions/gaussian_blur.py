"""
gaussian_blur.py  --  Separable 2D Gaussian blur (CPU Numba JIT + CUDA GPU).

Public API
----------
gaussian_blur_run(stack, sigma, cuda_available)  ->  np.ndarray
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


# ── CPU Gaussian blur ──────────────────────────────────────────────────────────


@jit(nopython=True, parallel=True)
def _cpu_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel. Size auto-set to 6σ (odd) if not given."""
    if size is None:
        size = math.ceil(sigma * 6)
        if size % 2 == 0:
            size += 1
    center = size // 2
    x = np.arange(-center, center + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


@jit(nopython=True)
def _cpu_conv(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """
    Apply a 1D kernel along `axis` with edge normalization.

    axis=0 → vertical (along rows), axis=1 → horizontal (along columns).
    Out-of-bounds positions are skipped; weights are renormalized accordingly.
    """
    result = np.zeros_like(arr)
    kernel_half = len(kernel) // 2
    kernel_size = len(kernel)
    height, width = arr.shape

    for row in range(height):
        for col in range(width):
            val = 0.0
            w_sum = 0.0
            if axis == 0:  # vertical
                for kp in range(kernel_size):
                    nr = row + kp - kernel_half
                    if 0 <= nr < height:
                        val += arr[nr, col] * kernel[kp]
                        w_sum += kernel[kp]
            else:           # horizontal
                for kp in range(kernel_size):
                    nc = col + kp - kernel_half
                    if 0 <= nc < width:
                        val += arr[row, nc] * kernel[kp]
                        w_sum += kernel[kp]
            result[row, col] = val / w_sum if w_sum > 0 else arr[row, col]

    return result


@jit(nopython=True, parallel=True)
def _cpu_gaussian_blur(stack: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separable 2D Gaussian blur on CPU (parallel over frames).

    Applies horizontal then vertical 1D convolution — O(n·k) vs O(n·k²) for 2D.
    """
    n_frames = stack.shape[0]
    output = np.zeros_like(stack)
    kernel = _cpu_kernel(sigma)
    for frame_idx in prange(n_frames):
        blurred_h = _cpu_conv(stack[frame_idx], kernel, axis=1)
        output[frame_idx] = _cpu_conv(blurred_h, kernel, axis=0)
    return output


# ── GPU Gaussian blur ──────────────────────────────────────────────────────────


@cuda.jit
def _gpu_kernel(kernel_out: np.ndarray, sigma: float, size: int) -> None:
    """CUDA kernel: each thread computes one kernel element; thread 0 normalizes."""
    idx = cuda.grid(1)
    if idx >= size:
        return
    center = size // 2
    dist = idx - center
    kernel_out[idx] = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
    cuda.syncthreads()
    if idx == 0:
        total = 0.0
        for i in range(size):
            total += kernel_out[i]
        for i in range(size):
            kernel_out[i] /= total


@cuda.jit
def _gpu_conv(
    input_img: np.ndarray, output_img: np.ndarray, kernel: np.ndarray, height: int, width: int, kernel_size: int, axis: int
) -> None:
    """
    CUDA kernel: 1D convolution along a given axis with boundary clamping.

    axis=0 → vertical (along rows), axis=1 → horizontal (along columns).
    All threads in a warp take the same branch — no warp divergence.
    """
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row >= height or col >= width:
        return
    half = kernel_size // 2
    val = 0.0
    w_sum = 0.0
    for kp in range(kernel_size):
        if axis == 0:  # vertical
            nr = min(max(row + kp - half, 0), height - 1)
            val += input_img[nr * width + col] * kernel[kp]
        else:           # horizontal
            nc = min(max(col + kp - half, 0), width - 1)
            val += input_img[row * width + nc] * kernel[kp]
        w_sum += kernel[kp]
    output_img[row * width + col] = val / w_sum if w_sum > 0 else input_img[row * width + col]


def _gpu_gaussian_blur(stack: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separable 2D Gaussian blur on GPU.

    Pre-allocates GPU buffers once and reuses them across all frames to
    minimize host↔device transfer overhead.
    """
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32)

    n_frames, height, width = stack.shape
    kernel_size = math.ceil(sigma * 6)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Generate normalized kernel on GPU
    kernel_gpu = cuda.device_array(kernel_size, dtype=np.float32)
    tpb_k = min(256, kernel_size)
    bpg_k = math.ceil(kernel_size / tpb_k)
    _gpu_kernel[bpg_k, tpb_k](kernel_gpu, sigma, kernel_size)
    cuda.synchronize()

    # 2D thread blocks for image convolution (16×16 = 256 threads/block)
    threads_2d = (16, 16)
    blocks_2d = (math.ceil(width / 16), math.ceil(height / 16))

    # Pre-allocate GPU buffers once (reused across all frames)
    frame_size = height * width
    buf_in = cuda.device_array(frame_size, dtype=np.float32)
    buf_tmp = cuda.device_array(frame_size, dtype=np.float32)
    buf_out = cuda.device_array(frame_size, dtype=np.float32)
    host_buf = np.empty(frame_size, dtype=np.float32)

    output = np.zeros_like(stack)
    for frame_idx in range(n_frames):
        cuda.to_device(np.ascontiguousarray(stack[frame_idx]).flatten(), to=buf_in)
        _gpu_conv[blocks_2d, threads_2d](buf_in, buf_tmp, kernel_gpu, height, width, kernel_size, 1)  # horizontal
        cuda.synchronize()
        _gpu_conv[blocks_2d, threads_2d](buf_tmp, buf_out, kernel_gpu, height, width, kernel_size, 0)  # vertical
        cuda.synchronize()
        buf_out.copy_to_host(host_buf)
        output[frame_idx] = host_buf.reshape(height, width)

    return output


# ── Public dispatch ────────────────────────────────────────────────────────────


def gaussian_blur_run(stack: np.ndarray, sigma: float, cuda_available: bool) -> np.ndarray:
    """
    Apply separable 2D Gaussian blur to an image stack.

    Routes to GPU (CUDA) or CPU (Numba JIT) based on `cuda_available`.

    Args:
        stack: Input array of shape (n_frames, H, W).
        sigma: Standard deviation of the Gaussian kernel (larger = more blur).
        cuda_available: Route to GPU implementation if True.

    Returns:
        Blurred array of same shape as input.
    """
    if cuda_available:
        return _gpu_gaussian_blur(stack, sigma)
    return _cpu_gaussian_blur(stack, sigma)
