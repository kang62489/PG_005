# Standard library imports
import logging

# Third-party imports
import math
import os
import warnings

import numpy as np
from numba import jit, prange
from numba.core.errors import NumbaPerformanceWarning

# Suppress numba performance warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

# Suppress CUDA memory management info messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["NUMBA_DISABLE_CUDA"] = "0"
os.environ["NUMBA_CUDA_LOG_LEVEL"] = "40"  # WARNING level

logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.ERROR)
logging.getLogger("numba.cuda.cudadrv.runtime").setLevel(logging.ERROR)


@jit(nopython=True, parallel=True)
def gaussian_kernel_1d(sigma: float, size: int | None = None) -> np.ndarray:
    """
    Create a 1D Gaussian kernel array.

    Args:
        sigma: Standard deviation of the Gaussian distribution
        size: Size of the kernel (should be odd number). If None, auto-calculated using 6-sigma rule.

    Returns:
        Normalized Gaussian kernel where all values sum to 1.0

    Example:
        sigma=2.0 creates kernel: [0.003, 0.016, 0.061, 0.184, 0.267, 0.303, 0.267, 0.184, 0.061, 0.016, 0.003]

    """
    # Auto-calculate kernel size using 6-sigma rule (captures 99.7% of distribution)
    if size is None:
        size = math.ceil(sigma * 6)
        # Ensure odd size for symmetric kernel
        if size % 2 == 0:
            size += 1

    # Create coordinate array centered at 0
    # For size=11: x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    center = size // 2
    x = np.arange(-center, center + 1, dtype=np.float32)

    # Calculate Gaussian values: exp(-(x²)/(2σ²))
    kernel = np.exp(-(x**2) / (2 * sigma**2))

    # Normalize to make sum = 1.0 (preserves brightness)
    return kernel / kernel.sum()


@jit(nopython=True)
def convolve_1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """
    Apply 1D convolution along specified axis using 'same' mode with edge normalization.

    Args:
        arr: Input 2D array
        kernel: 1D convolution kernel
        axis: Axis along which to apply convolution (0=vertical, 1=horizontal)

    Returns:
        Convolved array with same shape as input

    """
    result = np.zeros_like(arr)
    kernel_half = len(kernel) // 2
    kernel_size = len(kernel)
    height, width = arr.shape

    # Apply convolution to every pixel
    for row in range(height):
        for col in range(width):
            val = 0.0
            weight_sum = 0.0

            if axis == 0:  # Vertical convolution
                for kernel_pos in range(kernel_size):
                    neighbor_row = row + kernel_pos - kernel_half
                    if 0 <= neighbor_row < height:
                        val += arr[neighbor_row, col] * kernel[kernel_pos]
                        weight_sum += kernel[kernel_pos]
            else:  # Horizontal convolution
                for kernel_pos in range(kernel_size):
                    neighbor_col = col + kernel_pos - kernel_half
                    if 0 <= neighbor_col < width:
                        val += arr[row, neighbor_col] * kernel[kernel_pos]
                        weight_sum += kernel[kernel_pos]

            # Normalize by actual weights used or fallback to original value
            result[row, col] = val / weight_sum if weight_sum > 0 else arr[row, col]

    return result


@jit(nopython=True, parallel=True)
def cpu_gaussian_blur(detrended_stack: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to image stack using separable convolution for efficiency.

    Uses separable filtering: applies 1D Gaussian horizontally, then vertically.
    This is much faster than 2D convolution: O(n*k) vs O(n*k²) complexity.

    Args:
        detrended_stack: Input array of shape (n_frames, height, width)
        sigma: Standard deviation for Gaussian kernel (larger = more blur)

    Returns:
        Blurred array of same shape as input

    Example:
        sigma=1.0 -> light smoothing
        sigma=4.0 -> heavy blurring

    """
    n_frames, height, width = detrended_stack.shape
    gaussian_stack = np.zeros_like(detrended_stack)

    # Create 1D Gaussian kernel (reused for all frames)
    kernel = gaussian_kernel_1d(sigma)

    # Process each frame in parallel using separable filtering
    for frame_idx in prange(n_frames):
        # Step 1: Apply horizontal blur (left-right smoothing)
        horizontal_blurred = convolve_1d(detrended_stack[frame_idx], kernel, axis=1)

        # Step 2: Apply vertical blur to horizontally blurred result (up-down smoothing)
        gaussian_stack[frame_idx] = convolve_1d(horizontal_blurred, kernel, axis=0)

    return gaussian_stack
