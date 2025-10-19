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
def gaussian_kernel_1d(sigma: float, size: int | None) -> np.ndarray:
    """
    Create a 1D Gaussian kernel array.

    Args:
        sigma: Standard deviation of the Gaussian distribution
        size: Size of the kernel (should be odd number)

    """
    if size is None:
        size = math.ceil(sigma * 6)
        if size % 2 == 0:
            size += 1

    x = np.arange(-(size // 2), (size // 2) + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normaize to make sum of kernel = 1


@jit(nopython=True)
def convolve_1d(arr: np.ndarray, kernel: np.ndarray, axis: int, mode: str = "valid") -> np.ndarray:
    """
    Apply 1D convolution along specified axis.

    Args:
        arr: Input 2D array
        kernel: 1D convolution kernel
        axis: Axis along which to apply convolution (0=vertical, 1=horizontal)
        mode: Convolution mode ('valid' or 'same')

    Returns:
        Convolved array

    """
    result = np.zeros_like(arr)
    kernel_size = len(kernel)
    kernel_half = kernel_size // 2

    if axis == 0:  # Vertical convolution
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = 0.0
                if mode == "same":
                    # For 'same' mode, normalize by the sum of weights actually used
                    weight_sum = 0.0
                    for k in range(kernel_size):
                        idx = i + k - kernel_half
                        if 0 <= idx < arr.shape[0]:
                            val += arr[idx, j] * kernel[k]
                            weight_sum += kernel[k]

                    # Normalize by the sum of weights that were actually used
                    if weight_sum > 0:
                        result[i, j] = val / weight_sum
                    else:
                        result[i, j] = arr[i, j]  # Fallback if no kernel weights were applied
                else:
                    # Original 'valid' mode behavior
                    for k in range(kernel_size):
                        idx = i + k - kernel_half
                        if 0 <= idx < arr.shape[0]:
                            val += arr[idx, j] * kernel[k]
                    result[i, j] = val
    else:  # axis == 1, Horizontal convolution
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = 0.0
                if mode == "same":
                    # For 'same' mode, normalize by the sum of weights actually used
                    weight_sum = 0.0
                    for k in range(kernel_size):
                        idx = j + k - kernel_half
                        if 0 <= idx < arr.shape[1]:
                            val += arr[i, idx] * kernel[k]
                            weight_sum += kernel[k]

                    # Normalize by the sum of weights that were actually used
                    if weight_sum > 0:
                        result[i, j] = val / weight_sum
                    else:
                        result[i, j] = arr[i, j]  # Fallback if no kernel weights were applied
                else:
                    # Original 'valid' mode behavior
                    for k in range(kernel_size):
                        idx = j + k - kernel_half
                        if 0 <= idx < arr.shape[1]:
                            val += arr[i, idx] * kernel[k]
                    result[i, j] = val

    return result


@jit(nopython=True, parallel=True)
def cpu_gaussian_blur(detrended_stack: np.ndarray, sigma: float, mode: str = "same") -> np.ndarray:
    """
    Apply Gaussian blur to detrended stack using separable convolution.

    Args:
        detrended_stack: Input array of shape (n_frames, height, width)
        sigma: Standard deviation for Gaussian kernel
        mode: Convolution mode ('valid' or 'same')

    Returns:
        Blurred array of same shape as input

    """
    n_frames, height, width = detrended_stack.shape
    gaussian_stack = np.zeros_like(detrended_stack)

    # Create 1D Gaussian kernel
    kernel = gaussian_kernel_1d(sigma)

    # Process each frame in parallel
    for i in prange(n_frames):
        # Apply horizontal convolution
        temp = convolve_1d(detrended_stack[i], kernel, axis=1, mode=mode)
        # Apply vertical convolution
        gaussian_stack[i] = convolve_1d(temp, kernel, axis=0, mode=mode)

    return gaussian_stack
