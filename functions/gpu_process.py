# Standard library imports
import logging
import math
import os
import time

# Third-party imports
import numpy as np
from numba import cuda
from rich.console import Console
from rich.logging import RichHandler

# Reduce CUDA memory allocation logging verbosity
os.environ.setdefault("NUMBA_CUDA_LOG_LEVEL", "30")  # WARNING level

# Local application imports (use relative imports to avoid circular dependency)
from .gpu_detrend import gpu_detrend_jitted
from .gpu_gauss import gpu_gaussian_blur

# Setup rich console and logging
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("rich")


def process_on_gpu(image_stack: np.ndarray, window_size: int = 101, sigma: float = 6.0) -> tuple:
    """
    Process image stack using GPU for detrending and CPU for spatial averaging.

    Args:
        image_stack: Input array of shape (n_frames, height, width)
        roi_size: Size of Region of Interest (ROI)
        window_size: Size of moving average window for detrending
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Tuple of (detrended_stack, averaged_stack, gaussian_stack)

    """
    n_frames, height, width = image_stack.shape

    # Prepare data for detrending
    pixels_time_series = image_stack.reshape(n_frames, -1).T
    detrended_pixels = np.zeros_like(pixels_time_series, dtype=np.float32)

    # Transfer data to GPU
    gpu_input = cuda.to_device(pixels_time_series.astype(np.float32))
    gpu_output = cuda.to_device(detrended_pixels)

    # Configure CUDA grid
    threads_per_block = 256
    blocks_per_grid = math.ceil(pixels_time_series.shape[0] / threads_per_block)

    # Perform detrending on GPU
    console.print("[cyan]Detrending pixels on GPU...")
    start_time = time.time()
    gpu_detrend_jitted[blocks_per_grid, threads_per_block](gpu_input, gpu_output, window_size)
    cuda.synchronize()
    detrended_pixels = gpu_output.copy_to_host()
    console.print(f"Detrending time: {time.time() - start_time:.2f} seconds")

    # Reshape detrended data back to original dimensions
    detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)
    pixel_offsets = np.mean(detrended_stack, axis=0)
    pixel_offsets_adjust = pixel_offsets - np.min(pixel_offsets)
    ## Shift all pixel values to the center of the uint16 range
    # pixel_offsets_adjust = pixel_offsets - 450
    detrended_stack -= pixel_offsets_adjust

    # Apply Gaussian blur to detrended stack using CUDA acceleration
    console.print("[cyan]Applying Gaussian blur on GPU...")
    start_time = time.time()
    gaussian_stack = gpu_gaussian_blur(detrended_stack, sigma)
    console.print(f"GPU Gaussian blur time: {time.time() - start_time:.2f} seconds")

    return detrended_stack, gaussian_stack
