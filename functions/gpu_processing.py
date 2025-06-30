# Standard library imports
import logging
import math
import time

# Third-party imports
import numpy as np
from numba import cuda
from rich.console import Console
from rich.logging import RichHandler

# Local imports
from .cuda_kernel_detrend import detrend_kernel
from .spatial_processing import compute_spatial_averages

# Setup rich console and logging
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("rich")


def process_on_gpu(image_stack: np.ndarray, roi_size: int, window_size: int = 100) -> tuple:
    """
    Process image stack using GPU for detrending and CPU for spatial averaging.

    Args:
        image_stack: Input array of shape (n_frames, height, width)
        roi_size: Size of Region of Interest (ROI)
        window_size: Size of moving average window for detrending

    Returns:
        Tuple of (detrended_stack, averaged_stack)

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
    detrend_kernel[blocks_per_grid, threads_per_block](gpu_input, gpu_output, window_size)
    cuda.synchronize()
    detrended_pixels = gpu_output.copy_to_host()
    console.print(f"Detrending time: {time.time() - start_time:.2f} seconds")

    # Reshape detrended data back to original dimensions
    detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)
    pixel_offsets = np.mean(detrended_stack, axis=0)
    pixel_offsets_adjust = pixel_offsets - np.min(pixel_offsets)
    detrended_stack -= pixel_offsets_adjust

    # Compute spatial averages
    console.print("[cyan]Computing spatial averages...")
    start_time = time.time()
    averaged_stack = compute_spatial_averages(detrended_stack, roi_size)

    console.print(f"Spatial averaging time: {time.time() - start_time:.2f} seconds")

    return detrended_stack, averaged_stack
