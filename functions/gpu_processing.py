# Standard library imports
import math
import time

# Third-party imports
import numpy as np
from numba import cuda
from rich.console import Console

# Local imports
from .cuda_kernel_detrend import detrend_kernel

console = Console()


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

    # Compute spatial averages using pure NumPy (no JIT)
    console.print("[cyan]Computing spatial averages (NumPy)...")
    start_time = time.time()

    # Pure NumPy implementation of spatial averaging
    n_frames, height, width = detrended_stack.shape
    averaged_stack = np.zeros_like(detrended_stack, dtype=np.float32)

    # Adjust height and width to be multiples of roi_size
    height_adjusted = height - (height % roi_size)
    width_adjusted = width - (width % roi_size)

    # Process each frame
    for frame_idx in range(n_frames):
        # Reshape to group pixels into ROIs
        frame_reshaped = detrended_stack[frame_idx, :height_adjusted, :width_adjusted].reshape(
            height_adjusted // roi_size, roi_size, width_adjusted // roi_size, roi_size
        )

        # Calculate mean for each ROI
        roi_means = frame_reshaped.mean(axis=(1, 3))

        # Expand back to original size
        for i in range(height_adjusted // roi_size):
            for j in range(width_adjusted // roi_size):
                y_start = i * roi_size
                y_end = (i + 1) * roi_size
                x_start = j * roi_size
                x_end = (j + 1) * roi_size
                averaged_stack[frame_idx, y_start:y_end, x_start:x_end] = roi_means[i, j]

    console.print(f"Spatial averaging time: {time.time() - start_time:.2f} seconds")

    return detrended_stack, averaged_stack
