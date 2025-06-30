# Standard library imports
import time

# Third-party imports
import numpy as np
from numba import jit, prange
from rich.console import Console

# Local imports
from .spatial_processing import compute_spatial_averages

console = Console()


def process_on_cpu(image_stack: np.ndarray, roi_size: int, window_size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Process image stack entirely on CPU using JIT.

    Args:
        image_stack: Input array of shape (n_frames, height, width)
        roi_size: Size of Region of Interest (ROI)
        window_size: Size of moving average window for detrending

    Returns:
        Tuple of (detrended_stack, averaged_stack)

    """

    @jit(nopython=True, parallel=True)
    def detrend_parallel(pixel_data: np.ndarray, window_size: int) -> np.ndarray:
        """Detrend pixels by removing moving average in parallel."""
        n_pixels, n_frames = pixel_data.shape
        detrended = np.zeros_like(pixel_data, dtype=np.float32)

        for pixel_idx in prange(n_pixels):
            for frame_idx in range(n_frames):
                window_start = max(0, frame_idx - window_size // 2)
                window_end = min(n_frames, frame_idx + window_size // 2)
                moving_avg = np.mean(pixel_data[pixel_idx, window_start:window_end])
                detrended[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - moving_avg

        return detrended

    n_frames, height, width = image_stack.shape
    pixels_time_series = image_stack.reshape(n_frames, -1).T

    # Perform detrending
    console.print("[cyan]Detrending pixels on CPU...")
    start_time = time.time()
    detrended_pixels = detrend_parallel(pixels_time_series.astype(np.float32), window_size)
    detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)
    detrended_stack -= np.min(detrended_stack)
    console.print(f"Detrending time: {time.time() - start_time:.2f} seconds")

    # Compute spatial averages
    console.print("[cyan]Computing spatial averages on CPU...")
    start_time = time.time()
    averaged_stack = compute_spatial_averages(detrended_stack, roi_size)
    console.print(f"Spatial averaging time: {time.time() - start_time:.2f} seconds")

    return detrended_stack, averaged_stack
