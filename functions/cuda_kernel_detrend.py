# Third-party imports
import numpy as np
from numba import cuda


@cuda.jit
def detrend_kernel(pixel_data: np.ndarray, output: np.ndarray, window_size: int) -> None:
    """
    CUDA kernel for detrending pixels by removing trend component of moving average.

    Args:
        pixel_data: Input array of shape (n_pixels, n_frames)
        output: Output array of same shape as input
        window_size: Size of moving average window

    """
    pixel_idx = cuda.grid(1)
    if pixel_idx >= pixel_data.shape[0]:
        return

    n_frames = pixel_data.shape[1]
    half_window = window_size // 2

    # Local array to store moving averages
    # use 2048 incase the number of frames is larger than 1200
    moving_averages = cuda.local.array(2048, dtype=np.float32)

    for frame_idx in range(n_frames):
        # Define window boundaries
        window_start = max(0, frame_idx - half_window)
        window_end = min(n_frames, frame_idx + half_window + 1)

        # Calculate moving average
        window_sum = 0.0
        window_size_actual = window_end - window_start
        for k in range(window_start, window_end):
            window_sum += pixel_data[pixel_idx, k]
        moving_averages[frame_idx] = window_sum / window_size_actual

    # Find minimum edge value
    # use n_frames-1 instead of -1 becuase the local array is 2048 long
    edge_min = min(moving_averages[0], moving_averages[n_frames - 1])

    for frame_idx in range(n_frames):
        trend = moving_averages[frame_idx] - edge_min
        output[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - trend
