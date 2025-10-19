# Third-party imports
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def cpu_detrend_jitted(pixel_data: np.ndarray, window_size: int) -> np.ndarray:
    """Detrend pixels by removing moving average in parallel."""
    n_pixels, n_frames = pixel_data.shape
    detrended = np.zeros_like(pixel_data, dtype=np.float32)

    for pixel_idx in prange(n_pixels):
        moving_avgs = np.zeros(n_frames, dtype=np.float32)
        for frame_idx in prange(n_frames):
            window_start = max(0, frame_idx - window_size // 2)
            window_end = min(n_frames, frame_idx + window_size // 2)
            moving_avgs[frame_idx] = np.mean(pixel_data[pixel_idx, window_start:window_end])

        # Find minimum edge value
        edge_min = min(moving_avgs[0], moving_avgs[-1])

        for frame_idx in prange(n_frames):
            trend = moving_avgs[frame_idx] - edge_min
            # trend = moving_avgs[frame_idx]
            detrended[pixel_idx, frame_idx] = pixel_data[pixel_idx, frame_idx] - trend

    return detrended
