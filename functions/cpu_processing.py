# Standard library imports
import logging
import time

# Third-party imports
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .cpu_parallel_detrend import detrend_parallel

# Local imports
from .spatial_processing import compute_spatial_averages

# Setup rich console and logging
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("rich")


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
    n_frames, height, width = image_stack.shape
    pixels_time_series = image_stack.reshape(n_frames, -1).T

    # Warm up Numba functions
    log.info("Initializing Numba functions...")
    rng = np.random.default_rng()
    test_data = rng.random((10, 100), dtype=np.float32)
    _ = detrend_parallel(test_data, 10)
    test_frame = rng.random((128, 128), dtype=np.float32)
    _ = compute_spatial_averages(test_frame, 32)
    log.info("Initialization complete!")

    # Detrend pixels
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        task2 = progress.add_task("[cyan]Detrending pixels...", total=1)
        t_start = time.time()
        detrended_pixels = detrend_parallel(pixels_time_series.astype(np.float32), window_size)
        detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)

        pixel_offsets = np.mean(detrended_stack, axis=0)
        pixel_offsets_adjust = pixel_offsets - np.min(pixel_offsets)
        detrended_stack -= pixel_offsets_adjust
        progress.update(task2, advance=1)
        log.info("Detrending time: %s seconds", str(time.time() - t_start))

        # Process spatial averaging
        task3 = progress.add_task("[cyan]Processing spatial averaging...", total=1)
        t_start = time.time()
        averaged_stack = compute_spatial_averages(detrended_stack, roi_size)
        progress.update(task3, advance=1)
        log.info("Processing time: %s seconds", str(time.time() - t_start))

    return detrended_stack, averaged_stack
