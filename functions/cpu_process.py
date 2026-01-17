# Standard library imports
import logging
import time

# Third-party imports
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# Local application imports
from functions import cpu_detrend_jitted, cpu_gaussian_blur

# Setup rich console and logging
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("rich")


def process_on_cpu(
    image_stack: np.ndarray, window_size: int = 101, sigma: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process image stack entirely on CPU using JIT.

    Args:
        image_stack: Input array of shape (n_frames, height, width)
        roi_size: Size of Region of Interest (ROI)
        window_size: Size of moving average window for detrending
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Tuple of (detrended_stack, averaged_stack, gaussian_stack)

    """
    n_frames, height, width = image_stack.shape
    pixels_time_series = image_stack.reshape(n_frames, -1).T

    # Warm up Numba functions
    log.info("Initializing Numba functions...")
    rng = np.random.default_rng()
    test_data = rng.random((10, 100), dtype=np.float32)
    _ = cpu_detrend_jitted(test_data, 10)
    test_stack = rng.random((2, 8, 8), dtype=np.float32)
    _ = cpu_gaussian_blur(test_stack, 4)
    log.info("Initialization complete!")

    # Detrend pixels
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        task1 = progress.add_task("[cyan]Detrending pixels...", total=1)
        t_start = time.time()
        detrended_pixels = cpu_detrend_jitted(pixels_time_series.astype(np.float32), window_size)
        detrended_stack = detrended_pixels.T.reshape(n_frames, height, width)

        all_pixels_averages = np.mean(detrended_stack, axis=0)
        align_pixels_means_to_min = all_pixels_averages - np.min(all_pixels_averages)

        detrended_stack -= align_pixels_means_to_min
        progress.update(task1, advance=1)
    log.info("Detrending time: %s seconds", f"{time.time() - t_start:.2f}")

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        # Process spatial averaging
        task2 = progress.add_task("[cyan]Processing Gaussian blur...", total=1)
        t_start = time.time()
        gaussian_stack = cpu_gaussian_blur(detrended_stack, sigma)
        progress.update(task2, advance=1)
    log.info("Processing time:  %s seconds", f"{time.time() - t_start:.2f}")

    return detrended_stack, gaussian_stack
