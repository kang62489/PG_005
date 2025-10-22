# Standard library imports
import logging
import os
import time
from pathlib import Path

# Third-party imports
import imageio.v2 as imageio
import numpy as np
from numba import cuda  # noqa: F401
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

# Local application imports
from functions import (  # noqa: F401
    check_cuda,
    get_memory_usage,
    process_on_cpu,
    process_on_gpu,
    test_cuda,
)

# Force Numba to recompile
os.environ["NUMBA_DISABLE_FUNCTION_CACHING"] = "1"

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")


def main() -> None:
    # Parameters
    filename = "2025_10_13-0035.tif"
    raw_path = Path(__file__).parent / "raw_images"
    output_name_1 = f"{Path(filename).stem}_Cal.tif"
    output_name_2 = f"{Path(filename).stem}_Gauss.tif"

    # Delete existing output files
    for output_file in [output_name_1, output_name_2]:
        Path(output_file).unlink(missing_ok=True)

    # Set a progressbar to show image loading progress
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Load image stack
        task1 = progress.add_task("[cyan]Loading image stack...", total=1)
        t_start = time.time()
        img_raw = imageio.volread(raw_path / filename).astype(np.uint16)
        progress.update(task1, advance=1)

    log.info("Loading time: %s seconds", f"{time.time() - t_start:.2f}")
    log.info("Image shape: %s", f"{img_raw.shape}")
    log.info("Memory usage: %s GB", f"{get_memory_usage():.2f}")

    # Process on either GPU or CPU
    console.print("[bold green]Processing data...")
    t_start = time.time()

    if cuda.is_available():
        detrended, gaussian = process_on_gpu(img_raw)
    else:
        detrended, averaged, gaussian = process_on_cpu(img_raw)
    # detrended, gaussian = process_on_cpu(img_raw)
    console.print(f"Total processing time: {time.time() - t_start:.2f} seconds")

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Save results
        task2 = progress.add_task("[cyan]Saving results...", total=1)
        detrended_uint16 = np.clip(detrended, 0, 65535).astype(np.uint16)
        gaussian_uint16 = np.clip(gaussian, 0, 65535).astype(np.uint16)

        imageio.volwrite(output_name_1, detrended_uint16)
        imageio.volwrite(output_name_2, gaussian_uint16)

        progress.update(task2, advance=1)

    log.info("Results %s and %s saved!", output_name_1, output_name_2)
    console.print("[bold green]Processing completed successfully!")


# Check CUDA availability with diagnostics
cuda_available = check_cuda()

if cuda_available:
    # Verify GPU functionality
    cuda_available = test_cuda()

if not cuda_available:
    console.print("[yellow]Falling back to CPU processing...")

main()
