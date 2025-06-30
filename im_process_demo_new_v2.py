# Standard library imports
import os
import time
from pathlib import Path

# Third-party imports
import imageio.v2 as imageio
import numpy as np
from numba import cuda
from rich.console import Console

# Local imports
from functions import check_cuda, process_on_cpu, process_on_gpu, test_cuda

# Force Numba to recompile
os.environ["NUMBA_DISABLE_FUNCTION_CACHING"] = "1"
console = Console()


def main() -> None:
    # Parameters
    filepath = "2025_06_11-0012.tif"
    output_name_1 = f"Corr_{filepath}"
    output_name_2 = f"Conv_{filepath}"
    roi_size = 16

    # Delete existing output files
    for output_file in [output_name_1, output_name_2]:
        Path(output_file).unlink(missing_ok=True)

    try:
        # Load image stack
        console.print("[bold green]Loading image stack...")
        t_start = time.time()
        img_raw = imageio.volread(f"raw_images/{filepath}").astype(np.float32)
        console.print(f"Loading time: {time.time() - t_start:.2f} seconds")
        console.print(f"Image shape: {img_raw.shape}")

        # Process on either GPU or CPU
        console.print("[bold green]Processing data...")
        t_start = time.time()
        if cuda.is_available():
            detrended, averaged = process_on_gpu(img_raw, roi_size)
        else:
            detrended, averaged = process_on_cpu(img_raw, roi_size)
        console.print(f"Total processing time: {time.time() - t_start:.2f} seconds")

        # Save results
        console.print("[bold green]Saving results...")
        detrended_uint16 = np.clip(detrended, 0, 65535).astype(np.uint16)
        averaged_uint16 = np.clip(averaged, 0, 65535).astype(np.uint16)

        imageio.volwrite(output_name_1, detrended_uint16)
        imageio.volwrite(output_name_2, averaged_uint16)

        console.print("[bold green]Processing completed successfully!")

    except Exception as e:
        console.print(f"[bold red]Error: {e!s}")
        raise


if __name__ == "__main__":
    # Check CUDA availability with diagnostics
    cuda_available = check_cuda()

    if cuda_available:
        # Verify GPU functionality
        cuda_available = test_cuda()

    if not cuda_available:
        console.print("[yellow]Falling back to CPU processing...")

    main()
