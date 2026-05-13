"""
im_preprocess.py  --  Moving-average detrend + Gaussian blur (GPU/CPU)
=======================================================================
Pipeline per file:
  1. Delete any existing output TIFFs
  2. Load raw image stack
  3. Detrend (moving-average) + Gaussian blur  →  GPU if available, else CPU
  4. Save <stem>_MOV_CAL.tif and <stem>_MOV_GAUSS.tif
"""

# ── Imports ──────────────────────────────────────────────────────────
# Standard library imports
import os
import time
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile
from numba import cuda
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Local application imports
from functions import check_cuda, get_memory_usage, process_on_cpu, process_on_gpu, test_cuda

# ── Config ───────────────────────────────────────────────────────────
# Force Numba to recompile
os.environ["NUMBA_DISABLE_FUNCTION_CACHING"] = "1"

RAW_DIR = Path(__file__).parent / "raw_tiffs"
OUT_DIR = Path(__file__).parent / "proc_tiffs"

# ── Setup ────────────────────────────────────────────────────────────
console = Console()


# ── Main function ────────────────────────────────────────────────────

def main(file: str, cuda_available: bool) -> None:
    # Step 1 -- parameters
    filename = file
    output_name_1 = f"{Path(filename).stem}_MOV_CAL.tif"
    output_name_2 = f"{Path(filename).stem}_MOV_GAUSS.tif"

    # Step 2 -- delete existing output files
    for output_file in [output_name_1, output_name_2]:
        Path(output_file).unlink(missing_ok=True)

    # Step 3 -- load image stack
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"), TimeElapsedColumn(), console=console
    ) as progress:
        task1 = progress.add_task("Loading image stack...", total=None)
        t_start = time.time()
        img_raw = tifffile.imread(RAW_DIR / filename).astype(np.uint16)
        progress.remove_task(task1)

    console.log(f"Loading time: {time.time() - t_start:.2f} seconds")
    console.log(f"Image shape: {img_raw.shape}")
    console.log(f"Memory usage: {get_memory_usage():.2f} GB")

    # Step 4 -- process (GPU with CPU fallback)
    console.log("[bold green]Processing data...")
    t_start = time.time()

    try:
        if cuda_available:
            detrended, gaussian = process_on_gpu(img_raw)
        else:
            detrended, gaussian = process_on_cpu(img_raw)
    except (cuda.cudadrv.driver.CudaAPIError, RuntimeError, Exception) as e:
        console.log(f"[yellow]GPU processing failed: {e}[/yellow]")
        console.log("[yellow]Falling back to CPU processing...[/yellow]")
        detrended, gaussian = process_on_cpu(img_raw)

    # detrended, gaussian = process_on_cpu(img_raw)
    console.log(f"Total processing time: {time.time() - t_start:.2f} seconds")

    # Step 5 -- save results
    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"), TimeElapsedColumn(), console=console
    ) as progress:
        task2 = progress.add_task("Saving results...", total=None)
        t_start = time.time()
        detrended_uint16 = np.clip(detrended, 0, 65535).astype(np.uint16)
        gaussian_uint16 = np.clip(gaussian, 0, 65535).astype(np.uint16)

        tifffile.imwrite(OUT_DIR / output_name_1, detrended_uint16)
        tifffile.imwrite(OUT_DIR / output_name_2, gaussian_uint16)
        progress.remove_task(task2)

    console.log(f"Saving time: {time.time() - t_start:.2f} seconds")
    console.log(f"Results {output_name_1} and {output_name_2} saved!")
    console.log("[bold green]Processing completed successfully!")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check CUDA availability
    cuda_available = check_cuda()
    test_cuda()

    # Process files
    date = "2026_03_20"
    serials = ["0040"]
    file_list = [f"{date}-{serial}.tif" for serial in serials]
    for file in file_list:
        main(file, cuda_available)
