"""
Standalone Numba-accelerated statistical sorting script based on mu and sigma.
Processes frames 60-80 from 2025_06_11-0002_Corr.tif with 5-level statistical binning.
"""

import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from numba import cuda, jit
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()


@jit(nopython=True, cache=True)
def calculate_stats_numba(pixels: np.ndarray):
    """Fast calculation of mean and standard deviation using Numba JIT."""
    # Input pixels should already be float32 or float64
    mean = np.mean(pixels)
    std = np.std(pixels)
    return mean, std


@jit(nopython=True, cache=True)
def statistical_binning_numba(pixels: np.ndarray, mean: float, std: float, n_levels: int = 5) -> np.ndarray:
    """
    Fast statistical binning using mean and standard deviation with Numba JIT.

    Creates bins based on statistical ranges:
    Level 1: < mean - 2*std
    Level 2: mean - 2*std to mean - std
    Level 3: mean - std to mean + std
    Level 4: mean + std to mean + 2*std
    Level 5: > mean + 2*std
    """
    result = np.empty_like(pixels, dtype=np.uint8)

    # Define thresholds based on statistical ranges
    thresh1 = mean - 2.0 * std
    thresh2 = mean - std
    thresh3 = mean + std
    thresh4 = mean + 2.0 * std

    for i in range(pixels.size):
        pixel = float(pixels.flat[i])

        if pixel < thresh1:
            result.flat[i] = 1
        elif pixel < thresh2:
            result.flat[i] = 2
        elif pixel < thresh3:
            result.flat[i] = 3
        elif pixel < thresh4:
            result.flat[i] = 4
        else:
            result.flat[i] = 5

    return result


@cuda.jit
def gpu_statistical_binning_kernel(pixels, result, mean, std):
    """CUDA kernel for statistical binning based on mean and standard deviation."""
    idx = cuda.grid(1)
    if idx < pixels.size:
        pixel = float(pixels[idx])

        # Define thresholds
        thresh1 = mean - 2.0 * std
        thresh2 = mean - std
        thresh3 = mean + std
        thresh4 = mean + 2.0 * std

        if pixel < thresh1:
            result[idx] = 1
        elif pixel < thresh2:
            result[idx] = 2
        elif pixel < thresh3:
            result[idx] = 3
        elif pixel < thresh4:
            result[idx] = 4
        else:
            result[idx] = 5


def numba_statistical_sorting(image_stack: np.ndarray, n_levels: int = 5) -> np.ndarray:
    """
    Apply Numba-accelerated statistical sorting to pixel values into levels.

    Args:
        image_stack: Input image stack as numpy array
        n_levels: Number of intensity levels (default: 5)

    Returns:
        Sorted labels as uint8 array with values 1-n_levels

    """
    console.print(f"[cyan]Applying Numba-accelerated statistical sorting into {n_levels} levels...")

    # Store original shape
    original_shape = image_stack.shape

    # Flatten the image stack and convert to float32 (Numba compatible)
    pixels = image_stack.flatten().astype(np.float32)

    # Handle edge cases
    if pixels.min() == pixels.max():
        return np.full(original_shape, 3, dtype=np.uint8)  # All pixels in middle level

    # Calculate statistics using Numba
    mean, std = calculate_stats_numba(pixels)
    console.print(f"[green]Statistics - Mean: {mean:.2f}, Std: {std:.2f}")

    # Handle case where std is very small
    if std < 1e-6:
        return np.full(original_shape, 3, dtype=np.uint8)

    # Use CUDA for binning if available
    if cuda.is_available():
        console.print("[green]Using CUDA for statistical binning")

        # Allocate GPU memory
        pixels_gpu = cuda.to_device(pixels.astype(np.float32))
        result_gpu = cuda.device_array(pixels.size, dtype=np.uint8)

        # Launch CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (pixels.size + threads_per_block - 1) // threads_per_block
        gpu_statistical_binning_kernel[blocks_per_grid, threads_per_block](
            pixels_gpu, result_gpu, float(mean), float(std)
        )

        # Copy result back to CPU
        result = result_gpu.copy_to_host()

    else:
        console.print("[yellow]Using CPU Numba JIT for statistical binning")
        result = statistical_binning_numba(pixels, mean, std, n_levels)

    return result.reshape(original_shape)


def cpu_only_statistical_sorting(image_stack: np.ndarray, n_levels: int = 5) -> np.ndarray:
    """
    CPU-only statistical sorting using Numba acceleration for data processing.
    """
    console.print(f"[yellow]Using CPU-only statistical sorting into {n_levels} levels...")

    # Store original shape
    original_shape = image_stack.shape

    # Flatten the image and convert to float32 (Numba compatible)
    pixels = image_stack.flatten().astype(np.float32)

    if pixels.min() == pixels.max():
        return np.full(original_shape, 3, dtype=np.uint8)

    # Calculate statistics using Numba
    mean, std = calculate_stats_numba(pixels)
    console.print(f"[green]Statistics - Mean: {mean:.2f}, Std: {std:.2f}")

    if std < 1e-6:
        return np.full(original_shape, 3, dtype=np.uint8)

    # Apply statistical binning using Numba
    result = statistical_binning_numba(pixels, mean, std, n_levels)
    return result.reshape(original_shape)


def display_statistical_info(image_stack: np.ndarray, result: np.ndarray, n_levels: int):
    """Display detailed statistical information about the sorting results."""
    pixels = image_stack.flatten()
    mean = np.mean(pixels)
    std = np.std(pixels)

    console.print("[cyan]Statistical Analysis:")
    console.print(f"Mean (μ): {mean:.2f}")
    console.print(f"Standard deviation (σ): {std:.2f}")
    console.print("")
    console.print("[cyan]Threshold Ranges:")
    console.print(f"Level 1: < {mean - 2 * std:.2f} (μ - 2σ)")
    console.print(f"Level 2: {mean - 2 * std:.2f} to {mean - std:.2f} (μ - 2σ to μ - σ)")
    console.print(f"Level 3: {mean - std:.2f} to {mean + std:.2f} (μ - σ to μ + σ)")
    console.print(f"Level 4: {mean + std:.2f} to {mean + 2 * std:.2f} (μ + σ to μ + 2σ)")
    console.print(f"Level 5: > {mean + 2 * std:.2f} (μ + 2σ)")
    console.print("")

    # Count pixels per level
    for level in range(1, n_levels + 1):
        count = np.sum(result == level)
        percentage = (count / result.size) * 100
        console.print(f"Level {level}: {count:,} pixels ({percentage:.1f}%)")


def main():
    """Main function to test statistical sorting on selected frames."""
    # Parameters
    input_file = "2025_06_11-0012_Gauss.tif"
    output_file = "test_statistical_frames_60_80.tif"
    start_frame = 60
    end_frame = 80
    n_levels = 5

    console.print("[bold green]Statistical Sorting Testing Script (μ ± σ)")
    console.print(f"Input: {input_file}")
    console.print(f"Frames: {start_frame}-{end_frame}")
    console.print(f"Levels: {n_levels}")

    # Check if input file exists
    if not Path(input_file).exists():
        console.print(f"[red]Error: Input file {input_file} not found!")
        return

    # Load image stack
    console.print("[cyan]Loading image stack...")
    t_load_start = time.time()

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Loading...", total=1)

        # Load full stack and extract frames
        img_full = imageio.volread(input_file)
        img_subset = img_full[start_frame : end_frame + 1]  # +1 because end is exclusive

        progress.update(task, advance=1)

    console.print(f"Loading time: {time.time() - t_load_start:.2f} seconds")
    console.print(f"Subset shape: {img_subset.shape}")
    console.print(f"Data type: {img_subset.dtype}")
    console.print(f"Value range: [{img_subset.min()}, {img_subset.max()}]")

    # Try Numba CUDA acceleration first, fallback to CPU if needed
    console.print("[bold green]Applying statistical sorting...")
    t_sort_start = time.time()

    try:
        # Check if CUDA is available through Numba
        if cuda.is_available():
            console.print("[green]Using Numba CUDA acceleration")
            result = numba_statistical_sorting(img_subset, n_levels)
            method_used = "Numba CUDA"
        else:
            raise RuntimeError("CUDA not available")

    except Exception as e:
        console.print(f"[yellow]CUDA not available ({e}), using CPU with Numba JIT")
        result = cpu_only_statistical_sorting(img_subset, n_levels)
        method_used = "CPU + Numba JIT"

    sort_time = time.time() - t_sort_start
    console.print(f"Statistical sorting ({method_used}) time: {sort_time:.2f} seconds")

    # Display results statistics
    console.print("[cyan]Results:")
    console.print(f"Output shape: {result.shape}")
    console.print(f"Output dtype: {result.dtype}")
    console.print(f"Unique levels: {np.unique(result)}")

    # Display detailed statistical analysis
    display_statistical_info(img_subset, result, n_levels)

    # Save results
    console.print("[cyan]Saving results...")
    t_save_start = time.time()

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Saving...", total=1)
        imageio.volwrite(output_file, result)
        progress.update(task, advance=1)

    console.print(f"Save time: {time.time() - t_save_start:.2f} seconds")
    console.print(f"[bold green]Results saved to: {output_file}")

    # Total time
    total_time = time.time() - t_load_start
    console.print(f"[bold green]Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
