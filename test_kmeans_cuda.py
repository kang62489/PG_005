"""
Standalone Numba-accelerated K-means testing script for image segmentation.
Processes frames 60-80 from 2025_06_11-0002_Corr.tif with 5-level clustering.
"""

import time
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from numba import cuda, jit
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from sklearn.cluster import KMeans

console = Console()


@jit(nopython=True, cache=True)
def normalize_pixels_numba(pixels: np.ndarray) -> np.ndarray:
    """Fast pixel normalization using Numba JIT."""
    pixel_min = np.min(pixels)
    pixel_max = np.max(pixels)
    pixel_range = pixel_max - pixel_min

    if pixel_range == 0:
        return np.zeros_like(pixels, dtype=np.float64)
    return (pixels.astype(np.float64) - pixel_min) / pixel_range


@jit(nopython=True, cache=True)
def apply_label_mapping_numba(labels: np.ndarray, label_mapping: np.ndarray) -> np.ndarray:
    """Fast label mapping using Numba JIT."""
    result = np.empty_like(labels, dtype=np.uint8)
    for i in range(labels.size):
        result[i] = label_mapping[labels[i]]
    return result


@cuda.jit
def gpu_normalize_kernel(pixels, normalized, pixel_min, pixel_range):
    """CUDA kernel for pixel normalization."""
    idx = cuda.grid(1)
    if idx < pixels.size:
        if pixel_range == 0:
            normalized[idx] = 0.0
        else:
            normalized[idx] = (pixels[idx] - pixel_min) / pixel_range


def numba_kmeans_sorting(image_stack: np.ndarray, n_levels: int = 5) -> np.ndarray:
    """
    Apply Numba-accelerated K-means clustering to sort pixel values into levels.

    Args:
        image_stack: Input image stack as numpy array
        n_levels: Number of intensity levels (default: 5)

    Returns:
        Sorted labels as uint8 array with values 1-n_levels

    """
    console.print(f"[cyan]Applying Numba-accelerated K-means sorting into {n_levels} levels...")

    # Store original shape
    original_shape = image_stack.shape

    # Flatten the image stack
    pixels = image_stack.flatten()

    # Handle edge cases
    if pixels.min() == pixels.max():
        return np.ones(original_shape, dtype=np.uint8)

    # Use CUDA for normalization if available
    if cuda.is_available():
        console.print("[green]Using CUDA for pixel normalization")

        # Allocate GPU memory
        pixels_gpu = cuda.to_device(pixels.astype(np.float32))
        normalized_gpu = cuda.device_array(pixels.size, dtype=np.float32)

        # Calculate normalization parameters
        pixel_min = float(pixels.min())
        pixel_max = float(pixels.max())
        pixel_range = pixel_max - pixel_min

        # Launch CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (pixels.size + threads_per_block - 1) // threads_per_block
        gpu_normalize_kernel[blocks_per_grid, threads_per_block](pixels_gpu, normalized_gpu, pixel_min, pixel_range)

        # Copy result back to CPU
        pixels_normalized = normalized_gpu.copy_to_host()

    else:
        console.print("[yellow]Using CPU Numba JIT for normalization")
        pixels_normalized = normalize_pixels_numba(pixels)

    # Reshape for K-means
    pixels_normalized = pixels_normalized.reshape(-1, 1)

    # Apply K-means clustering using scikit-learn
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        kmeans = KMeans(n_clusters=n_levels, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(pixels_normalized)

    # Get cluster centers and sort them
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)

    # Create mapping from old labels to new sorted labels (1-n_levels)
    label_mapping = np.zeros(n_levels, dtype=np.uint8)
    for new_label, old_label in enumerate(sorted_indices):
        label_mapping[old_label] = new_label + 1

    # Apply mapping using Numba
    sorted_labels = apply_label_mapping_numba(labels, label_mapping)

    return sorted_labels.reshape(original_shape).astype(np.uint8)


def cpu_only_kmeans(image_stack: np.ndarray, n_levels: int = 5) -> np.ndarray:
    """
    CPU-only K-means clustering using scikit-learn with Numba acceleration for data processing.
    """
    console.print(f"[yellow]Using CPU-only K-means sorting into {n_levels} levels...")

    # Store original shape
    original_shape = image_stack.shape

    # Flatten and normalize using Numba
    pixels = image_stack.flatten()

    if pixels.min() == pixels.max():
        return np.ones(original_shape, dtype=np.uint8)

    # Use Numba for fast normalization
    pixels_normalized = normalize_pixels_numba(pixels)
    pixels_normalized = pixels_normalized.reshape(-1, 1)

    # Apply K-means clustering
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        kmeans = KMeans(n_clusters=n_levels, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(pixels_normalized)

    # Sort labels by cluster centers
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)

    label_mapping = np.zeros(n_levels, dtype=np.uint8)
    for new_label, old_label in enumerate(sorted_indices):
        label_mapping[old_label] = new_label + 1

    # Use Numba for fast label mapping
    sorted_labels = apply_label_mapping_numba(labels, label_mapping)
    return sorted_labels.reshape(original_shape).astype(np.uint8)


def main():
    """Main function to test K-means sorting on selected frames."""
    # Parameters
    input_file = "2025_06_11-0002_Gauss.tif"
    output_file = "test_kmeans_frames_60_80.tif"
    start_frame = 60
    end_frame = 80
    n_levels = 5

    console.print("[bold green]CUDA K-means Testing Script")
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
    console.print("[bold green]Applying K-means clustering...")
    t_kmeans_start = time.time()

    try:
        # Check if CUDA is available through Numba
        if cuda.is_available():
            console.print("[green]Using Numba CUDA acceleration")
            result = numba_kmeans_sorting(img_subset, n_levels)
            method_used = "Numba CUDA"
        else:
            raise RuntimeError("CUDA not available")

    except Exception as e:
        console.print(f"[yellow]CUDA not available ({e}), using CPU with Numba JIT")
        result = cpu_only_kmeans(img_subset, n_levels)
        method_used = "CPU + Numba JIT"

    kmeans_time = time.time() - t_kmeans_start
    console.print(f"K-means ({method_used}) time: {kmeans_time:.2f} seconds")

    # Display results statistics
    console.print("[cyan]Results:")
    console.print(f"Output shape: {result.shape}")
    console.print(f"Output dtype: {result.dtype}")
    console.print(f"Unique levels: {np.unique(result)}")

    # Count pixels per level
    for level in range(1, n_levels + 1):
        count = np.sum(result == level)
        percentage = (count / result.size) * 100
        console.print(f"Level {level}: {count:,} pixels ({percentage:.1f}%)")

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
