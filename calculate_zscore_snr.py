"""
Calculate Z-score and SNR for signal detection using noise baseline.

Z-score = (Signal - Noise_mean) / Noise_std
SNR = (Signal - Noise_mean) / Noise_std  (same formula, different interpretation)

Both methods normalize signal relative to noise statistics for robust detection.
"""

import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()


def calculate_noise_statistics(noise_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate per-pixel mean and standard deviation from noise data.

    Args:
        noise_data: Noise image stack of shape (frames, height, width)

    Returns:
        tuple: (mean_per_pixel, std_per_pixel) both of shape (height, width)

    """
    console.print("[cyan]Calculating noise statistics...")

    # Calculate statistics along temporal axis (axis=0)
    noise_mean = np.mean(noise_data, axis=0, dtype=np.float32)
    noise_std = np.std(noise_data, axis=0, dtype=np.float32)

    # Handle zero std (add small epsilon to avoid division by zero)
    noise_std = np.where(noise_std < 1e-6, 1e-6, noise_std)

    console.print(f"[green]Noise mean range: [{noise_mean.min():.2f}, {noise_mean.max():.2f}]")
    console.print(f"[green]Noise std range: [{noise_std.min():.2f}, {noise_std.max():.2f}]")

    return noise_mean, noise_std


def calculate_relative_zscore_and_delta_f_over_f(
    signal_data: np.ndarray, noise_data: np.ndarray, noise_mean: np.ndarray, noise_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate relative Z-score and ΔF/F for signal detection.

    Args:
        signal_data: Signal image stack of shape (frames, height, width)
        noise_data: Noise image stack of shape (frames, height, width)
        noise_mean: Per-pixel noise mean of shape (height, width)
        noise_std: Per-pixel noise std of shape (height, width)

    Returns:
        tuple: (relative_zscore, delta_f_over_f) both of shape (frames, height, width)

    """
    console.print("[cyan]Calculating relative Z-score and ΔF/F...")

    # Calculate global means for normalization
    signal_mean_global = np.mean(signal_data)
    noise_mean_global = np.mean(noise_data)

    console.print(f"[green]Signal global mean: {signal_mean_global:.2f}")
    console.print(f"[green]Noise global mean: {noise_mean_global:.2f}")

    # Broadcast arrays for calculation
    noise_mean_3d = noise_mean[np.newaxis, :, :]  # Shape: (1, height, width)
    noise_std_3d = noise_std[np.newaxis, :, :]  # Shape: (1, height, width)

    # Method 2: Relative change Z-score (accounts for global intensity differences)
    signal_relative = signal_data / signal_mean_global
    noise_relative = noise_mean_3d / noise_mean_global
    noise_std_relative = noise_std_3d / noise_mean_global

    relative_zscore = (signal_relative - noise_relative) / noise_std_relative

    # Method 4: ΔF/F (delta F over F) - standard in calcium imaging
    # First normalize signal to same global level as noise
    signal_normalized = signal_data * (noise_mean_global / signal_mean_global)
    delta_f_over_f = (signal_normalized - noise_mean_3d) / noise_mean_3d

    console.print(f"[green]Relative Z-score range: [{relative_zscore.min():.2f}, {relative_zscore.max():.2f}]")
    console.print(f"[green]ΔF/F range: [{delta_f_over_f.min():.3f}, {delta_f_over_f.max():.3f}]")

    return relative_zscore.astype(np.float32), delta_f_over_f.astype(np.float32)


def main():
    """Main function to calculate relative Z-score and ΔF/F."""
    # Parameters
    signal_file = "2025_06_11-0002_Corr.tif"
    noise_file = "2025_09_22-0002_Corr.tif"

    relative_zscore_output = f"{Path(signal_file).stem}_RelativeZScore.tif"
    delta_f_over_f_output = f"{Path(signal_file).stem}_DeltaF_over_F.tif"

    console.print("[bold green]Relative Z-score and ΔF/F Calculation")
    console.print(f"Signal file: {signal_file}")
    console.print(f"Noise file: {noise_file}")
    console.print(f"Outputs: {relative_zscore_output}, {delta_f_over_f_output}")

    # Check if input files exist
    if not Path(signal_file).exists():
        console.print(f"[red]Error: Signal file {signal_file} not found!")
        return
    if not Path(noise_file).exists():
        console.print(f"[red]Error: Noise file {noise_file} not found!")
        return

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        # Load files
        task1 = progress.add_task("[cyan]Loading signal file...", total=1)
        t_start = time.time()
        signal_data = imageio.volread(signal_file).astype(np.float32)
        progress.update(task1, advance=1)

        task2 = progress.add_task("[cyan]Loading noise file...", total=1)
        noise_data = imageio.volread(noise_file).astype(np.float32)
        progress.update(task2, advance=1)

    console.print(f"Loading time: {time.time() - t_start:.2f} seconds")
    console.print(f"Signal shape: {signal_data.shape}")
    console.print(f"Noise shape: {noise_data.shape}")

    # Calculate noise statistics
    t_calc_start = time.time()
    noise_mean, noise_std = calculate_noise_statistics(noise_data)

    # Calculate relative Z-score and ΔF/F
    relative_zscore, delta_f_over_f = calculate_relative_zscore_and_delta_f_over_f(
        signal_data, noise_data, noise_mean, noise_std
    )

    console.print(f"Calculation time: {time.time() - t_calc_start:.2f} seconds")

    # Save results
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), console=console) as progress:
        task3 = progress.add_task("[cyan]Saving results...", total=2)

        # Save as float16 to preserve negative values and save space
        imageio.volwrite(relative_zscore_output, relative_zscore.astype(np.float16))
        progress.update(task3, advance=1)

        imageio.volwrite(delta_f_over_f_output, delta_f_over_f.astype(np.float16))
        progress.update(task3, advance=1)

    console.print("[bold green]Results saved:")
    console.print(f"  Relative Z-score: {relative_zscore_output}")
    console.print(f"  ΔF/F: {delta_f_over_f_output}")

    # Display interpretation guide
    console.print("\n[cyan]Interpretation Guide:")
    console.print("Relative Z-score > 2: Signal is 2 std deviations above normalized noise")
    console.print("Relative Z-score > 3: Strong signal (99.7% confidence)")
    console.print("ΔF/F > 0.1 (10%): Noticeable signal increase")
    console.print("ΔF/F > 0.2 (20%): Strong signal increase")
    console.print("ΔF/F < 0: Signal decrease relative to baseline")

    total_time = time.time() - t_start
    console.print(f"\n[bold green]Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
