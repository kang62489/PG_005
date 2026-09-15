"""fit_bg_hist.py -- stack-wide background-noise z-score normalization (CPU Numba JIT + CUDA GPU)."""

## Modules
# Standard library imports
import math

# Third-party imports
import numba
import numpy as np
from numba import cuda, njit, prange
from scipy.optimize import curve_fit

# Constants
N_HIST_BINS = 1000


def _gaussian(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


@njit(parallel=True, cache=True)
def _cpu_histogram_counts(values: np.ndarray, n_bins: int, lo: float, bin_width: float) -> np.ndarray:
    n = values.shape[0]
    n_threads = numba.get_num_threads()
    local_counts = np.zeros((n_threads, n_bins), dtype=np.int64)  # per-thread bins, avoids write races

    for i in prange(n):
        tid = numba.get_thread_id()
        idx = int((values[i] - lo) / bin_width)
        # clamp: float rounding can push the max value 1 past the last bin
        if idx < 0:
            idx = 0
        elif idx >= n_bins:
            idx = n_bins - 1
        local_counts[tid, idx] += 1

    counts = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        counts += local_counts[t]  # combine per-thread bins
    return counts


@njit(parallel=True, cache=True)
def _cpu_masked_std(values: np.ndarray, threshold: float) -> float:
    # std of values <= threshold, single pass (sum/sum_sq/count), no boolean-mask copy
    n = values.shape[0]
    n_threads = numba.get_num_threads()
    local_sum = np.zeros(n_threads, dtype=np.float64)
    local_sum_sq = np.zeros(n_threads, dtype=np.float64)
    local_count = np.zeros(n_threads, dtype=np.int64)

    for i in prange(n):
        tid = numba.get_thread_id()
        v = values[i]
        if v <= threshold:
            local_sum[tid] += v
            local_sum_sq[tid] += v * v
            local_count[tid] += 1

    total_sum = local_sum.sum()
    total_sum_sq = local_sum_sq.sum()
    total_count = local_count.sum()

    mean = total_sum / total_count
    variance = total_sum_sq / total_count - mean * mean
    return np.sqrt(variance) if variance > 0.0 else 0.0


@cuda.jit
def _gpu_histogram_kernel(
    values: np.ndarray, counts: np.ndarray, n_bins: int, lo: float, bin_width: float
) -> None:
    # one thread per value, atomic-add into the shared bin counter (GPU equivalent of the
    # CPU version's per-thread-bins-then-combine trick)
    i = cuda.grid(1)
    if i >= values.shape[0]:
        return
    idx = int((values[i] - lo) / bin_width)
    if idx < 0:
        idx = 0
    elif idx >= n_bins:
        idx = n_bins - 1
    cuda.atomic.add(counts, idx, 1)


def _gpu_histogram_counts(values: np.ndarray, n_bins: int, lo: float, bin_width: float) -> np.ndarray:
    n = values.shape[0]
    d_values = cuda.to_device(values.astype(np.float32))
    d_counts = cuda.to_device(np.zeros(n_bins, dtype=np.int64))
    threads = 256
    blocks = math.ceil(n / threads)
    _gpu_histogram_kernel[blocks, threads](d_values, d_counts, n_bins, np.float32(lo), np.float32(bin_width))
    cuda.synchronize()
    return d_counts.copy_to_host()


@cuda.jit
def _gpu_masked_sum_kernel(values: np.ndarray, threshold: float, sums: np.ndarray) -> None:
    # sums = [sum, sum_sq, count] accumulator, one thread per value, atomic-add
    i = cuda.grid(1)
    if i >= values.shape[0]:
        return
    v = values[i]
    if v <= threshold:
        cuda.atomic.add(sums, 0, v)
        cuda.atomic.add(sums, 1, v * v)
        cuda.atomic.add(sums, 2, 1.0)


def _gpu_masked_std(values: np.ndarray, threshold: float) -> float:
    n = values.shape[0]
    d_values = cuda.to_device(values.astype(np.float64))
    d_sums = cuda.to_device(np.zeros(3, dtype=np.float64))
    threads = 256
    blocks = math.ceil(n / threads)
    _gpu_masked_sum_kernel[blocks, threads](d_values, np.float64(threshold), d_sums)
    cuda.synchronize()
    total_sum, total_sum_sq, total_count = d_sums.copy_to_host()

    mean = total_sum / total_count
    variance = total_sum_sq / total_count - mean * mean
    return float(np.sqrt(variance)) if variance > 0.0 else 0.0


def fit_hist_sigma(
    detrended: np.ndarray, n_bins: int = N_HIST_BINS, cuda_available: bool = False
) -> tuple[float, float]:
    # Pool every pixel/frame value into one histogram, find the peak (background mode), fit a
    # Gaussian to the peak + left side only -- right side is contaminated by real signal
    # transients (hotspots), so fitting only the clean half keeps sigma robust.
    values = detrended.ravel()
    lo = float(values.min())
    hi = float(values.max())
    bin_width = (hi - lo) / n_bins

    # histogram build + masked-std seed scale with stack size -> GPU when available.
    # The final curve_fit below always runs on CPU: it only ever fits n_bins points, already
    # trivially fast regardless of stack size.
    if cuda_available:
        counts = _gpu_histogram_counts(values, n_bins, lo, bin_width)
    else:
        counts = _cpu_histogram_counts(values, n_bins, lo, bin_width)
    edges = lo + bin_width * np.arange(n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    peak_idx = int(np.argmax(counts))
    x_peak = float(centers[peak_idx])

    left_mask = centers <= x_peak
    x_left = centers[left_mask]
    y_left = counts[left_mask]

    sigma_seed = _gpu_masked_std(values, x_peak) if cuda_available else _cpu_masked_std(values, x_peak)
    p0 = [float(counts[peak_idx]), x_peak, float(sigma_seed)]
    popt, _ = curve_fit(_gaussian, x_left, y_left, p0=p0, maxfev=5000)
    _amp_fit, mean_fit, sigma_fit = popt
    return float(mean_fit), float(abs(sigma_fit))


def img_zscore_convert(detrended: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return (detrended - mean) / sigma
