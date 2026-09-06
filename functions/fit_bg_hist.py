"""
fit_bg_hist.py  --  Stack-wide background-noise z-score normalization (CPU, Numba JIT).

Public API
----------
fit_hist_sigma(detrended, n_bins)         ->  tuple[float, float]
img_zscore_convert(detrended, mean, sigma) ->  np.ndarray
"""

## Modules
# Third-party imports
import numba
import numpy as np
from numba import njit, prange
from scipy.optimize import curve_fit

# Constants
N_HIST_BINS = 1000


def _gaussian(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


@njit(parallel=True)
def _cpu_histogram_counts(values: np.ndarray, n_bins: int, lo: float, bin_width: float) -> np.ndarray:
    """Parallel histogram: per-thread local bins combined at the end (avoids write races)."""
    n = values.shape[0]
    n_threads = numba.get_num_threads()
    local_counts = np.zeros((n_threads, n_bins), dtype=np.int64)

    for i in prange(n):
        tid = numba.get_thread_id()
        idx = int((values[i] - lo) / bin_width)
        if idx < 0:
            idx = 0
        elif idx >= n_bins:
            idx = n_bins - 1
        local_counts[tid, idx] += 1

    counts = np.zeros(n_bins, dtype=np.int64)
    for t in range(n_threads):
        counts += local_counts[t]
    return counts


@njit(parallel=True)
def _cpu_masked_std(values: np.ndarray, threshold: float) -> float:
    """Std of values <= threshold, single pass, no boolean-mask copy."""
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


def fit_hist_sigma(detrended: np.ndarray, n_bins: int = N_HIST_BINS) -> tuple[float, float]:
    """
    Estimate the background-noise center/sigma from a stack-wide histogram.

    Pools every pixel/frame value in `detrended` (bi-exp detrend residual, y - trend)
    into one histogram, finds the peak (background mode), then fits a Gaussian to
    the peak and its left side only — the right side is contaminated by real signal
    transients (hotspots), so fitting only the uncontaminated half keeps the sigma
    estimate robust.

    Args:
        detrended: Detrend residual stack, any shape.
        n_bins: Number of histogram bins spanning the full data range (min to max).

    Returns:
        (mean, sigma) of the fitted background-noise Gaussian.
    """
    values = detrended.ravel()
    lo = float(values.min())
    hi = float(values.max())
    bin_width = (hi - lo) / n_bins

    counts = _cpu_histogram_counts(values, n_bins, lo, bin_width)
    edges = lo + bin_width * np.arange(n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    peak_idx = int(np.argmax(counts))
    x_peak = float(centers[peak_idx])

    left_mask = centers <= x_peak
    x_left = centers[left_mask]
    y_left = counts[left_mask]

    sigma_seed = _cpu_masked_std(values, x_peak)
    p0 = [float(counts[peak_idx]), x_peak, float(sigma_seed)]
    popt, _ = curve_fit(_gaussian, x_left, y_left, p0=p0, maxfev=5000)
    _amp_fit, mean_fit, sigma_fit = popt
    return float(mean_fit), float(abs(sigma_fit))


def img_zscore_convert(detrended: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Convert a detrend residual stack to z-scores using a stack-wide background mean/sigma."""
    return (detrended - mean) / sigma
