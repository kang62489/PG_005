"""
fit_hist.py  --  Two left-side Gaussian fits of a pixel-value histogram (CPU Numba JIT + CUDA GPU).

  Fit 1. Z-score : detrended raw stack -> histogram over min..max (1000 bins) -> Gaussian on the peak + left side
                   (mean free) -> (mean, sigma); img_proc z-scores the stack with them
  Fit 2. Flash   : preprocessed *_BIEXP_ALS.tif -> histogram over the 0.1-99.9 percentiles (512 bins) -> smoothed peak
                   -> Gaussian on the left side (mean pinned at the peak) -> (center, sigma);
                   spontaneous analysis + permutation test: flash threshold = center + 2 sigma
  Shared         : histogram counts (GPU when available), the left-side Gaussian fit, (x - mean) / sigma

The right side of the histogram holds the flashes, so both fits use only the peak + left side.
"""

## Modules
# Standard library imports
import math

# Third-party imports
import numba
import numpy as np
from numba import cuda, njit, prange
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Fit 1: z-score --------------------------------------------------------
ZSCORE_HIST_BINS = 1000             # histogram bins over min..max

# --- Fit 2: flash ----------------------------------------------------------
FLASH_HIST_BINS = 512               # histogram bins over the percentile range
FLASH_HIST_RANGE_PCT = (0.1, 99.9)  # histogram spans these percentiles, so rare outliers can't widen the bins
FLASH_SMOOTH_FRAC = 0.05            # Savitzky-Golay window = this fraction of the histogram range
FLASH_SIGMA_SEED = 0.0016           # initial sigma guess for the fixed-mean fit
CROSSOVER_RATIO = 2.0               # flash threshold = background peak + this many fitted sigmas


# ===========================================================================
#
#   SHARED -- histogram, left-side Gaussian fit, z-score
#
# ===========================================================================

# --- Sa. Histogram: CPU ----------------------------------------------------

@njit(parallel=True)
def _cpu_histogram_counts(values: np.ndarray, n_bins: int, lo: float, hi: float, bin_width: float,
                          drop_outside: bool) -> np.ndarray:
    """Bin counts over [lo, hi] on the CPU (parallel); outside values clamped to the edge bins, or dropped."""
    n = values.shape[0]
    n_threads = numba.get_num_threads()
    local_counts = np.zeros((n_threads, n_bins), dtype=np.int64)  # per-thread bins, avoids write races

    for i in prange(n):
        v = values[i]
        if drop_outside and (v < lo or v > hi):
            continue
        tid = numba.get_thread_id()
        idx = int((v - lo) / bin_width)
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


# --- Sb. Histogram: GPU ----------------------------------------------------

@cuda.jit
def _gpu_histogram_kernel(values: np.ndarray, counts: np.ndarray, n_bins: int, lo: float, hi: float,
                          bin_width: float, drop_outside: bool) -> None:
    """One thread per value, atomic-add into the shared bin counter."""
    i = cuda.grid(1)
    if i >= values.shape[0]:
        return
    v = values[i]
    if drop_outside and (v < lo or v > hi):
        return
    idx = int((v - lo) / bin_width)
    if idx < 0:
        idx = 0
    elif idx >= n_bins:
        idx = n_bins - 1
    cuda.atomic.add(counts, idx, 1)


def _gpu_histogram_counts(values: np.ndarray, n_bins: int, lo: float, hi: float, bin_width: float,
                          drop_outside: bool) -> np.ndarray:
    """Same counts as _cpu_histogram_counts, on the GPU."""
    d_values = cuda.to_device(values.astype(np.float32, copy=False))
    d_counts = cuda.to_device(np.zeros(n_bins, dtype=np.int64))
    threads = 256
    blocks = math.ceil(values.shape[0] / threads)
    _gpu_histogram_kernel[blocks, threads](d_values, d_counts, n_bins, np.float32(lo), np.float32(hi),
                                           np.float32(bin_width), drop_outside)
    cuda.synchronize()
    return d_counts.copy_to_host()


# --- Sc. Histogram: dispatch -----------------------------------------------

def histogram_counts(values: np.ndarray, n_bins: int, lo: float, hi: float, cuda_available: bool = False,
                     drop_outside: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(counts, bin_centers) over [lo, hi]; values outside are clamped to the edge bins, or dropped."""
    bin_width = (hi - lo) / n_bins
    if cuda_available:
        counts = _gpu_histogram_counts(values, n_bins, lo, hi, bin_width, drop_outside)
    else:
        counts = _cpu_histogram_counts(values, n_bins, lo, hi, bin_width, drop_outside)
    edges = lo + bin_width * np.arange(n_bins + 1)
    return counts, (edges[:-1] + edges[1:]) / 2


# --- Sd. Left-side Gaussian fit --------------------------------------------

def _gaussian(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
    """Gaussian curve amp * exp(-(x - mean)^2 / (2 sigma^2))."""
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def fit_left_gaussian(counts: np.ndarray, centers: np.ndarray, fix_mean: bool, sigma_seed: float,
                      center: float | None = None) -> tuple[float, float]:
    """(mean, sigma) of a Gaussian fit to the histogram's peak + left side.

    fix_mean=True (Fit 2): mean pinned at `center` or the tallest bin; False (Fit 1): mean fitted."""
    peak_idx = int(np.argmax(counts)) if center is None else int(np.searchsorted(centers, center, side="right")) - 1
    x_peak = float(centers[peak_idx]) if center is None else float(center)
    left_mask = centers <= x_peak
    x_left, y_left = centers[left_mask], counts[left_mask]

    if fix_mean:
        popt, _ = curve_fit(lambda x, amp, sigma: _gaussian(x, amp, x_peak, sigma), x_left, y_left,
                            p0=[float(counts[peak_idx]), sigma_seed])
        return x_peak, float(abs(popt[1]))

    popt, _ = curve_fit(_gaussian, x_left, y_left, p0=[float(counts[peak_idx]), x_peak, sigma_seed], maxfev=5000)
    return float(popt[1]), float(abs(popt[2]))


# --- Se. Z-score -----------------------------------------------------------

def img_zscore_convert(detrended: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """(x - mean) / sigma, elementwise (Fit 1 numbers in img_proc, Fit 2 numbers for the zone maps)."""
    return (detrended - mean) / sigma


# ===========================================================================
#
#   FIT 1 -- Z-SCORE: detrended raw stack -> (mean, sigma) for img_proc
#
# ===========================================================================

# --- 1a. Sigma seed: std of values at or below the peak --------------------

@njit(parallel=True)
def _cpu_masked_std(values: np.ndarray, threshold: float) -> float:
    """Std of values <= threshold, single pass (sum/sum_sq/count), no boolean-mask copy."""
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
def _gpu_masked_sum_kernel(values: np.ndarray, threshold: float, sums: np.ndarray) -> None:
    """sums = [sum, sum_sq, count] accumulator, one thread per value, atomic-add."""
    i = cuda.grid(1)
    if i >= values.shape[0]:
        return
    v = numba.float64(values[i])
    if v <= threshold:
        cuda.atomic.add(sums, 0, v)
        cuda.atomic.add(sums, 1, v * v)
        cuda.atomic.add(sums, 2, 1.0)


def _gpu_masked_std(values: np.ndarray, threshold: float) -> float:
    """Same std as _cpu_masked_std, on the GPU."""
    d_values = cuda.to_device(values.astype(np.float32, copy=False))  # float64 only in the sums (kernel)
    d_sums = cuda.to_device(np.zeros(3, dtype=np.float64))
    threads = 256
    blocks = math.ceil(values.shape[0] / threads)
    _gpu_masked_sum_kernel[blocks, threads](d_values, np.float64(threshold), d_sums)
    cuda.synchronize()
    total_sum, total_sum_sq, total_count = d_sums.copy_to_host()

    mean = total_sum / total_count
    variance = total_sum_sq / total_count - mean * mean
    return float(np.sqrt(variance)) if variance > 0.0 else 0.0


# --- 1b. Fit ---------------------------------------------------------------

def fit_zscore_background(detrended: np.ndarray, n_bins: int = ZSCORE_HIST_BINS,
                          cuda_available: bool = False) -> tuple[float, float]:
    """Fit 1: background (mean, sigma) over the full min->max range, for img_proc's stack-wide z-scoring."""
    values = detrended.ravel()
    counts, centers = histogram_counts(values, n_bins, float(values.min()), float(values.max()), cuda_available)

    x_peak = float(centers[int(np.argmax(counts))])
    sigma_seed = _gpu_masked_std(values, x_peak) if cuda_available else _cpu_masked_std(values, x_peak)
    return fit_left_gaussian(counts, centers, fix_mean=False, sigma_seed=float(sigma_seed))


# ===========================================================================
#
#   FIT 2 -- FLASH: *_BIEXP_ALS.tif -> (center, sigma); threshold = center + 2 sigma
#
# ===========================================================================

# --- 2a. float16 fast path: exact count of every float16 code -------------
# A float16 stack only holds 65536 distinct values, so one count per code is an exact,
# lossless histogram -- percentiles and any coarser histogram can then be read off it
# without sorting or re-reading the stack.

@njit(parallel=True)
def _cpu_code_counts(codes: np.ndarray) -> np.ndarray:
    """Count of each of the 65536 float16 codes (uint16 view), parallel."""
    n = codes.shape[0]
    n_threads = numba.get_num_threads()
    local_counts = np.zeros((n_threads, 65536), dtype=np.int64)  # per-thread bins, avoids write races
    for i in prange(n):
        local_counts[numba.get_thread_id(), codes[i]] += 1
    counts = np.zeros(65536, dtype=np.int64)
    for t in range(n_threads):
        counts += local_counts[t]
    return counts


def float16_code_counts(stack_f16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(values, counts) of every non-NaN float16 value present, values sorted ascending."""
    counts = _cpu_code_counts(stack_f16.view(np.uint16).ravel())
    values = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    keep = (counts > 0) & ~np.isnan(values)
    order = np.argsort(values[keep], kind="stable")
    return values[keep][order], counts[keep][order]


def percentiles_from_counts(values: np.ndarray, counts: np.ndarray, pcts: tuple[float, ...]) -> tuple[float, ...]:
    """np.percentile(..., method='linear') of the data described by sorted (values, counts)."""
    cum = np.cumsum(counts)
    n = int(cum[-1])
    out = []
    for pct in pcts:
        h = (n - 1) * pct / 100.0  # virtual index into the sorted data
        k = int(np.floor(h))
        v_lo = values[np.searchsorted(cum, k, side="right")]
        v_hi = values[np.searchsorted(cum, min(k + 1, n - 1), side="right")]
        out.append(float(v_lo + (h - k) * (v_hi - v_lo)))
    return tuple(out)


def rebin_code_counts(values: np.ndarray, counts: np.ndarray, n_bins: int, lo: float,
                      hi: float) -> tuple[np.ndarray, np.ndarray]:
    """(counts, bin_centers) of n_bins over [lo, hi], values outside dropped -- same binning as histogram_counts()."""
    bin_width = (hi - lo) / n_bins
    inside = (values >= lo) & (values <= hi)
    idx = np.clip(((values[inside] - lo) / bin_width).astype(np.int64), 0, n_bins - 1)
    binned = np.bincount(idx, weights=counts[inside], minlength=n_bins).astype(np.int64)
    edges = lo + bin_width * np.arange(n_bins + 1)
    return binned, (edges[:-1] + edges[1:]) / 2


# --- 2b. Peak --------------------------------------------------------------

def smoothed_peak(counts: np.ndarray, centers: np.ndarray) -> float:
    """Histogram peak = + -> - zero crossing of the Savitzky-Golay derivative nearest the smoothed maximum.

    Smoothing removes the float16 comb, so the peak doesn't jump between comb spikes."""
    bin_w = float(centers[1] - centers[0])
    window = max(5, round(FLASH_SMOOTH_FRAC * (centers[-1] - centers[0]) / bin_w) | 1)  # odd, >= 5
    smooth = savgol_filter(counts.astype(np.float64), window, 2)
    deriv = savgol_filter(counts.astype(np.float64), window, 2, deriv=1)
    crossings = np.flatnonzero((deriv[:-1] > 0) & (deriv[1:] <= 0))
    if not crossings.size:
        return float(centers[int(np.argmax(smooth))])
    i = int(crossings[np.argmin(np.abs(crossings - np.argmax(smooth)))])
    return float(centers[i] + bin_w * deriv[i] / (deriv[i] - deriv[i + 1]))  # interpolated zero


# --- 2c. Fit ---------------------------------------------------------------

def fit_flash_background(stack: np.ndarray, n_bins: int = FLASH_HIST_BINS,
                         cuda_available: bool = False) -> tuple[float, float]:
    """Fit 2: background (center, sigma) over the 0.1-99.9 percentile range.

    Center = smoothed_peak(); sigma from a left-side fit with the center pinned; float16 uses exact code counts."""
    if stack.dtype == np.float16:
        codes_value, codes_count = float16_code_counts(stack)
        lo, hi = percentiles_from_counts(codes_value, codes_count, FLASH_HIST_RANGE_PCT)
        counts, centers = rebin_code_counts(codes_value, codes_count, n_bins, lo, hi)
    else:
        values = stack.ravel()
        lo, hi = np.percentile(values, FLASH_HIST_RANGE_PCT)
        counts, centers = histogram_counts(values, n_bins, float(lo), float(hi), cuda_available, drop_outside=True)

    return fit_left_gaussian(counts, centers, fix_mean=True, sigma_seed=FLASH_SIGMA_SEED,
                             center=smoothed_peak(counts, centers))
