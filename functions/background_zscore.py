"""
background_zscore.py  --  Stack-wide background-noise z-score normalization (CPU).

Public API
----------
fit_background_sigma(residual, n_bins)   ->  tuple[float, float]
zscore_normalize(residual, mean, sigma)  ->  np.ndarray
"""

## Modules
# Third-party imports
import numpy as np
from scipy.optimize import curve_fit

# Constants
N_HIST_BINS = 1000


def _gaussian(x: np.ndarray, amp: float, mean: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def fit_background_sigma(residual: np.ndarray, n_bins: int = N_HIST_BINS) -> tuple[float, float]:
    """
    Estimate the background-noise center/sigma from a stack-wide histogram.

    Pools every pixel/frame value in `residual` (bi-exp detrend residual, y - trend)
    into one histogram, finds the peak (background mode), then fits a Gaussian to
    the peak and its left side only — the right side is contaminated by real signal
    transients (hotspots), so fitting only the uncontaminated half keeps the sigma
    estimate robust.

    Args:
        residual: Detrend residual stack, any shape.
        n_bins: Number of histogram bins spanning the 0.01-99.99 percentile range.

    Returns:
        (mean, sigma) of the fitted background-noise Gaussian.
    """
    values = residual.ravel()
    lo, hi = np.percentile(values, [0.01, 99.99])
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2

    peak_idx = int(np.argmax(counts))
    x_peak = float(centers[peak_idx])

    left_mask = centers <= x_peak
    x_left = centers[left_mask]
    y_left = counts[left_mask]

    p0 = [float(counts[peak_idx]), x_peak, float(np.std(values[values <= x_peak]))]
    popt, _ = curve_fit(_gaussian, x_left, y_left, p0=p0, maxfev=5000)
    _amp_fit, mean_fit, sigma_fit = popt
    return float(mean_fit), float(abs(sigma_fit))


def zscore_normalize(residual: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Convert a detrend residual stack to z-scores using a stack-wide background mean/sigma."""
    return (residual - mean) / sigma
