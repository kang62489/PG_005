## Modules
# Third-party imports
import numpy as np
from numba import njit, prange
from rich.console import Console

# Set up rich console
console = Console()


@njit(parallel=True)
def _median_axis0(stacked: np.ndarray) -> np.ndarray:
    """Compute median along axis-0 (segments) of a (S, F, H, W) array.

    Parallelised over H rows using numba prange.
    """
    s_count, n_frames, height, width = stacked.shape
    result = np.empty((n_frames, height, width), dtype=np.float64)
    mid = s_count // 2

    for h in prange(height):
        for w in range(width):
            for f in range(n_frames):
                vals = np.empty(s_count, dtype=np.float64)
                for s in range(s_count):
                    vals[s] = stacked[s, f, h, w]
                vals.sort()
                if s_count % 2 == 1:
                    result[f, h, w] = vals[mid]
                else:
                    result[f, h, w] = (vals[mid - 1] + vals[mid]) * 0.5
    return result


def spike_centered_median(
    lst_img_segments: list[np.ndarray],
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Compute spike-centered median across all segments.
    Segments are aligned by their central frame (spike frame).
    Median is robust to outliers - removes random bright spots.

    Args:
        lst_img_segments: List of 3D arrays (frames, height, width), all identical shape.

    Returns:
        (median_segment, zscore_range) where:
        - median_segment: shape (target_frames, height, width)
        - zscore_range: (vmin, vmax) tuple for consistent color scaling
    """
    n_segments = len(lst_img_segments)

    console.print(f"Spike-centered median: {n_segments} segments")

    # All segments are identical shape — stack then run numba parallel median
    stacked = np.stack(lst_img_segments, axis=0).astype(np.float64)  # (S, F, H, W)
    result = _median_axis0(stacked)

    # Calculate z-score range (1st and 99th percentile) for consistent color scaling
    vmin, vmax = np.percentile(result, [1, 99])
    zscore_range = (float(vmin), float(vmax))

    console.print(f"Output shape: {result.shape}, Z-score range: [{vmin:.2f}, {vmax:.2f}]")
    return result, zscore_range


def spike_centered_avg(
    lst_img_segments: list[np.ndarray],
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Compute spike-centered average (mean) across all segments.
    Segments are aligned by their central frame (spike frame).
    Note: Mean is sensitive to outliers.

    Args:
        lst_img_segments: List of 3D arrays (frames, height, width)

    Returns:
        (avg_segment, zscore_range) where:
        - avg_segment: shape (target_frames, height, width)
        - zscore_range: (vmin, vmax) tuple for consistent color scaling
    """
    target_frames = max(seg.shape[0] for seg in lst_img_segments)
    target_center = target_frames // 2
    img_shape = lst_img_segments[0].shape[1:]  # (height, width)
    n_segments = len(lst_img_segments)

    console.print(f"Spike-centered average: {n_segments} segments")

    # Accumulator arrays
    frame_sum = np.zeros((target_frames, *img_shape), dtype=np.float64)
    frame_count = np.zeros(target_frames, dtype=np.int32)

    for segment in lst_img_segments:
        n_frames = segment.shape[0]
        seg_center = n_frames // 2
        start = target_center - seg_center
        end = start + n_frames

        frame_sum[start:end] += segment
        frame_count[start:end] += 1

    # Average
    result = frame_sum / frame_count[:, None, None]

    # Calculate z-score range (1st and 99th percentile) for consistent color scaling
    vmin, vmax = np.percentile(result, [1, 99])
    zscore_range = (float(vmin), float(vmax))

    console.print(f"Output shape: {result.shape}, Z-score range: [{vmin:.2f}, {vmax:.2f}]")
    return result, zscore_range
