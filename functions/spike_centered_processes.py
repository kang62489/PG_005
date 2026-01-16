## Modules
# Third-party imports
import numpy as np
from rich.console import Console

# Set up rich console
console = Console()


def spike_centered_median(
    lst_img_segments: list[np.ndarray],
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Compute spike-centered median across all segments.
    Segments are aligned by their central frame (spike frame).
    Median is robust to outliers - removes random bright spots.

    Args:
        lst_img_segments: List of 3D arrays (frames, height, width)

    Returns:
        (median_segment, zscore_range) where:
        - median_segment: shape (target_frames, height, width)
        - zscore_range: (vmin, vmax) tuple for consistent color scaling
    """
    target_frames = max(seg.shape[0] for seg in lst_img_segments)
    target_center = target_frames // 2
    img_shape = lst_img_segments[0].shape[1:]  # (height, width)
    n_segments = len(lst_img_segments)

    console.print(f"Spike-centered median: {n_segments} segments")

    # Stack all segments into 4D array for median calculation
    # Shape: (n_segments, target_frames, height, width)
    stacked = np.full((n_segments, target_frames, *img_shape), np.nan, dtype=np.float64)

    for seg_idx, segment in enumerate(lst_img_segments):
        n_frames = segment.shape[0]
        seg_center = n_frames // 2
        start = target_center - seg_center
        end = start + n_frames
        stacked[seg_idx, start:end] = segment

    # Median along segment axis, ignoring NaN
    result = np.nanmedian(stacked, axis=0)

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
