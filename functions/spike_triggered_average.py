## Modules
# Third-party imports
import numpy as np
from rich.console import Console

# Set up rich console
console = Console()


def spike_trig_avg(
    lst_img_segments: list[np.ndarray],
    method: str = "mean",
) -> np.ndarray:
    """
    Compute spike-triggered average across all segments.
    Segments are aligned by their central frame (spike frame).

    Args:
        lst_img_segments: List of 3D arrays (frames, height, width)
        method: Averaging method - "mean" (default) or "median" (robust to outliers)

    Returns:
        Averaged segment with shape (target_frames, height, width)
    """
    target_frames = max(seg.shape[0] for seg in lst_img_segments)
    target_center = target_frames // 2
    img_shape = lst_img_segments[0].shape[1:]  # (height, width)
    n_segments = len(lst_img_segments)

    console.print(f"Spike-triggered averaging: {n_segments} segments, method={method}")

    if method == "median":
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
        averaged_segment = np.nanmedian(stacked, axis=0)

    else:  # method == "mean" (default, original behavior)
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
        averaged_segment = frame_sum / frame_count[:, None, None]

    console.print(f"Output shape: {averaged_segment.shape}")

    return averaged_segment
