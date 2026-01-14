## Modules
# Third-party imports
import numpy as np
from rich.console import Console

# Set up rich console
console = Console()


def spike_trig_avg(lst_img_segments: list[np.ndarray]) -> np.ndarray:
    """
    Compute spike-triggered average across all segments.
    Segments are aligned by their central frame (spike frame).
    """
    target_frames = max(seg.shape[0] for seg in lst_img_segments)
    target_center = target_frames // 2
    img_shape = lst_img_segments[0].shape[1:]  # (height, width)

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

    console.print(f"Averaged {len(lst_img_segments)} segments, output shape: {averaged_segment.shape}")

    return averaged_segment
