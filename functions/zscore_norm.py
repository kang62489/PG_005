## Modules
# Third-party imports
import numpy as np


def zscore_normalize_segments(lst_img_segments: list[np.ndarray]) -> list[np.ndarray]:
    """
    Apply z-score normalization to each image segment.

    Args:
        lst_img_segments: List of 3D arrays (frames, height, width)

    Returns:
        lst_img_segments_zscore: List of z-score normalized segments
    """
    lst_img_segments_zscore = []

    for segment in lst_img_segments:
        seg_length = segment.shape[0]
        spike_frame_idx = seg_length // 2
        baseline_frames_count = spike_frame_idx
        baseline = segment[:baseline_frames_count]

        # Calculate pixel-wise mean and std
        baseline_mean = np.mean(baseline, axis=0)
        baseline_std = np.std(baseline, axis=0)
        baseline_std[baseline_std == 0] = 1  # Avoid division by zero

        # Calculate z-scores for all frames in segment
        zscore_segment = (segment - baseline_mean) / baseline_std
        lst_img_segments_zscore.append(zscore_segment)

    return lst_img_segments_zscore
