"""Whole-frame strip profiles and an exploratory directional-persistence null test.

Inspired by Matityahu 2023. No claim to reproduce their manually tuned wave
detector: here zero velocity breaks a run, and all nonzero runs enter the mean.
"""

import numpy as np
import tifffile

AXIS_NAMES = ("Left to right", "Top to bottom", "Top-left to bottom-right", "Bottom-left to top-right")
BAND_PX = 16


def stack_profiles(path) -> list[np.ndarray]:
    """Average all image pixels in 16-pixel bands; stream memory-mapped frames."""
    stack = tifffile.memmap(path, mode="r")
    _, height, width = stack.shape
    yy, xx = np.indices((height, width))
    projections = [xx, yy, (xx + yy) / np.sqrt(2), (xx - yy + height - 1) / np.sqrt(2)]
    bins = [(p / BAND_PX).astype(np.int64).ravel() for p in projections]
    counts = [np.bincount(b) for b in bins]
    profiles = [np.empty((len(c), len(stack)), dtype=np.float32) for c in counts]
    for t, frame in enumerate(stack):
        values = frame.astype(np.float64).ravel()
        for result, b, count in zip(profiles, bins, counts, strict=True):
            result[:, t] = np.bincount(b, weights=values, minlength=len(count)) / count
    return profiles


def mean_directional_run(positions: np.ndarray) -> float:
    """Mean duration in frame intervals; zeros break runs and are excluded."""
    signs = np.sign(np.diff(positions))
    if not np.any(signs):
        return 0.0
    boundaries = np.r_[0, np.flatnonzero(np.diff(signs)) + 1, len(signs)]
    lengths = np.diff(boundaries)
    return float(lengths[signs[boundaries[:-1]] != 0].mean())


def persistence_test(profile: np.ndarray, rng, n_permutations=1000) -> dict:
    """Shuffle contiguous peak-position blocks, retaining within-block structure."""
    positions = np.argmax(profile, axis=0)
    observed = mean_directional_run(positions) * 50
    result = {"observed_ms": observed, "block_frames": [], "null95_ms": [], "null_max_ms": [], "p": [],
              "edge_peak_fraction": float(np.mean((positions == 0) | (positions == len(profile) - 1)))}
    for size in range(1, 21):
        blocks = [positions[start:start + size] for start in range(0, len(positions), size)]
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
            null[i] = mean_directional_run(shuffled) * 50
        result["block_frames"].append(size)
        result["null95_ms"].append(float(np.percentile(null, 95)))
        result["null_max_ms"].append(float(null.max()))
        result["p"].append(float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)))
    return result
