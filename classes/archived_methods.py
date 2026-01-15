"""
Archived spatial categorization methods.

These methods were removed from SpatialCategorizer as they didn't perform well
for the ACh imaging project, but are preserved here for future reference.

Original location: classes/spatial_categorization.py
Archived: 2026-01-15
"""

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import DBSCAN


def method_dbscan(
    frame: np.ndarray,
    frame_idx: int,
    thresh_dim: float,
    thresh_bright: float,
    thresholds_used: tuple,
    eps: float = 3.0,
    min_samples: int = 10,
    intensity_scale: float = 10.0,
    min_region_size: int = 20,
) -> tuple[np.ndarray, dict]:
    """
    DBSCAN spatial clustering.

    Args:
        frame: 2D image array
        frame_idx: Frame index for stats
        thresh_dim: Threshold for dim signal
        thresh_bright: Threshold for bright signal
        thresholds_used: Tuple of thresholds for stats
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        intensity_scale: Scale factor for intensity in feature space
        min_region_size: Minimum pixels per region

    Returns:
        categorized: 2D array with 0=background, 1=dim, 2=bright
        stats: Dictionary with frame statistics
    """
    mask = frame > thresh_dim
    coords = np.argwhere(mask)
    intensities = frame[mask].reshape(-1, 1)

    if len(coords) == 0:
        return np.zeros_like(frame, dtype=int), {
            "frame_idx": frame_idx,
            "num_regions": 0,
            "region_details": [],
            "thresholds": thresholds_used,
        }

    features = np.hstack([coords, intensities * intensity_scale])
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(features)

    labeled_image = np.zeros_like(frame, dtype=int)
    for idx, (y, x) in enumerate(coords):
        if cluster_labels[idx] != -1:
            labeled_image[y, x] = cluster_labels[idx] + 1

    categorized = np.zeros_like(frame, dtype=int)
    region_stats = []

    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
        cluster_mask = labeled_image == (cluster_id + 1)
        cluster_size = np.sum(cluster_mask)

        if cluster_size < min_region_size:
            continue

        mean_intensity = np.mean(frame[cluster_mask])

        if mean_intensity > thresh_bright:
            categorized[cluster_mask] = 2
            category_name = "bright"
        else:
            categorized[cluster_mask] = 1
            category_name = "dim"

        region_stats.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_size,
                "mean_z": mean_intensity,
                "category_name": category_name,
            }
        )

    return categorized, {
        "frame_idx": frame_idx,
        "num_regions": len(region_stats),
        "num_noise_pixels": np.sum(cluster_labels == -1),
        "region_details": region_stats,
        "thresholds": thresholds_used,
    }


def method_region_growing(
    frame: np.ndarray,
    frame_idx: int,
    _thresh_dim: float,  # Unused but kept for API consistency
    thresh_bright: float,
    thresholds_used: tuple,
    seed_threshold: float = 2.0,
    growth_threshold: float = 0.5,
    max_diff: float = 0.5,
    min_region_size: int = 20,
) -> tuple[np.ndarray, dict]:
    """
    Region growing from seeds.

    Args:
        frame: 2D image array
        frame_idx: Frame index for stats
        thresh_dim: Threshold for dim signal
        thresh_bright: Threshold for bright signal
        thresholds_used: Tuple of thresholds for stats
        seed_threshold: Threshold for seed detection
        growth_threshold: Minimum value for region growth
        max_diff: Maximum intensity difference for growth
        min_region_size: Minimum pixels per region

    Returns:
        categorized: 2D array with 0=background, 1=dim, 2=bright
        stats: Dictionary with frame statistics
    """
    seeds = frame > seed_threshold
    labeled_seeds, num_seeds = label(seeds)

    regions = np.zeros_like(frame, dtype=int)
    visited = np.zeros_like(frame, dtype=bool)

    region_id = 1
    for seed_label in range(1, num_seeds + 1):
        seed_coords = np.argwhere(labeled_seeds == seed_label)
        if len(seed_coords) == 0:
            continue

        seed_y, seed_x = seed_coords[0]
        seed_value = frame[seed_y, seed_x]

        to_check = [(seed_y, seed_x)]
        regions[seed_y, seed_x] = region_id
        visited[seed_y, seed_x] = True

        while to_check:
            y, x = to_check.pop(0)

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue

                    ny, nx = y + dy, x + dx

                    if ny < 0 or ny >= frame.shape[0] or nx < 0 or nx >= frame.shape[1]:
                        continue

                    if visited[ny, nx]:
                        continue

                    neighbor_value = frame[ny, nx]
                    if neighbor_value > growth_threshold and abs(neighbor_value - seed_value) < max_diff:
                        regions[ny, nx] = region_id
                        visited[ny, nx] = True
                        to_check.append((ny, nx))

        region_id += 1

    # Categorize regions by intensity
    categorized = np.zeros_like(frame, dtype=int)
    region_stats = []

    for rid in range(1, region_id):
        region_mask = regions == rid
        region_size = np.sum(region_mask)

        if region_size < min_region_size:
            continue

        mean_intensity = np.mean(frame[region_mask])

        if mean_intensity > thresh_bright:
            categorized[region_mask] = 2
            category_name = "bright"
        else:
            categorized[region_mask] = 1
            category_name = "dim"

        region_stats.append(
            {"region_id": rid, "size": region_size, "mean_z": mean_intensity, "category_name": category_name}
        )

    return categorized, {
        "frame_idx": frame_idx,
        "num_regions": len(region_stats),
        "num_seeds": num_seeds,
        "region_details": region_stats,
        "thresholds": thresholds_used,
    }
