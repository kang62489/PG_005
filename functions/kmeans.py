"""
K-means clustering functions for calcium imaging data analysis.
Used to identify neural activity regions in each frame.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def prepare_frame_for_kmeans(frame, n_clusters=3):
    """
    Reshape 2D calcium imaging frame to 1D array for k-means clustering.

    WHY: K-means needs pixel intensities as 1D data, but images are 2D
    GOAL: Convert spatial data to intensity-based clustering data

    Args:
        frame: 2D numpy array (image frame)
        n_clusters: Number of clusters (default 3: background, inactive, active)

    Returns:
        pixels: Flattened pixel intensities
        h, w: Original frame dimensions

    """
    h, w = frame.shape
    pixels = frame.reshape(-1, 1)  # Flatten to (n_pixels, 1)
    return pixels, h, w


def apply_kmeans_to_frame(frame, n_clusters=3):
    """
    Apply k-means clustering to identify neural activity regions.

    WHY: Segment pixels into functional groups based on calcium signal intensity
    GOAL: Separate background (low), inactive neurons (medium), active neurons (high)

    Args:
        frame: 2D numpy array (image frame)
        n_clusters: Number of clusters (default 3)

    Returns:
        clustered_frame: 2D array with cluster labels (0=background, 2=active)
        sorted_centers: Cluster centers sorted by intensity (low to high)

    """
    pixels, h, w = prepare_frame_for_kmeans(frame, n_clusters)

    # K-means clustering with fixed random state for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    raw_labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.flatten()

    # CRITICAL: Sort clusters by intensity to ensure consistent labeling
    # 0 = background (lowest intensity)
    # 1 = medium activity
    # 2 = high activity (active neurons)
    sorted_indices = np.argsort(centers)

    # Create mapping from random k-means labels to intensity-ordered labels
    label_mapping = np.zeros(n_clusters, dtype=int)
    label_mapping[sorted_indices] = np.arange(n_clusters)

    # Remap labels to intensity order
    intensity_ordered_labels = label_mapping[raw_labels]
    clustered_frame = intensity_ordered_labels.reshape(h, w)

    # Return sorted centers for reference
    sorted_centers = centers[sorted_indices]

    return clustered_frame, sorted_centers


def visualize_clustering_results(original_segment, clustered_segment, seg_idx=0, frame_numbers=None):
    """
    Visualize original calcium signals vs clustered neural activity regions.

    WHY: See which neurons fire together and how spatial patterns change
    GOAL: Identify coordinated neural activity during spike events

    Args:
        original_segment: List of original frames
        clustered_segment: List of clustered frames
        seg_idx: Segment index for title

    """
    n_frames = len(original_segment)

    # 5 rows: Original + Cluster 0 + Cluster 1 + Cluster 2 + All Clusters
    fig, axes = plt.subplots(5, n_frames, figsize=(3 * n_frames, 12))

    # Handle single frame case
    if n_frames == 1:
        axes = axes.reshape(5, 1)

    for i in range(n_frames):
        # Use actual frame numbers if provided
        if frame_numbers is not None:
            frame_title = f"Frame {frame_numbers[i]}"
        else:
            frame_title = f"Frame {i + 1}"

        # Row 0: Original frame
        axes[0, i].imshow(original_segment[i], cmap="gray")
        axes[0, i].set_title(f"Original {frame_title}")
        axes[0, i].axis("off")

        # Get clustered frame for this iteration
        clustered_frame = clustered_segment[i]

        # Row 1: Cluster 0 only (Background)
        cluster_0 = (clustered_frame == 0).astype(int)
        axes[1, i].imshow(cluster_0, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("Cluster 0 (Background)")
        axes[1, i].axis("off")

        # Row 2: Cluster 1 only (Medium Activity)
        cluster_1 = (clustered_frame == 1).astype(int)
        axes[2, i].imshow(cluster_1, cmap="Blues", vmin=0, vmax=1)
        axes[2, i].set_title("Cluster 1 (Medium)")
        axes[2, i].axis("off")

        # Row 3: Cluster 2 only (High Activity)
        cluster_2 = (clustered_frame == 2).astype(int)
        axes[3, i].imshow(cluster_2, cmap="Reds", vmin=0, vmax=1)
        axes[3, i].set_title("Cluster 2 (Active)")
        axes[3, i].axis("off")

        # Row 4: All clusters combined with colors
        from matplotlib.colors import ListedColormap

        custom_colors = ["black", "blue", "red"]  # 0=black, 1=blue, 2=red
        custom_cmap = ListedColormap(custom_colors)
        axes[4, i].imshow(clustered_frame, cmap=custom_cmap, vmin=0, vmax=2)
        axes[4, i].set_title("All Clusters")
        axes[4, i].axis("off")

    plt.suptitle(f"Segment {seg_idx + 1}: K-means Clustering Analysis")
    plt.tight_layout()
    plt.show()


def process_segment_kmeans(segment, n_clusters=3):
    """
    Apply k-means clustering to all frames in a segment.

    WHY: Neural activity changes rapidly - track frame-by-frame activation patterns
    GOAL: See how neuron activation patterns change during spike events

    Args:
        segment: 3D array (frames, height, width)
        n_clusters: Number of clusters

    Returns:
        clustered_frames: List of clustered frames
        all_centers: List of cluster centers for each frame

    """
    clustered_frames = []
    all_centers = []

    for frame_idx, frame in enumerate(segment):
        # Apply k-means to identify active vs inactive regions
        clustered_frame, centers = apply_kmeans_to_frame(frame, n_clusters)

        clustered_frames.append(clustered_frame)
        all_centers.append(centers)

        print(f"  Frame {frame_idx + 1}/{len(segment)} processed")

    return clustered_frames, all_centers
