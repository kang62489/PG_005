"""
Purpose

K-means clustering functions for ACh imaging data analysis.
Used to identify ACh releasing areas related to firing activities in each frame.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans


def prepare_frame_for_kmeans(image_frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Reshape 2D ACh imaging frame to 1D array for k-means clustering.

    WHY: K-means needs pixel intensities as 1D data, but images are 2D
    GOAL: Convert spatial data to intensity-based clustering data

    Args:
        image_frame: 2D numpy array (image frame)

    Returns:
        flattened_pixels: Flattened pixel intensities for k-means input
        frame_height, frame_width: Original frame dimensions

    """
    frame_height, frame_width = image_frame.shape
    flattened_pixels = image_frame.reshape(-1, 1)  # Flatten to (n_pixels, 1)
    return flattened_pixels, frame_height, frame_width


def calculate_wcss(flattened_pixels: np.ndarray, max_clusters: int = 6) -> list[float]:
    """
    Calculate Within-Cluster Sum of Squares for different cluster counts.

    Args:
        flattened_pixels: Flattened pixel intensities
        max_clusters: Maximum number of clusters to test

    Returns:
        wcss_values: List of WCSS values for each cluster count

    """
    wcss_values = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=100)
        kmeans.fit(flattened_pixels)
        wcss_values.append(kmeans.inertia_)
    return wcss_values


def find_optimal_clusters_elbow(wcss_values: list[float], max_clusters: int = 6) -> int:
    """
    Find optimal number of clusters using elbow method.

    Args:
        wcss_values: List of WCSS values
        max_clusters: Maximum clusters tested

    Returns:
        optimal_k: Optimal number of clusters

    """
    # Calculate rate of change (differences between consecutive WCSS values)
    differences = np.diff(wcss_values)

    # Calculate second differences (rate of change of the rate of change)
    second_differences = np.diff(differences)

    # Find elbow point - where second difference is maximum (steepest change in slope)
    if len(second_differences) > 0:
        elbow_index = np.argmax(second_differences) + 2  # +2 because we lost 2 indices from diff operations
        return min(elbow_index, max_clusters)
    return 1  # Default to 1 cluster if calculation fails


def apply_kmeans_to_frame(
    image_frame: np.ndarray, n_clusters: int = None, max_clusters: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply k-means clustering to identify ACh releasing areas with adaptive cluster count.

    WHY: Segment pixels into functional groups based on ACh signal intensity
    GOAL: Automatically detect optimal number of clusters (1 for pre-spike, 2-3 during release)

    Args:
        image_frame: 2D numpy array (image frame)
        n_clusters: Number of clusters (if None, uses elbow method to find optimal)
        max_clusters: Maximum clusters to test for elbow method

    Returns:
        intensity_ordered_frame: 2D array with cluster labels (0=background, highest=max ACh release)
        intensity_ordered_centers: Cluster centers sorted by intensity (low to high)

    """
    flattened_pixels, frame_height, frame_width = prepare_frame_for_kmeans(image_frame)

    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        wcss_values = calculate_wcss(flattened_pixels, max_clusters)
        n_clusters = find_optimal_clusters_elbow(wcss_values, max_clusters)
        print(f"    Optimal clusters detected: {n_clusters}")

    # K-means clustering with fixed random state for reproducibility
    kmeans_algorithm = KMeans(n_clusters=n_clusters, random_state=100)
    random_cluster_labels = kmeans_algorithm.fit_predict(flattened_pixels)
    cluster_centers = kmeans_algorithm.cluster_centers_.flatten()

    # CRITICAL: Sort clusters by intensity to ensure consistent labeling
    # 0 = background (lowest intensity)
    # 1 = moderate ACh release
    # 2 = high ACh release areas
    # Sort the indices of the centers based on their values
    intensity_sorted_indices = np.argsort(cluster_centers)

    # Create mapping from random k-means labels to intensity-ordered labels
    # Very cool method (fancy indexing/array indexing in Numpy)
    label_remapping_table = np.zeros(n_clusters, dtype=int)
    label_remapping_table[intensity_sorted_indices] = np.arange(n_clusters)

    # Remap labels to intensity order using fancy indexing
    # random_cluster_labels as indices for lookup in remapping table
    intensity_ordered_labels = label_remapping_table[random_cluster_labels]
    intensity_ordered_frame = intensity_ordered_labels.reshape(frame_height, frame_width)

    # Return sorted centers for reference
    intensity_ordered_centers = cluster_centers[intensity_sorted_indices]

    return intensity_ordered_frame, intensity_ordered_centers


def visualize_clustering_results(
    original_frames: list[np.ndarray],
    clustered_frames: list[np.ndarray],
    spike_trace: list[np.ndarray],
    span_of_frames: list[int],
    seg_index: int = 0,
) -> None:
    """
    Visualize original ACh signals vs clustered ACh releasing areas.

    WHY: See which areas release ACh together and how spatial patterns change
    GOAL: Identify coordinated ACh release during firing activities

    Args:
        original_frames: List of original frames
        clustered_frames: List of clustered frames
        seg_index: Segment index for title
        span_of_frames: List of frame numbers spanning the segment

    """
    n_frames = len(original_frames)

    # Layout constants
    n_image_rows = 2  # Original + clustered
    n_total_rows = n_image_rows + 1  # 2 image rows + 1 trace row
    trace_row_position = n_total_rows  # Bottom row for spike trace

    # Create figure and manually add subplots using 3-row grid
    fig = plt.figure(figsize=(3 * n_frames, 9))

    for i in range(n_frames):
        frame_title = f"Frame {span_of_frames[i]}"

        # Row 1: Original frame
        original_position = i + 1
        ax1 = fig.add_subplot(n_total_rows, n_frames, original_position)
        ax1.imshow(original_frames[i], cmap="gray")
        ax1.set_title(f"Original {frame_title}")
        ax1.axis("off")

        # Row 2: Clustered frame
        clustered_position = n_frames + i + 1
        ax2 = fig.add_subplot(n_total_rows, n_frames, clustered_position)
        current_clustered_frame = clustered_frames[i]
        cluster_colors = ["white", "yellow", "red"]  # 0=background, 1=moderate, 2=high ACh
        cluster_colormap = ListedColormap(cluster_colors)
        ax2.imshow(current_clustered_frame, cmap=cluster_colormap, vmin=0, vmax=2)
        ax2.set_title(f"Clustered {frame_title}")
        ax2.axis("off")

    # Row 3: Spike trace spanning all columns
    spike_ax = fig.add_subplot(n_total_rows, 1, trace_row_position)
    spike_ax.plot(spike_trace[0], spike_trace[1])

    # Plot vertical lines for each frame boundary
    div_width: int = int(len(spike_trace[0]) / n_frames)
    for i in range(n_frames + 1):
        time_position = spike_trace[0][i * div_width] if i * div_width < len(spike_trace[0]) else spike_trace[0][-1]
        spike_ax.axvline(x=time_position, color="gray", linestyle=":", alpha=0.7)

    spike_ax.set_xlabel("Time")
    spike_ax.set_ylabel("Vm")
    spike_ax.set_xlim(spike_trace[0][0], spike_trace[0][-1])
    spike_ax.set_title("Spike Trace")

    if seg_index == -1:
        plt.suptitle("Spike-Triggered Average: K-means Clustering Analysis")
    else:
        plt.suptitle(f"Segment {seg_index + 1}: K-means Clustering Analysis")
    plt.subplots_adjust(wspace=0)  # Remove horizontal spacing between image columns
    plt.show(block=False)
    plt.pause(0.001)


def process_segment_kmeans(
    image_seg: list[np.ndarray], n_clusters: int = None, max_clusters: int = 6
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Apply k-means clustering to all frames in a segment with adaptive clustering.

    WHY: ACh release changes rapidly - track frame-by-frame release patterns
    GOAL: Automatically detect appropriate cluster count for each frame (1 pre-spike, 2-3 during release)

    Args:
        image_seg: 3D array (frames, height, width)
        n_clusters: Number of clusters (if None, uses elbow method for each frame)
        max_clusters: Maximum clusters to test for elbow method

    Returns:
        all_clustered_frames: List of clustered frames with intensity-ordered labels
        all_cluster_centers: List of cluster centers for each frame (sorted by intensity)

    """
    all_clustered_frames = []
    all_cluster_centers = []

    for frame_idx, current_frame in enumerate(image_seg):
        # Apply k-means to identify ACh releasing areas with adaptive clustering
        intensity_ordered_frame, intensity_ordered_centers = apply_kmeans_to_frame(
            current_frame, n_clusters, max_clusters
        )

        all_clustered_frames.append(intensity_ordered_frame)
        all_cluster_centers.append(intensity_ordered_centers)

        print(f"  Frame {frame_idx + 1}/{len(image_seg)} processed")

    return all_clustered_frames, all_cluster_centers
