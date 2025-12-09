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


def apply_kmeans_to_frame(image_frame: np.ndarray, n_clusters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply k-means clustering to identify ACh releasing areas.

    WHY: Segment pixels into functional groups based on ACh signal intensity
    GOAL: Separate background (low), moderate ACh release (medium), high ACh release (high)

    Args:
        image_frame: 2D numpy array (image frame)
        n_clusters: Number of clusters (default 3)

    Returns:
        intensity_ordered_frame: 2D array with cluster labels (0=background, 2=high ACh release)
        intensity_ordered_centers: Cluster centers sorted by intensity (low to high)

    """
    flattened_pixels, frame_height, frame_width = prepare_frame_for_kmeans(image_frame)

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
    spike_trace: list[np.ndarray] | list[list[np.ndarray]],
    span_of_frames: list[int],
    seg_index: int = 0,
    img_filename: str = None,
) -> None:
    """
    Visualize original ACh signals vs clustered ACh releasing areas.

    WHY: See which areas release ACh together and how spatial patterns change
    GOAL: Identify coordinated ACh release during firing activities

    Args:
        original_frames: List of original frames
        clustered_frames: List of clustered frames
        spike_trace: Either [time, voltage] for single trace, or list of [time, voltage] pairs for multiple traces
        seg_index: Segment index for title
        span_of_frames: List of frame numbers spanning the segment
        img_filename: Name of the analyzed image file (optional)

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

    # Check if spike_trace is a single trace or multiple traces
    is_multi_trace = isinstance(spike_trace[0], list) or (
        isinstance(spike_trace[0], np.ndarray) and spike_trace[0].ndim > 1
    )

    if is_multi_trace:
        # Plot multiple traces as overlays with distinct colors
        # Use a colormap with distinct colors
        from matplotlib import cm

        n_traces = len(spike_trace)
        colors = cm.tab20(np.linspace(0, 1, n_traces))  # tab20 provides 20 distinct colors

        for idx, trace in enumerate(spike_trace):
            spike_ax.plot(trace[0], trace[1], alpha=0.6, linewidth=0.8, color=colors[idx])
        reference_time = spike_trace[0][0]
        reference_voltage = spike_trace[0][1]
    else:
        # Plot single trace
        spike_ax.plot(spike_trace[0], spike_trace[1])
        reference_time = spike_trace[0]
        reference_voltage = spike_trace[1]

    # Plot vertical lines for each frame boundary
    div_width: int = int(len(reference_time) / n_frames)
    for i in range(n_frames + 1):
        time_position = reference_time[i * div_width] if i * div_width < len(reference_time) else reference_time[-1]
        spike_ax.axvline(x=time_position, color="gray", linestyle=":", alpha=0.7)

    spike_ax.set_xlabel("Time (ms)")
    spike_ax.set_ylabel("Vm (mV)")
    spike_ax.set_xlim(reference_time[0], reference_time[-1])

    if is_multi_trace:
        spike_ax.set_title(f"Spike Traces (n={len(spike_trace)} overlays)")
    else:
        spike_ax.set_title("Spike Trace")

    # Create title based on image filename or default
    if img_filename:
        base_title = f"{img_filename}: K-means Clustering Analysis"
    elif seg_index == -1:
        base_title = "Spike-Triggered Average: K-means Clustering Analysis"
    else:
        base_title = f"Segment {seg_index + 1}: K-means Clustering Analysis"

    plt.suptitle(base_title)
    plt.subplots_adjust(wspace=0)  # Remove horizontal spacing between image columns
    plt.show(block=False)
    plt.pause(0.001)


def process_segment_kmeans(
    image_seg: list[np.ndarray], n_clusters: int = 3
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Apply k-means clustering to all frames in a segment.

    WHY: ACh release changes rapidly - track frame-by-frame release patterns
    GOAL: See how ACh release patterns change during firing activities

    Args:
        image_seg: 3D array (frames, height, width)
        n_clusters: Number of clusters

    Returns:
        all_clustered_frames: List of clustered frames with intensity-ordered labels
        all_cluster_centers: List of cluster centers for each frame (sorted by intensity)

    """
    all_clustered_frames = []
    all_cluster_centers = []

    for frame_idx, current_frame in enumerate(image_seg):
        # Apply k-means to identify ACh releasing areas
        intensity_ordered_frame, intensity_ordered_centers = apply_kmeans_to_frame(current_frame, n_clusters)

        all_clustered_frames.append(intensity_ordered_frame)
        all_cluster_centers.append(intensity_ordered_centers)

        print(f"  Frame {frame_idx + 1}/{len(image_seg)} processed")

    return all_clustered_frames, all_cluster_centers


def concatenate_frames_horizontally(image_frames: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple frames horizontally into a single wide image.

    WHY: Create single image (1024, 1024*9) for k-means clustering
    GOAL: Analyze all frames together as one spatial pattern

    Args:
        image_frames: List of 2D numpy arrays (9 frames)

    Returns:
        concatenated_image: Single wide image (height, width*n_frames)

    """
    concatenated_image = np.concatenate(image_frames, axis=1)
    print(f"Concatenated {len(image_frames)} frames into shape: {concatenated_image.shape}")
    return concatenated_image


def apply_kmeans_to_concatenated(image_frames: list[np.ndarray], n_clusters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply k-means clustering to horizontally concatenated frames.

    WHY: Find spatial patterns across entire concatenated sequence
    GOAL: Identify regions that behave consistently across time

    Args:
        image_frames: List of 2D numpy arrays (9 frames)
        n_clusters: Number of clusters (default 3)

    Returns:
        clustered_concatenated: Wide clustered image (height, width*n_frames)
        cluster_centers: Cluster centers sorted by intensity

    """
    # Concatenate frames horizontally
    concatenated_image = concatenate_frames_horizontally(image_frames)

    # Apply standard k-means to the concatenated image
    clustered_concatenated, cluster_centers = apply_kmeans_to_frame(concatenated_image, n_clusters)

    return clustered_concatenated, cluster_centers


def split_concatenated_result(clustered_concatenated: np.ndarray, n_frames: int) -> list[np.ndarray]:
    """
    Split concatenated clustering result back into individual frames.

    Args:
        clustered_concatenated: Wide clustered image (height, width*n_frames)
        n_frames: Number of original frames

    Returns:
        clustered_frames: List of individual clustered frames

    """
    height, total_width = clustered_concatenated.shape
    frame_width = total_width // n_frames

    clustered_frames = []
    for i in range(n_frames):
        start_col = i * frame_width
        end_col = (i + 1) * frame_width
        frame = clustered_concatenated[:, start_col:end_col]
        clustered_frames.append(frame)

    return clustered_frames


def process_segment_kmeans_concatenated(
    image_seg: list[np.ndarray], n_clusters: int = 3
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Apply k-means clustering to segment by concatenating all frames horizontally.

    WHY: Analyze entire sequence as single spatial pattern
    GOAL: Find consistent regions across time sequence

    Args:
        image_seg: List of 2D arrays (9 frames)
        n_clusters: Number of clusters

    Returns:
        clustered_frames: List of individual clustered frames
        cluster_centers: Cluster centers from concatenated analysis

    """
    print(f"Processing {len(image_seg)} frames as concatenated image with {n_clusters} clusters...")

    # Apply k-means to concatenated frames
    clustered_concatenated, cluster_centers = apply_kmeans_to_concatenated(image_seg, n_clusters)

    # Split result back into individual frames for visualization
    clustered_frames = split_concatenated_result(clustered_concatenated, len(image_seg))

    print(f"Concatenated clustering completed. Found {n_clusters} spatial patterns.")

    return clustered_frames, cluster_centers
