"""
Purpose

K-means clustering functions for ACh imaging data analysis.
Used to identify ACh releasing areas related to firing activities in each frame.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from tabulate import tabulate


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


def calculate_cluster_areas(clustered_frames: list[np.ndarray], magnification: str, n_clusters: int = 3) -> dict:
    """
    Calculate area sizes for each cluster label across all frames.

    Args:
        clustered_frames: List of clustered frames with labels (0, 1, 2, ...)
        magnification: Microscope magnification ("10X", "40X", or "60X")
        n_clusters: Number of clusters

    Returns:
        Dictionary with area statistics for each frame

    """
    # Define pixel-to-micron conversion based on magnification
    conversion_factors = {
        "10X": (1 / 0.75) ** 2,  # μm² per pixel
        "40X": (1 / 3) ** 2,
        "60X": (1 / 4.5) ** 2,
    }

    if magnification not in conversion_factors:
        valid_mags = list(conversion_factors.keys())
        msg = f"Magnification must be one of {valid_mags}"
        raise ValueError(msg)

    pixel_area = conversion_factors[magnification]

    # Calculate areas for each frame
    frame_areas = []
    for frame in clustered_frames:
        cluster_areas = {}
        for cluster_label in range(n_clusters):
            pixel_count = np.sum(frame == cluster_label)
            area_um2 = pixel_count * pixel_area
            cluster_areas[cluster_label] = {"pixel_count": pixel_count, "area_um2": area_um2}
        frame_areas.append(cluster_areas)

    return frame_areas


def visualize_clustering_results(  # noqa: C901, PLR0912, PLR0915
    original_frames: list[np.ndarray],
    clustered_frames: list[np.ndarray],
    spike_trace: list[np.ndarray] | list[list[np.ndarray]],
    span_of_frames: list[int],
    seg_index: int = 0,
    img_filename: str | None = None,
    magnification: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame | None]:
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
        magnification: Microscope magnification ("10X", "40X", or "60X") for area calculation

    """
    n_frames = len(original_frames)

    # Calculate cluster areas if magnification is provided
    df_areas = None
    if magnification is not None:
        print(f"\n=== Cluster Area Analysis (Magnification: {magnification}) ===")
        # Check all frames for unique labels, not just the first one
        all_labels = np.unique(np.concatenate([frame.flatten() for frame in clustered_frames]))
        n_clusters = len(all_labels)
        frame_areas = calculate_cluster_areas(clustered_frames, magnification, n_clusters)

        # Map cluster labels to color names
        cluster_colors_map = {0: "White", 1: "Yellow", 2: "Red"}

        # Create table data with area and sqrt (distance estimate)
        table_data = []
        for frame_idx, areas in enumerate(frame_areas):
            row = {"Frame": span_of_frames[frame_idx]}
            for cluster_label in sorted(areas.keys()):
                color_name = cluster_colors_map.get(cluster_label, f"Cluster {cluster_label}")
                area_um2 = areas[cluster_label]["area_um2"]
                sqrt_area = np.sqrt(area_um2)
                row[f"{color_name} (μm²)"] = f"{area_um2:.2f}"
                row[f"sqrt({color_name}) (μm) "] = f"{sqrt_area:.2f}"
            table_data.append(row)

        df_areas = pd.DataFrame(table_data)
        print("\n" + tabulate(df_areas, headers="keys", showindex=False, tablefmt="pretty"))

    # Layout constants
    n_image_rows = 2  # Original + clustered
    n_total_rows = n_image_rows + 1  # 2 image rows + 1 trace row
    n_cols = n_frames + 1  # frames + extra column for colorbar/legend

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(3 * n_frames + 1.5, 9))
    gs = GridSpec(n_total_rows, n_cols, figure=fig, width_ratios=[1] * n_frames + [0.3])

    # Calculate z-score range for consistent coloring
    all_frames_data = np.concatenate([frame.flatten() for frame in original_frames])
    vmin, vmax = np.percentile(all_frames_data, [1, 99])

    for i in range(n_frames):
        frame_title = f"Frame {span_of_frames[i]}"

        # Row 1: Original frame
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.imshow(original_frames[i], cmap="gray", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Original {frame_title}")
        ax1.axis("off")

        # Row 2: Clustered frame
        ax2 = fig.add_subplot(gs[1, i])
        current_clustered_frame = clustered_frames[i]
        cluster_colors = ["white", "yellow", "red"]  # 0=background, 1=moderate, 2=high ACh
        cluster_colormap = ListedColormap(cluster_colors)
        im2 = ax2.imshow(current_clustered_frame, cmap=cluster_colormap, vmin=0, vmax=2)
        ax2.set_title(f"Clustered {frame_title}")
        ax2.axis("off")

    # Add z-score colorbar in the extra column, row 1
    cbar_ax1 = fig.add_subplot(gs[0, n_frames])
    cbar1 = plt.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label("Z-score", rotation=90, labelpad=15)
    # Rotate tick labels vertically
    for label in cbar1.ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")

    # Add cluster colorbar in the extra column, row 2
    cbar_ax2 = fig.add_subplot(gs[1, n_frames])
    cluster_colors = ["white", "yellow", "red"]
    cluster_colormap = ListedColormap(cluster_colors)
    cbar2 = plt.colorbar(im2, cax=cbar_ax2, ticks=[0, 1, 2])
    cbar2.set_label("Cluster", rotation=90, labelpad=15)
    cbar2.ax.set_yticklabels(["Background", "Moderate", "High ACh"], rotation=90, va="center")

    # Row 3: Spike trace spanning only frame columns (aligns with images)
    spike_ax = fig.add_subplot(gs[2, :n_frames])

    # Check if spike_trace is a single trace or multiple traces
    is_multi_trace = isinstance(spike_trace[0], list) or (
        isinstance(spike_trace[0], np.ndarray) and spike_trace[0].ndim > 1
    )

    if is_multi_trace:
        # Plot multiple traces as overlays with distinct colors
        # Use a colormap with distinct colors
        n_traces = len(spike_trace)
        colors = cm.tab20(np.linspace(0, 1, n_traces))  # tab20 provides 20 distinct colors

        for idx, trace in enumerate(spike_trace):
            spike_ax.plot(trace[0], trace[1], alpha=0.6, linewidth=0.8, color=colors[idx])
        reference_time = spike_trace[0][0]
    else:
        # Plot single trace
        spike_ax.plot(spike_trace[0], spike_trace[1])
        reference_time = spike_trace[0]

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
    mag_prefix = f"({magnification}) " if magnification else ""
    if img_filename:
        base_title = f"{mag_prefix}{img_filename}: K-means Clustering Analysis"
    elif seg_index == -1:
        base_title = f"{mag_prefix}Spike-Triggered Average: K-means Clustering Analysis"
    else:
        base_title = f"{mag_prefix}Segment {seg_index + 1}: K-means Clustering Analysis"

    plt.suptitle(base_title)
    gs.update(wspace=0.05, hspace=0.3)  # Adjust spacing for GridSpec layout
    plt.show(block=False)
    plt.pause(0.001)

    return fig, df_areas


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
    _height, total_width = clustered_concatenated.shape
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
