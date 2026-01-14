"""
Spatial-aware intensity categorization for ACh imaging.

This module provides methods that consider both intensity AND spatial connectivity,
addressing the limitation of k-means which treats pixels independently.

Use these functions to:
1. Identify spatially connected ACh release regions
2. Filter out isolated noise pixels
3. Categorize regions based on intensity (background/dim/bright)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, binary_erosion, binary_dilation, generate_binary_structure
from skimage.filters import threshold_multiotsu, threshold_li, threshold_otsu, threshold_yen


def categorize_spatial_connected(
    image: np.ndarray,
    threshold_dim: float | None = 0.5,
    threshold_bright: float | None = 1.5,
    min_region_size: int = 20,
    threshold_method: str = "manual",
) -> dict:
    """
    Categorize pixels into background/dim/bright using spatial connectivity.

    This is a drop-in replacement for k-means that considers spatial relationships.

    Process:
    1. Threshold image into potential signal regions
    2. Find spatially connected components
    3. Filter out small regions (noise)
    4. Categorize by mean intensity

    Args:
        image: 2D z-scored array (frame from your imaging data)
        threshold_dim: Z-score threshold for dim signal (default: 0.5, ignored if threshold_method != "manual")
        threshold_bright: Z-score threshold for bright signal (default: 1.5, ignored if threshold_method != "manual")
        min_region_size: Minimum pixels in region (removes noise, default: 20)
        threshold_method: Thresholding method to use:
            - "manual": Use provided threshold_dim and threshold_bright values
            - "multiotsu": Auto-detect 2 thresholds using Multi-Otsu
            - "li_double": Two-pass Li (1st: background, 2nd: dim/bright) - recommended
            - "otsu_double": Two-pass Otsu (1st: background, 2nd: dim/bright)
            - "li": Single Li threshold (no dim/bright distinction)
            - "otsu": Single Otsu threshold (no dim/bright distinction)
            - "yen": Single Yen threshold (no dim/bright distinction)

    Returns:
        dict with:
            'categorized': 2D array (0=background, 1=dim, 2=bright)
            'regions': 2D array with region labels (1, 2, 3, ...)
            'stats': List of dicts with region info
            'num_regions': Number of valid regions found
            'thresholds': Tuple of (threshold_dim, threshold_bright) used

    Example:
        >>> frame = zscore_normalized_frame  # Your z-scored data
        >>> # Manual thresholds:
        >>> result = categorize_spatial_connected(frame, threshold_dim=1.0, threshold_bright=2.0)
        >>> # Auto thresholds (recommended):
        >>> result = categorize_spatial_connected(frame, threshold_method="multiotsu")
        >>> print(f"Auto thresholds: {result['thresholds']}")
    """
    # Determine thresholds based on method
    if threshold_method == "manual":
        thresh_dim = threshold_dim
        thresh_bright = threshold_bright
    elif threshold_method == "multiotsu":
        # Multi-Otsu finds optimal thresholds for n classes
        thresholds = threshold_multiotsu(image, classes=3)
        thresh_dim = thresholds[0]  # Lower threshold (background vs dim)
        thresh_bright = thresholds[1]  # Upper threshold (dim vs bright)
    elif threshold_method == "li":
        thresh_dim = threshold_li(image)
        thresh_bright = thresh_dim  # Single threshold, all signal is same category
    elif threshold_method == "otsu":
        thresh_dim = threshold_otsu(image)
        thresh_bright = thresh_dim
    elif threshold_method == "yen":
        thresh_dim = threshold_yen(image)
        thresh_bright = thresh_dim
    elif threshold_method == "li_double":
        # Two-pass Li: 1st separates background, 2nd separates dim/bright
        thresh_dim = threshold_li(image)
        signal_pixels = image[image > thresh_dim]
        if len(signal_pixels) > 0:
            thresh_bright = threshold_li(signal_pixels)
        else:
            thresh_bright = thresh_dim
    elif threshold_method == "otsu_double":
        # Two-pass Otsu: 1st separates background, 2nd separates dim/bright
        thresh_dim = threshold_otsu(image)
        signal_pixels = image[image > thresh_dim]
        if len(signal_pixels) > 0:
            thresh_bright = threshold_otsu(signal_pixels)
        else:
            thresh_bright = thresh_dim
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")

    # Find pixels above threshold
    signal_mask = image > thresh_dim

    # Find spatially connected regions (8-connectivity)
    labeled_regions, num_regions = label(signal_mask)

    # Filter regions by size and categorize
    categorized = np.zeros_like(image, dtype=int)
    valid_regions = np.zeros_like(image, dtype=int)
    region_stats = []

    valid_region_id = 1
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_size = np.sum(region_mask)

        # Skip small regions (likely noise)
        if region_size < min_region_size:
            continue

        # Calculate region statistics
        region_pixels = image[region_mask]
        mean_intensity = np.mean(region_pixels)
        max_intensity = np.max(region_pixels)
        min_intensity = np.min(region_pixels)

        # Store region
        valid_regions[region_mask] = valid_region_id

        # Categorize based on mean intensity
        if mean_intensity > thresh_bright:
            category = 2  # Bright
            category_name = "bright"
        else:
            category = 1  # Dim
            category_name = "dim"

        categorized[region_mask] = category

        # Store statistics
        region_stats.append({
            'region_id': valid_region_id,
            'size': region_size,
            'mean_z': mean_intensity,
            'max_z': max_intensity,
            'min_z': min_intensity,
            'category': category,
            'category_name': category_name
        })

        valid_region_id += 1

    return {
        'categorized': categorized,
        'regions': valid_regions,
        'stats': region_stats,
        'num_regions': len(region_stats),
        'thresholds': (thresh_dim, thresh_bright)
    }


def categorize_spatial_morphological(
    image: np.ndarray,
    threshold_dim: float = 0.5,
    threshold_bright: float = 1.5,
    cleanup_size: int = 2
) -> np.ndarray:
    """
    Quick spatial cleanup using morphological operations.

    This is the FASTEST spatial method - good for enhancing existing k-means results.

    Process:
    1. Threshold into categories
    2. Morphological opening: Remove isolated pixels
    3. Morphological closing: Fill small holes

    Args:
        image: 2D z-scored array
        threshold_dim: Z-score threshold for dim signal
        threshold_bright: Z-score threshold for bright signal
        cleanup_size: Size of morphological structuring element (default: 2)

    Returns:
        categorized: 2D array (0=background, 1=dim, 2=bright)

    Example:
        >>> # Add after your k-means to clean up results
        >>> kmeans_result = apply_kmeans_to_frame(frame)
        >>> cleaned = categorize_spatial_morphological(frame)
    """
    # Initial categorization
    categorized = np.zeros_like(image, dtype=int)
    categorized[image > threshold_dim] = 1
    categorized[image > threshold_bright] = 2

    # Create structuring element
    struct = generate_binary_structure(2, 2)  # 8-connectivity
    if cleanup_size > 1:
        struct = binary_dilation(struct, iterations=cleanup_size - 1)

    # Clean up each category
    dim_mask = categorized == 1
    bright_mask = categorized == 2

    # Opening: erosion followed by dilation (removes small objects)
    dim_cleaned = binary_erosion(dim_mask, structure=struct)
    dim_cleaned = binary_dilation(dim_cleaned, structure=struct)

    bright_cleaned = binary_erosion(bright_mask, structure=struct)
    bright_cleaned = binary_dilation(bright_cleaned, structure=struct)

    # Closing: dilation followed by erosion (fills small holes)
    dim_cleaned = binary_dilation(dim_cleaned, structure=struct)
    dim_cleaned = binary_erosion(dim_cleaned, structure=struct)

    bright_cleaned = binary_dilation(bright_cleaned, structure=struct)
    bright_cleaned = binary_erosion(bright_cleaned, structure=struct)

    # Combine
    result = np.zeros_like(image, dtype=int)
    result[dim_cleaned] = 1
    result[bright_cleaned] = 2

    return result


def process_segment_spatial(
    image_segment: np.ndarray,
    method: str = "connected",
    plot: bool = False,
    **kwargs
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Process entire image segment (multiple frames) using spatial methods.

    This is a replacement for process_segment_kmeans() from kmeans.py

    Args:
        image_segment: 3D array (frames, height, width) or list of 2D arrays
        method: 'connected' or 'morphological'
        plot: If True, display visualization of results
        **kwargs: Additional parameters for the chosen method

    Returns:
        categorized_frames: List of categorized frames
        frame_stats: List of statistics dicts (one per frame)

    Example:
        >>> # Replace your k-means call:
        >>> # clustered, centers = process_segment_kmeans(segment)
        >>> # With:
        >>> clustered, stats = process_segment_spatial(segment, method='connected')
    """
    if not isinstance(image_segment, list):
        # Convert 3D array to list of 2D frames
        frames = [image_segment[i] for i in range(image_segment.shape[0])]
    else:
        frames = image_segment

    categorized_frames = []
    all_stats = []

    thresholds_used = None  # Track thresholds for plotting

    for frame_idx, frame in enumerate(frames):
        if method == "connected":
            result = categorize_spatial_connected(frame, **kwargs)
            categorized_frames.append(result['categorized'])
            thresholds_used = result['thresholds']  # Get thresholds from result
            all_stats.append({
                'frame_idx': frame_idx,
                'num_regions': result['num_regions'],
                'region_details': result['stats'],
                'thresholds': result['thresholds']
            })

        elif method == "morphological":
            categorized = categorize_spatial_morphological(frame, **kwargs)
            categorized_frames.append(categorized)
            # Simple stats for morphological method
            all_stats.append({
                'frame_idx': frame_idx,
                'dim_pixels': np.sum(categorized == 1),
                'bright_pixels': np.sum(categorized == 2)
            })

        else:
            raise ValueError(f"Unknown method: {method}. Use 'connected' or 'morphological'")

    # Plot if requested
    if plot:
        n_frames = len(frames)
        fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))

        # Custom colormap: black=background, green=dim, yellow=bright
        cmap_cat = ListedColormap(['black', 'green', 'yellow'])

        # Calculate global vmin/vmax for consistent z-score display
        all_data = np.concatenate([f.flatten() for f in frames])
        vmin, vmax = np.percentile(all_data, [1, 99])

        for i, (orig, cat) in enumerate(zip(frames, categorized_frames)):
            # Top row: original z-scored
            ax_orig = axes[0, i] if n_frames > 1 else axes[0]
            ax_orig.imshow(orig, cmap='gray', vmin=vmin, vmax=vmax)
            ax_orig.set_title(f'Frame {i}\n(Z-score)', fontsize=9)
            ax_orig.axis('off')

            # Bottom row: categorized
            ax_cat = axes[1, i] if n_frames > 1 else axes[1]
            ax_cat.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)
            n_regions = all_stats[i].get('num_regions', 0)
            ax_cat.set_title(f'Categorized\n({n_regions} regions)', fontsize=9)
            ax_cat.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', edgecolor='white', label='Background'),
            Patch(facecolor='green', label='Dim'),
            Patch(facecolor='yellow', label='Bright')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Build title with threshold info
        title = 'Spatial Categorization Results'
        if thresholds_used is not None:
            thresh_dim, thresh_bright = thresholds_used
            title += f'\nThresholds: dim>{thresh_dim:.2f}, bright>{thresh_bright:.2f}'
        plt.suptitle(title, fontweight='bold')
        plt.tight_layout()
        plt.show()

    return categorized_frames, all_stats


def apply_spatial_to_concatenated(
    frames: list[np.ndarray],
    threshold_dim: float = 0.5,
    threshold_bright: float = 1.5,
    min_region_size: int = 50
) -> tuple[list[np.ndarray], dict]:
    """
    Spatial version of process_segment_kmeans_concatenated().

    Instead of concatenating frames, this finds regions that appear
    consistently across multiple frames (spatial + temporal consistency).

    Args:
        frames: List of 2D arrays (frames to analyze)
        threshold_dim: Z-score threshold for dim signal
        threshold_bright: Z-score threshold for bright signal
        min_region_size: Minimum pixels per region

    Returns:
        categorized_frames: List of categorized frames
        consistency_stats: Dict with cross-frame statistics
    """
    n_frames = len(frames)

    # Find regions in each frame
    frame_results = []
    for frame in frames:
        result = categorize_spatial_connected(
            frame,
            threshold_dim=threshold_dim,
            threshold_bright=threshold_bright,
            min_region_size=min_region_size
        )
        frame_results.append(result)

    # Find spatially consistent regions across frames
    # (Pixels that are active in same location across multiple frames)
    activation_map = np.zeros_like(frames[0], dtype=int)
    for result in frame_results:
        activation_map[result['categorized'] > 0] += 1

    # Pixels active in >50% of frames are considered consistent
    consistent_threshold = n_frames // 2
    consistent_mask = activation_map >= consistent_threshold

    # Apply consistency filter to each frame
    filtered_frames = []
    for result in frame_results:
        filtered = result['categorized'].copy()
        filtered[~consistent_mask] = 0  # Remove inconsistent pixels
        filtered_frames.append(filtered)

    # Statistics
    stats = {
        'total_frames': n_frames,
        'consistent_threshold': consistent_threshold,
        'num_consistent_pixels': np.sum(consistent_mask),
        'consistency_map': activation_map
    }

    return filtered_frames, stats


# ============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK INTEGRATION
# ============================================================================

def quick_spatial_categorize(zscore_frame: np.ndarray) -> np.ndarray:
    """
    Simplest function: Just give me categorized frame!

    Uses sensible defaults for ACh imaging.

    Args:
        zscore_frame: Your z-score normalized frame

    Returns:
        categorized: 0=background, 1=dim ACh, 2=bright ACh
    """
    return categorize_spatial_connected(
        zscore_frame,
        threshold_dim=0.5,
        threshold_bright=1.5,
        min_region_size=20
    )['categorized']


def compare_kmeans_vs_spatial(frame: np.ndarray) -> dict:
    """
    Compare k-means (your current method) vs spatial method.

    Returns dict with both results for comparison.
    """
    from .kmeans import apply_kmeans_to_frame

    # K-means (current)
    kmeans_result, _ = apply_kmeans_to_frame(frame, n_clusters=3)

    # Spatial (new)
    spatial_result = categorize_spatial_connected(frame)

    # Calculate differences
    different_pixels = np.sum(kmeans_result != spatial_result['categorized'])
    total_pixels = frame.size

    return {
        'kmeans': kmeans_result,
        'spatial': spatial_result['categorized'],
        'difference_pixels': different_pixels,
        'difference_percent': different_pixels / total_pixels * 100,
        'spatial_regions': spatial_result['num_regions'],
        'spatial_stats': spatial_result['stats']
    }
