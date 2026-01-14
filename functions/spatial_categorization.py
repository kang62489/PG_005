"""
Spatial-aware intensity categorization for ACh imaging.

This module provides a class-based interface for spatial categorization methods
that consider both intensity AND spatial connectivity.

Methods available:
1. connected: Connected components analysis (fast, simple)
2. watershed: Watershed segmentation (good for overlapping regions)
3. dbscan: DBSCAN clustering (handles complex patterns)
4. morphological: Morphological cleanup (erosion/dilation)
5. region_growing: Region growing from seeds (grows from bright peaks)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_multiotsu, threshold_otsu, threshold_yen
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN


class SpatialCategorizer:
    """
    Spatial-aware intensity categorization for image segments.

    This class provides multiple methods for categorizing pixels into
    background/dim/bright while considering spatial connectivity.

    Attributes:
        method: Categorization method ('connected', 'watershed', 'dbscan', 'morphological', 'region_growing')
        threshold_method: Auto-thresholding method ('manual', 'multiotsu', 'li_double', etc.)
        threshold_dim: Manual threshold for dim signal
        threshold_bright: Manual threshold for bright signal
        min_region_size: Minimum pixels per region
        global_threshold: Whether to use global thresholds across all frames

    Example:
        >>> categorizer = SpatialCategorizer(method="connected", threshold_method="li_double")
        >>> categorizer.fit(image_segment)  # 3D array (frames, H, W)
        >>> categorizer.plot()
    """

    METHODS = ["connected", "watershed", "dbscan", "morphological", "region_growing"]
    THRESHOLD_METHODS = ["manual", "multiotsu", "li_double", "otsu_double", "li", "otsu", "yen"]

    def __init__(
        self,
        method: str = "connected",
        threshold_method: str = "li_double",
        threshold_dim: float = 0.5,
        threshold_bright: float = 1.5,
        min_region_size: int = 20,
        global_threshold: bool = True,
        # Watershed parameters
        min_distance: int = 10,
        # DBSCAN parameters
        eps: float = 3.0,
        min_samples: int = 10,
        intensity_scale: float = 10.0,
        # Morphological parameters
        kernel_size: int = 3,
        # Region growing parameters
        seed_threshold: float = 2.0,
        growth_threshold: float = 0.5,
        max_diff: float = 0.5,
    ) -> None:
        """Initialize the SpatialCategorizer."""
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {self.METHODS}")
        if threshold_method not in self.THRESHOLD_METHODS:
            raise ValueError(f"Unknown threshold_method: {threshold_method}. Choose from {self.THRESHOLD_METHODS}")

        self.method = method
        self.threshold_method = threshold_method
        self.threshold_dim = threshold_dim
        self.threshold_bright = threshold_bright
        self.min_region_size = min_region_size
        self.global_threshold = global_threshold

        # Method-specific parameters
        self.min_distance = min_distance  # watershed
        self.eps = eps  # dbscan
        self.min_samples = min_samples  # dbscan
        self.intensity_scale = intensity_scale  # dbscan
        self.kernel_size = kernel_size  # morphological
        self.seed_threshold = seed_threshold  # region_growing
        self.growth_threshold = growth_threshold  # region_growing
        self.max_diff = max_diff  # region_growing

        # Results (populated after fit)
        self.frames: list[np.ndarray] = []
        self.categorized_frames: list[np.ndarray] = []
        self.frame_stats: list[dict] = []
        self.thresholds_used: tuple | None = None

    def fit(self, image_segment: np.ndarray) -> "SpatialCategorizer":
        """
        Fit the categorizer to an image segment.

        Args:
            image_segment: 3D array (frames, height, width) or 2D array (single frame)

        Returns:
            self (for method chaining)
        """
        # Convert to list of frames
        if image_segment.ndim == 2:
            self.frames = [image_segment]
        else:
            self.frames = [image_segment[i] for i in range(image_segment.shape[0])]

        # Calculate global thresholds if needed
        if self.global_threshold and self.threshold_method != "manual":
            self._calculate_global_thresholds()
        else:
            self.thresholds_used = (self.threshold_dim, self.threshold_bright)

        # Process each frame
        self.categorized_frames = []
        self.frame_stats = []

        for frame_idx, frame in enumerate(self.frames):
            categorized, stats = self._categorize_frame(frame, frame_idx)
            self.categorized_frames.append(categorized)
            self.frame_stats.append(stats)

        return self

    def _calculate_global_thresholds(self) -> None:
        """Calculate thresholds using all frames combined."""
        all_pixels = np.concatenate([f.flatten() for f in self.frames])

        if self.threshold_method == "multiotsu":
            thresholds = threshold_multiotsu(all_pixels, classes=3)
            self.thresholds_used = (thresholds[0], thresholds[1])
        elif self.threshold_method == "li_double":
            thresh_dim = threshold_li(all_pixels)
            signal_pixels = all_pixels[all_pixels > thresh_dim]
            thresh_bright = threshold_li(signal_pixels) if len(signal_pixels) > 0 else thresh_dim
            self.thresholds_used = (thresh_dim, thresh_bright)
        elif self.threshold_method == "otsu_double":
            thresh_dim = threshold_otsu(all_pixels)
            signal_pixels = all_pixels[all_pixels > thresh_dim]
            thresh_bright = threshold_otsu(signal_pixels) if len(signal_pixels) > 0 else thresh_dim
            self.thresholds_used = (thresh_dim, thresh_bright)
        elif self.threshold_method == "li":
            thresh = threshold_li(all_pixels)
            self.thresholds_used = (thresh, thresh)
        elif self.threshold_method == "otsu":
            thresh = threshold_otsu(all_pixels)
            self.thresholds_used = (thresh, thresh)
        elif self.threshold_method == "yen":
            thresh = threshold_yen(all_pixels)
            self.thresholds_used = (thresh, thresh)
        else:
            self.thresholds_used = (self.threshold_dim, self.threshold_bright)

    def _categorize_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[np.ndarray, dict]:
        """Categorize a single frame using the selected method."""
        thresh_dim, thresh_bright = self.thresholds_used

        if self.method == "connected":
            return self._method_connected(frame, frame_idx, thresh_dim, thresh_bright)
        elif self.method == "watershed":
            return self._method_watershed(frame, frame_idx, thresh_dim, thresh_bright)
        elif self.method == "dbscan":
            return self._method_dbscan(frame, frame_idx, thresh_dim, thresh_bright)
        elif self.method == "morphological":
            return self._method_morphological(frame, frame_idx, thresh_dim, thresh_bright)
        elif self.method == "region_growing":
            return self._method_region_growing(frame, frame_idx, thresh_dim, thresh_bright)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _method_connected(
        self, frame: np.ndarray, frame_idx: int, thresh_dim: float, thresh_bright: float
    ) -> tuple[np.ndarray, dict]:
        """Connected components analysis."""
        signal_mask = frame > thresh_dim
        labeled_regions, num_regions = label(signal_mask)

        categorized = np.zeros_like(frame, dtype=int)
        region_stats = []
        valid_region_id = 1

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)

            if region_size < self.min_region_size:
                continue

            region_pixels = frame[region_mask]
            mean_intensity = np.mean(region_pixels)

            if mean_intensity > thresh_bright:
                category = 2
                category_name = "bright"
            else:
                category = 1
                category_name = "dim"

            categorized[region_mask] = category
            region_stats.append({
                "region_id": valid_region_id,
                "size": region_size,
                "mean_z": mean_intensity,
                "max_z": np.max(region_pixels),
                "category": category,
                "category_name": category_name,
            })
            valid_region_id += 1

        return categorized, {
            "frame_idx": frame_idx,
            "num_regions": len(region_stats),
            "region_details": region_stats,
            "thresholds": self.thresholds_used,
        }

    def _method_watershed(
        self, frame: np.ndarray, frame_idx: int, thresh_dim: float, thresh_bright: float
    ) -> tuple[np.ndarray, dict]:
        """Watershed segmentation."""
        smoothed = ndimage.gaussian_filter(frame, sigma=1.5)
        mask = smoothed > thresh_dim
        distance = ndimage.distance_transform_edt(mask)

        local_max = peak_local_max(distance, min_distance=self.min_distance, labels=mask, exclude_border=False)

        markers = np.zeros_like(frame, dtype=int)
        for idx, (y, x) in enumerate(local_max):
            markers[y, x] = idx + 1

        labels = watershed(-smoothed, markers, mask=mask)

        categorized = np.zeros_like(frame, dtype=int)
        region_stats = []

        for region_id in np.unique(labels):
            if region_id == 0:
                continue
            region_mask = labels == region_id
            region_size = np.sum(region_mask)

            if region_size < self.min_region_size:
                continue

            mean_intensity = np.mean(frame[region_mask])

            if mean_intensity > thresh_bright:
                categorized[region_mask] = 2
                category_name = "bright"
            elif mean_intensity > thresh_dim:
                categorized[region_mask] = 1
                category_name = "dim"
            else:
                continue

            region_stats.append({
                "region_id": region_id,
                "size": region_size,
                "mean_z": mean_intensity,
                "category_name": category_name,
            })

        return categorized, {
            "frame_idx": frame_idx,
            "num_regions": len(region_stats),
            "num_peaks": len(local_max),
            "region_details": region_stats,
            "thresholds": self.thresholds_used,
        }

    def _method_dbscan(
        self, frame: np.ndarray, frame_idx: int, thresh_dim: float, thresh_bright: float
    ) -> tuple[np.ndarray, dict]:
        """DBSCAN spatial clustering."""
        mask = frame > thresh_dim
        coords = np.argwhere(mask)
        intensities = frame[mask].reshape(-1, 1)

        if len(coords) == 0:
            return np.zeros_like(frame, dtype=int), {
                "frame_idx": frame_idx,
                "num_regions": 0,
                "region_details": [],
                "thresholds": self.thresholds_used,
            }

        features = np.hstack([coords, intensities * self.intensity_scale])
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
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

            if cluster_size < self.min_region_size:
                continue

            mean_intensity = np.mean(frame[cluster_mask])

            if mean_intensity > thresh_bright:
                categorized[cluster_mask] = 2
                category_name = "bright"
            else:
                categorized[cluster_mask] = 1
                category_name = "dim"

            region_stats.append({
                "cluster_id": cluster_id,
                "size": cluster_size,
                "mean_z": mean_intensity,
                "category_name": category_name,
            })

        return categorized, {
            "frame_idx": frame_idx,
            "num_regions": len(region_stats),
            "num_noise_pixels": np.sum(cluster_labels == -1),
            "region_details": region_stats,
            "thresholds": self.thresholds_used,
        }

    def _method_morphological(
        self, frame: np.ndarray, frame_idx: int, thresh_dim: float, thresh_bright: float
    ) -> tuple[np.ndarray, dict]:
        """Morphological cleanup."""
        categorized = np.zeros_like(frame, dtype=int)
        categorized[frame > thresh_dim] = 1
        categorized[frame > thresh_bright] = 2

        struct = generate_binary_structure(2, 2)
        if self.kernel_size > 1:
            struct = ndimage.iterate_structure(struct, self.kernel_size)

        dim_mask = categorized == 1
        bright_mask = categorized == 2

        # Opening then closing for each category
        dim_cleaned = binary_erosion(dim_mask, structure=struct)
        dim_cleaned = binary_dilation(dim_cleaned, structure=struct)
        dim_cleaned = binary_dilation(dim_cleaned, structure=struct)
        dim_cleaned = binary_erosion(dim_cleaned, structure=struct)

        bright_cleaned = binary_erosion(bright_mask, structure=struct)
        bright_cleaned = binary_dilation(bright_cleaned, structure=struct)
        bright_cleaned = binary_dilation(bright_cleaned, structure=struct)
        bright_cleaned = binary_erosion(bright_cleaned, structure=struct)

        result = np.zeros_like(frame, dtype=int)
        result[dim_cleaned] = 1
        result[bright_cleaned] = 2

        return result, {
            "frame_idx": frame_idx,
            "num_regions": 0,  # Not tracked for morphological
            "before_dim": np.sum(dim_mask),
            "after_dim": np.sum(dim_cleaned),
            "before_bright": np.sum(bright_mask),
            "after_bright": np.sum(bright_cleaned),
            "thresholds": self.thresholds_used,
        }

    def _method_region_growing(
        self, frame: np.ndarray, frame_idx: int, thresh_dim: float, thresh_bright: float
    ) -> tuple[np.ndarray, dict]:
        """Region growing from seeds."""
        seeds = frame > self.seed_threshold
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
                        if neighbor_value > self.growth_threshold and abs(neighbor_value - seed_value) < self.max_diff:
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

            if region_size < self.min_region_size:
                continue

            mean_intensity = np.mean(frame[region_mask])

            if mean_intensity > thresh_bright:
                categorized[region_mask] = 2
                category_name = "bright"
            else:
                categorized[region_mask] = 1
                category_name = "dim"

            region_stats.append({
                "region_id": rid,
                "size": region_size,
                "mean_z": mean_intensity,
                "category_name": category_name,
            })

        return categorized, {
            "frame_idx": frame_idx,
            "num_regions": len(region_stats),
            "num_seeds": num_seeds,
            "region_details": region_stats,
            "thresholds": self.thresholds_used,
        }

    def plot(self, figsize_per_frame: tuple[int, int] = (3, 6)) -> plt.Figure:
        """
        Plot the categorization results.

        Args:
            figsize_per_frame: Figure size per frame (width, height)

        Returns:
            matplotlib Figure object
        """
        if not self.categorized_frames:
            raise RuntimeError("No results to plot. Call fit() first.")

        n_frames = len(self.frames)
        fig, axes = plt.subplots(2, n_frames, figsize=(figsize_per_frame[0] * n_frames, figsize_per_frame[1]))

        # Handle single frame case
        if n_frames == 1:
            axes = axes.reshape(2, 1)

        cmap_cat = ListedColormap(["black", "green", "yellow"])

        all_data = np.concatenate([f.flatten() for f in self.frames])
        vmin, vmax = np.percentile(all_data, [1, 99])

        for i, (orig, cat) in enumerate(zip(self.frames, self.categorized_frames)):
            # Top row: original z-scored
            axes[0, i].imshow(orig, cmap="gray", vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f"Frame {i}\n(Z-score)", fontsize=9)
            axes[0, i].axis("off")

            # Bottom row: categorized
            axes[1, i].imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)
            n_regions = self.frame_stats[i].get("num_regions", 0)
            axes[1, i].set_title(f"Categorized\n({n_regions} regions)", fontsize=9)
            axes[1, i].axis("off")

        # Legend
        legend_elements = [
            Patch(facecolor="black", edgecolor="white", label="Background"),
            Patch(facecolor="green", label="Dim"),
            Patch(facecolor="yellow", label="Bright"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

        # Title
        title = f"Spatial Categorization: {self.method.upper()}"
        if self.thresholds_used:
            thresh_dim, thresh_bright = self.thresholds_used
            title += f"\nThresholds: dim>{thresh_dim:.2f}, bright>{thresh_bright:.2f}"
        title += f" (method: {self.threshold_method})"
        plt.suptitle(title, fontweight="bold")
        plt.tight_layout()

        return fig

    def show(self) -> None:
        """Plot and display the results."""
        self.plot()
        plt.show()

    def get_results(self) -> dict:
        """
        Get all results as a dictionary.

        Returns:
            dict with categorized_frames, frame_stats, thresholds_used
        """
        return {
            "categorized_frames": self.categorized_frames,
            "frame_stats": self.frame_stats,
            "thresholds_used": self.thresholds_used,
            "method": self.method,
            "threshold_method": self.threshold_method,
        }


# Convenience function for backward compatibility
def process_segment_spatial(
    image_segment: np.ndarray,
    method: str = "connected",
    plot: bool = False,
    global_threshold: bool = True,
    **kwargs,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Process image segment using spatial methods (backward compatible function).

    Args:
        image_segment: 3D array (frames, height, width)
        method: Categorization method
        plot: If True, display visualization
        global_threshold: If True, use global thresholds
        **kwargs: Additional parameters

    Returns:
        categorized_frames, frame_stats
    """
    categorizer = SpatialCategorizer(method=method, global_threshold=global_threshold, **kwargs)
    categorizer.fit(image_segment)

    if plot:
        categorizer.show()

    return categorizer.categorized_frames, categorizer.frame_stats
