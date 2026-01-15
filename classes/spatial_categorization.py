"""
Spatial-aware intensity categorization for ACh imaging.

This module provides a class-based interface for spatial categorization methods
that consider both intensity AND spatial connectivity.

Methods available:
1. connected: Connected components analysis (fast, simple)
2. watershed: Watershed segmentation (good for overlapping regions)
3. morphological: Morphological cleanup (erosion/dilation)
"""

from typing import ClassVar

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_multiotsu, threshold_otsu
from skimage.segmentation import watershed

# Constants
NDIM_SINGLE_FRAME = 2
CATEGORY_DIM = 1
CATEGORY_BRIGHT = 2


class SpatialCategorizer:
    """
    Spatial-aware intensity categorization for image segments.

    This class provides multiple methods for categorizing pixels into
    background/dim/bright while considering spatial connectivity.

    Attributes:
        method: Categorization method ('connected', 'watershed', 'morphological')
        threshold_method: Auto-thresholding method ('manual', 'multiotsu', 'li_double', 'otsu_double')
        threshold_dim: Manual threshold for dim signal
        threshold_bright: Manual threshold for bright signal
        min_region_size: Minimum pixels per region
        global_threshold: Whether to use global thresholds across all frames

    Example:
        >>> categorizer = SpatialCategorizer.connected(min_region_size=30)
        >>> categorizer.fit(image_segment)  # 3D array (frames, H, W)
        >>> categorizer.show()

        >>> categorizer = SpatialCategorizer.watershed(min_distance=5, min_region_size=20)
        >>> categorizer.fit(image_segment).show()

        >>> categorizer = SpatialCategorizer.morphological(kernel_size=5)
        >>> categorizer.fit(image_segment).show()
    """

    METHODS: ClassVar[list[str]] = ["connected", "watershed", "morphological"]
    THRESHOLD_METHODS: ClassVar[list[str]] = ["manual", "multiotsu", "li_double", "otsu_double"]

    def __init__(
        self,
        method: str,
        threshold_method: str = "li_double",
        *,
        global_threshold: bool = True,
        # Manual threshold (only when threshold_method="manual")
        threshold_dim: float | None = None,
        threshold_bright: float | None = None,
        # Connected/Watershed parameters
        min_region_size: int = 20,
        # Watershed parameters
        min_distance: int = 10,
        # Morphological parameters
        kernel_size: int = 3,
    ) -> None:
        """
        Initialize the SpatialCategorizer.

        Prefer using factory methods for clarity:
            - SpatialCategorizer.connected(...)
            - SpatialCategorizer.watershed(...)
            - SpatialCategorizer.morphological(...)
        """
        if method not in self.METHODS:
            msg = f"Unknown method: {method}. Choose from {self.METHODS}"
            raise ValueError(msg)
        if threshold_method not in self.THRESHOLD_METHODS:
            msg = f"Unknown threshold_method: {threshold_method}. Choose from {self.THRESHOLD_METHODS}"
            raise ValueError(msg)

        # Validate manual thresholds
        if threshold_method == "manual":
            if threshold_dim is None or threshold_bright is None:
                msg = "threshold_dim and threshold_bright are required when threshold_method='manual'"
                raise ValueError(msg)
        else:
            # Use placeholder values (will be overwritten by auto-calculation)
            if threshold_dim is None:
                threshold_dim = 0.0
            if threshold_bright is None:
                threshold_bright = 0.0

        self.method = method
        self.threshold_method = threshold_method
        self.threshold_dim = threshold_dim
        self.threshold_bright = threshold_bright
        self.min_region_size = min_region_size
        self.global_threshold = global_threshold

        # Method-specific parameters
        self.min_distance = min_distance  # watershed
        self.kernel_size = kernel_size  # morphological

        # Results (populated after fit)
        self.frames: list[np.ndarray] = []
        self.categorized_frames: list[np.ndarray] = []
        self.frame_stats: list[dict] = []
        self.thresholds_used: tuple | None = None

    @classmethod
    def connected(
        cls,
        threshold_method: str = "li_double",
        *,
        global_threshold: bool = True,
        threshold_dim: float | None = None,
        threshold_bright: float | None = None,
        min_region_size: int = 20,
    ) -> "SpatialCategorizer":
        """
        Create a SpatialCategorizer using connected components method.

        Args:
            threshold_method: 'manual', 'multiotsu', 'li_double', or 'otsu_double'
            global_threshold: Use global thresholds across all frames
            threshold_dim: Manual dim threshold (required if threshold_method='manual')
            threshold_bright: Manual bright threshold (required if threshold_method='manual')
            min_region_size: Minimum pixels per region (smaller regions are removed)

        Returns:
            SpatialCategorizer instance
        """
        return cls(
            method="connected",
            threshold_method=threshold_method,
            global_threshold=global_threshold,
            threshold_dim=threshold_dim,
            threshold_bright=threshold_bright,
            min_region_size=min_region_size,
        )

    @classmethod
    def watershed(
        cls,
        threshold_method: str = "li_double",
        *,
        global_threshold: bool = True,
        threshold_dim: float | None = None,
        threshold_bright: float | None = None,
        min_region_size: int = 20,
        min_distance: int = 10,
    ) -> "SpatialCategorizer":
        """
        Create a SpatialCategorizer using watershed segmentation.

        Args:
            threshold_method: 'manual', 'multiotsu', 'li_double', or 'otsu_double'
            global_threshold: Use global thresholds across all frames
            threshold_dim: Manual dim threshold (required if threshold_method='manual')
            threshold_bright: Manual bright threshold (required if threshold_method='manual')
            min_region_size: Minimum pixels per region (smaller regions are removed)
            min_distance: Minimum distance between peaks (larger = fewer regions)

        Returns:
            SpatialCategorizer instance
        """
        return cls(
            method="watershed",
            threshold_method=threshold_method,
            global_threshold=global_threshold,
            threshold_dim=threshold_dim,
            threshold_bright=threshold_bright,
            min_region_size=min_region_size,
            min_distance=min_distance,
        )

    @classmethod
    def morphological(
        cls,
        threshold_method: str = "li_double",
        *,
        global_threshold: bool = True,
        threshold_dim: float | None = None,
        threshold_bright: float | None = None,
        kernel_size: int = 3,
    ) -> "SpatialCategorizer":
        """
        Create a SpatialCategorizer using morphological cleanup.

        Args:
            threshold_method: 'manual', 'multiotsu', 'li_double', or 'otsu_double'
            global_threshold: Use global thresholds across all frames
            threshold_dim: Manual dim threshold (required if threshold_method='manual')
            threshold_bright: Manual bright threshold (required if threshold_method='manual')
            kernel_size: Size of erosion/dilation kernel (larger = more aggressive cleanup)

        Returns:
            SpatialCategorizer instance
        """
        return cls(
            method="morphological",
            threshold_method=threshold_method,
            global_threshold=global_threshold,
            threshold_dim=threshold_dim,
            threshold_bright=threshold_bright,
            kernel_size=kernel_size,
        )

    def fit(self, image_segment: np.ndarray) -> "SpatialCategorizer":
        """
        Fit the categorizer to an image segment.

        Args:
            image_segment: 3D array (frames, height, width) or 2D array (single frame)

        Returns:
            self (for method chaining)
        """
        # Convert to list of frames
        if image_segment.ndim == NDIM_SINGLE_FRAME:
            # input segment only has demention of ndim is 2 => (width, height)
            self.frames = [image_segment]
        else:
            # multiple frames in the segment ndim is 3 or higher => (frames, height, width) or (channels, frames, height, width)
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
        else:
            self.thresholds_used = (self.threshold_dim, self.threshold_bright)

    def _categorize_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[np.ndarray, dict]:
        """Categorize a single frame using the selected method."""
        thresh_dim, thresh_bright = self.thresholds_used

        if self.method == "connected":
            return self._method_connected(frame, frame_idx, thresh_dim, thresh_bright)
        if self.method == "watershed":
            return self._method_watershed(frame, frame_idx, thresh_dim, thresh_bright)
        if self.method == "morphological":
            return self._method_morphological(frame, frame_idx, thresh_dim, thresh_bright)
        msg = f"Unknown method: {self.method}"
        raise ValueError(msg)

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
            region_stats.append(
                {
                    "region_id": valid_region_id,
                    "size": region_size,
                    "mean_z": mean_intensity,
                    "max_z": np.max(region_pixels),
                    "category": category,
                    "category_name": category_name,
                }
            )
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

            region_stats.append(
                {"region_id": region_id, "size": region_size, "mean_z": mean_intensity, "category_name": category_name}
            )

        return categorized, {
            "frame_idx": frame_idx,
            "num_regions": len(region_stats),
            "num_peaks": len(local_max),
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

        dim_mask = categorized == CATEGORY_DIM
        bright_mask = categorized == CATEGORY_BRIGHT

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

    # def plot(self, figsize_per_frame: tuple[int, int] = (3, 6)) -> plt.Figure:
    #     """
    #     Plot the categorization results.

    #     Args:
    #         figsize_per_frame: Figure size per frame (width, height)

    #     Returns:
    #         matplotlib Figure object
    #     """
    #     if not self.categorized_frames:
    #         msg = "No results to plot. Call fit() first."
    #         raise RuntimeError(msg)

    #     n_frames = len(self.frames)
    #     fig, axes = plt.subplots(2, n_frames, figsize=(figsize_per_frame[0] * n_frames, figsize_per_frame[1]))

    #     # Handle single frame case
    #     if n_frames == 1:
    #         axes = axes.reshape(2, 1)

    #     cmap_cat = ListedColormap(["black", "green", "yellow"])

    #     all_data = np.concatenate([f.flatten() for f in self.frames])
    #     vmin, vmax = np.percentile(all_data, [1, 99])

    #     for i, (orig, cat) in enumerate(zip(self.frames, self.categorized_frames, strict=True)):
    #         # Top row: original z-scored
    #         axes[0, i].imshow(orig, cmap="gray", vmin=vmin, vmax=vmax)
    #         axes[0, i].set_title(f"Frame {i}\n(Z-score)", fontsize=9)
    #         axes[0, i].axis("off")

    #         # Bottom row: categorized
    #         axes[1, i].imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)
    #         n_regions = self.frame_stats[i].get("num_regions", 0)
    #         axes[1, i].set_title(f"Categorized\n({n_regions} regions)", fontsize=9)
    #         axes[1, i].axis("off")

    #     # Legend
    #     legend_elements = [
    #         Patch(facecolor="black", edgecolor="white", label="Background"),
    #         Patch(facecolor="green", label="Dim"),
    #         Patch(facecolor="yellow", label="Bright"),
    #     ]
    #     fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

    #     # Title
    #     title = f"Spatial Categorization: {self.method.upper()}"
    #     if self.thresholds_used:
    #         thresh_dim, thresh_bright = self.thresholds_used
    #         title += f"\nThresholds: dim>{thresh_dim:.2f}, bright>{thresh_bright:.2f}"
    #     title += f" (method: {self.threshold_method})"
    #     plt.suptitle(title, fontweight="bold")
    #     plt.tight_layout()

    #     return fig

    # def show(self) -> None:
    #     """Plot and display the results."""
    #     self.plot()
    #     plt.show()

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
    *,
    plot: bool = False,
    global_threshold: bool = True,
    **kwargs: float,
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
