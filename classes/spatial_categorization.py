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
from skimage.segmentation import watershed

# Constants
NDIM_SINGLE_FRAME = 2
CATEGORY_BRIGHT = 1
BASELINE_SIGMA_MULT = 3.0


class SpatialCategorizer:
    """
    Spatial-aware intensity categorization for image segments.

    This class provides multiple methods for categorizing pixels into
    background/bright while considering spatial connectivity.

    Attributes:
        method: Categorization method ('connected', 'watershed', 'morphological')
        threshold_method: Auto-thresholding method ('baseline_frames_2sigma')
        min_region_size: Minimum pixels per region

    Example:
        >>> categorizer = SpatialCategorizer.connected(min_region_size=30)
        >>> categorizer.fit(image_segment, spike_frame_idx=spike_frame_idx)  # 3D array (frames, H, W)
        >>> results = categorizer.get_results()

        >>> categorizer = SpatialCategorizer.watershed(min_distance=5, min_region_size=20)
        >>> categorizer.fit(image_segment, spike_frame_idx=spike_frame_idx)

        >>> categorizer = SpatialCategorizer.morphological(kernel_size=5)
        >>> categorizer.fit(image_segment, spike_frame_idx=spike_frame_idx)
    """

    GROUPING_METHODS: ClassVar[list[str]] = ["connected", "watershed", "morphological"]
    THRESHOLD_METHODS: ClassVar[list[str]] = ["baseline_frames_2sigma"]

    def __init__(
        self,
        grouping_method: str,
        threshold_method: str = "baseline_frames_2sigma",
        *,
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
        if grouping_method not in self.GROUPING_METHODS:
            msg = f"Unknown grouping_method: {grouping_method}. Choose from {self.GROUPING_METHODS}"
            raise ValueError(msg)
        if threshold_method not in self.THRESHOLD_METHODS:
            msg = f"Unknown threshold_method: {threshold_method}. Choose from {self.THRESHOLD_METHODS}"
            raise ValueError(msg)

        self.grouping_method = grouping_method
        self.threshold_method = threshold_method
        self.min_region_size = min_region_size

        # Method-specific parameters
        self.min_distance = min_distance  # watershed
        self.kernel_size = kernel_size  # morphological

        # Results (populated after fit)
        self.source_frames: list[np.ndarray] = []
        self.categorized_frames: list[np.ndarray] = []
        self.frame_regions: list[dict] = []
        self.threshold_used: float | None = None

    @classmethod
    def connected(
        cls,
        threshold_method: str = "baseline_frames_2sigma",
        *,
        min_region_size: int = 20,
    ) -> "SpatialCategorizer":
        """
        Create a SpatialCategorizer using connected components method.

        Args:
            threshold_method: 'baseline_frames_2sigma'
            min_region_size: Minimum pixels per region (smaller regions are removed)

        Returns:
            SpatialCategorizer instance
        """
        return cls(
            grouping_method="connected",
            threshold_method=threshold_method,
            min_region_size=min_region_size,
        )

    @classmethod
    def watershed(
        cls,
        threshold_method: str = "baseline_frames_2sigma",
        *,
        min_region_size: int = 20,
        min_distance: int = 10,
    ) -> "SpatialCategorizer":
        """
        Create a SpatialCategorizer using watershed segmentation.

        Args:
            threshold_method: 'baseline_frames_2sigma'
            min_region_size: Minimum pixels per region (smaller regions are removed)
            min_distance: Minimum distance between peaks (larger = fewer regions)

        Returns:
            SpatialCategorizer instance
        """
        return cls(
            grouping_method="watershed",
            threshold_method=threshold_method,
            min_region_size=min_region_size,
            min_distance=min_distance,
        )

    @classmethod
    def morphological(
        cls,
        threshold_method: str = "baseline_frames_2sigma",
        *,
        kernel_size: int = 3,
    ) -> "SpatialCategorizer":
        """
        Create a SpatialCategorizer using morphological cleanup.

        Args:
            threshold_method: 'baseline_frames_2sigma'
            kernel_size: Size of erosion/dilation kernel (larger = more aggressive cleanup)

        Returns:
            SpatialCategorizer instance
        """
        return cls(
            grouping_method="morphological",
            threshold_method=threshold_method,
            kernel_size=kernel_size,
        )

    def fit(self, image_segment: np.ndarray, spike_frame_idx: int) -> "SpatialCategorizer":
        """
        Fit the categorizer to an image segment.

        Args:
            image_segment: 3D array (frames, height, width) or 2D array (single frame)
            spike_frame_idx: Index of the spike frame within image_segment. Frames
                before this index are treated as the baseline window used to set
                threshold_used (baseline mean + 2*std).

        Returns:
            self (for method chaining)
        """
        self.source_frames = [image_segment[i] for i in range(image_segment.shape[0])]

        self._calculate_global_threshold(spike_frame_idx)

        # Process each frame
        self.categorized_frames = []
        self.frame_regions = []

        for frame_idx, frame in enumerate(self.source_frames):
            categorized, stats = self._dispatch_frame(frame, frame_idx)
            self.categorized_frames.append(categorized)
            self.frame_regions.append(stats)

        return self

    def _calculate_global_threshold(self, spike_frame_idx: int) -> None:
        """Calculate the bright threshold from the baseline (pre-spike) frames."""
        self.threshold_used = self.compute_baseline_threshold(self.source_frames[:spike_frame_idx])

    @staticmethod
    def compute_baseline_threshold(baseline_frames: list[np.ndarray]) -> float:
        """Bright threshold (mean + 2*std) from a set of baseline (pre-spike) frames.

        Pulled out of _calculate_global_threshold so a caller that already has a segment's
        baseline frames on hand (e.g. a per-segment reliability check) can compute the same
        threshold without fitting a full SpatialCategorizer instance first.

        Args:
            baseline_frames: pre-spike frames, e.g. image_segment[:spike_frame_idx].

        Returns:
            Bright-pixel threshold: baseline mean + BASELINE_SIGMA_MULT * baseline std.
        """
        baseline_pixels = np.concatenate([np.asarray(f).flatten() for f in baseline_frames])
        return float(baseline_pixels.mean() + BASELINE_SIGMA_MULT * baseline_pixels.std())

    def categorize_frame(self, frame: np.ndarray, frame_idx: int, threshold: float) -> np.ndarray:
        """Categorize a single frame using an already-known threshold.

        Unlike fit(), this doesn't run the full per-frame loop over every frame in a
        segment -- useful when only 1-2 specific frames (e.g. spike/spike+1 for a
        reliability check) actually need categorizing, not the whole segment.

        Args:
            frame: 2D array to categorize.
            frame_idx: index of this frame (bookkeeping only, see _dispatch_frame).
            threshold: bright-pixel threshold, e.g. from compute_baseline_threshold().

        Returns:
            Categorized frame (0=background, 1=bright).
        """
        self.threshold_used = threshold
        categorized, _ = self._dispatch_frame(frame, frame_idx)
        return categorized

    def _dispatch_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[np.ndarray, dict]:
        thresh_bright = self.threshold_used

        if self.grouping_method == "connected":
            return self._apply_connected(frame, frame_idx, thresh_bright)
        if self.grouping_method == "watershed":
            return self._apply_watershed(frame, frame_idx, thresh_bright)
        if self.grouping_method == "morphological":
            return self._apply_morphological(frame, frame_idx, thresh_bright)
        msg = f"Unknown grouping_method: {self.grouping_method}"
        raise ValueError(msg)

    def _apply_connected(self, frame: np.ndarray, frame_idx: int, thresh_bright: float) -> tuple[np.ndarray, dict]:
        """Connected components analysis."""
        signal_mask = frame > thresh_bright
        labeled_regions, num_regions = label(signal_mask)

        categorized = np.zeros_like(frame, dtype=int)

        for region_id in range(1, num_regions + 1):
            region_mask = labeled_regions == region_id
            region_size = np.sum(region_mask)

            if region_size < self.min_region_size:
                continue

            categorized[region_mask] = CATEGORY_BRIGHT

        return categorized, {"frame_idx": frame_idx, "threshold": self.threshold_used}

    def _apply_watershed(self, frame: np.ndarray, frame_idx: int, thresh_bright: float) -> tuple[np.ndarray, dict]:
        """Watershed segmentation."""
        smoothed = ndimage.gaussian_filter(frame, sigma=1.5)
        mask = smoothed > thresh_bright
        distance = ndimage.distance_transform_edt(mask)

        local_max = peak_local_max(distance, min_distance=self.min_distance, labels=mask, exclude_border=False)

        markers = np.zeros_like(frame, dtype=int)
        for idx, (y, x) in enumerate(local_max):
            markers[y, x] = idx + 1

        labels = watershed(-smoothed, markers, mask=mask)

        categorized = np.zeros_like(frame, dtype=int)

        for region_id in np.unique(labels):
            if region_id == 0:
                continue
            region_mask = labels == region_id
            region_size = np.sum(region_mask)

            if region_size < self.min_region_size:
                continue

            categorized[region_mask] = CATEGORY_BRIGHT

        return categorized, {"frame_idx": frame_idx, "threshold": self.threshold_used}

    def _apply_morphological(self, frame: np.ndarray, frame_idx: int, thresh_bright: float) -> tuple[np.ndarray, dict]:
        """Morphological cleanup."""
        bright_mask = frame > thresh_bright

        struct = generate_binary_structure(2, 2)
        if self.kernel_size > 1:
            struct = ndimage.iterate_structure(struct, self.kernel_size)

        # Opening then closing
        bright_cleaned = binary_erosion(bright_mask, structure=struct)
        bright_cleaned = binary_dilation(bright_cleaned, structure=struct)
        bright_cleaned = binary_dilation(bright_cleaned, structure=struct)
        bright_cleaned = binary_erosion(bright_cleaned, structure=struct)

        categorized = np.zeros_like(frame, dtype=int)
        categorized[bright_cleaned] = CATEGORY_BRIGHT

        return categorized, {"frame_idx": frame_idx, "threshold": self.threshold_used}

    def get_results(self) -> dict:
        """
        Get all results as a dictionary.

        Returns:
            dict with source_frames, categorized_frames, frame_regions, threshold_used, method, threshold_method
        """
        return {
            "source_frames": self.source_frames,
            "categorized_frames": self.categorized_frames,
            "frame_regions": self.frame_regions,
            "threshold_used": self.threshold_used,
            "grouping_method": self.grouping_method,
            "threshold_method": self.threshold_method,
        }

    def get_export_data(self) -> dict:
        """Get categorizer data for export."""
        return {
            "threshold_method": self.threshold_method,
            "categorized_frames": self.categorized_frames,
        }
