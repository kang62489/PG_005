"""
Spatial-aware intensity categorization (background / bright) of a spike-aligned segment.

  Step 1. Threshold : trimmed baseline (pre-spike) mean + BASELINE_SIGMA_MULT * std
  Step 2. Group     : per frame, bright pixels -> connected / watershed / morphological regions
                      (morphological: objects < MIN_OBJECT_UM2 dropped when pixel_per_um is given)
  Step 3. Collect   : categorized frames + threshold via get_results() / get_export_data()

Example:
    >>> categorizer = SpatialCategorizer.connected(min_region_size=30)
    >>> categorizer.fit(image_segment, spike_frame_idx=spike_frame_idx)  # 3D array (frames, H, W)
    >>> results = categorizer.get_results()

    >>> SpatialCategorizer.watershed(min_distance=5, min_region_size=20).fit(image_segment, spike_frame_idx)
    >>> SpatialCategorizer.morphological(kernel_size=5).fit(image_segment, spike_frame_idx)
"""

## Modules
# Standard library imports
from typing import ClassVar

# Third-party imports
import numpy as np
from scipy import ndimage
from scipy.ndimage import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Local imports
from functions.cluster_kernels import binary_open_close_square

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 1: threshold -----------------------------------------------------
BASELINE_SIGMA_MULT = 2.0         # threshold = trimmed baseline mean + this many stds
BASELINE_TRIM_PCT = (0.1, 99.9)  # baseline pixels outside these percentiles are dropped before mean/std

# --- Step 2: group ---------------------------------------------------------
CATEGORY_BRIGHT = 1    # CAT pixel value for bright (background = 0)
MIN_OBJECT_UM2 = 2700.0  # µm²: morphological bright objects smaller than this are dropped (8-connected)


class SpatialCategorizer:
    """Threshold -> group -> collect bright regions for one segment (frames, H, W).

    Build with a factory: SpatialCategorizer.connected / .watershed / .morphological.
    """

    GROUPING_METHODS: ClassVar[list[str]] = ["connected", "watershed", "morphological"]
    THRESHOLD_METHODS: ClassVar[list[str]] = ["baseline_n_sigma"]

    def __init__(
        self,
        grouping_method: str,
        threshold_method: str = "baseline_n_sigma",
        *,
        # Connected/Watershed parameters
        min_region_size: int = 20,
        # Watershed parameters
        min_distance: int = 10,
        # Morphological parameters
        kernel_size: int = 3,
        pixel_per_um: float | None = None,
    ) -> None:
        """Validate the method names and store the parameters; prefer the factory methods."""
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
        self.pixel_per_um = pixel_per_um  # morphological; None = no MIN_OBJECT_UM2 filter

        # Results (populated after fit)
        self.source_frames: list[np.ndarray] = []
        self.categorized_frames: list[np.ndarray] = []
        self.frame_regions: list[dict] = []
        self.threshold_used: float | None = None

    @classmethod
    def connected(
        cls,
        threshold_method: str = "baseline_n_sigma",
        *,
        min_region_size: int = 20,
    ) -> "SpatialCategorizer":
        """Connected components; regions smaller than min_region_size (px) are dropped."""
        return cls(
            grouping_method="connected",
            threshold_method=threshold_method,
            min_region_size=min_region_size,
        )

    @classmethod
    def watershed(
        cls,
        threshold_method: str = "baseline_n_sigma",
        *,
        min_region_size: int = 20,
        min_distance: int = 10,
    ) -> "SpatialCategorizer":
        """Watershed; min_distance (px) between peaks (larger = fewer regions), small regions dropped."""
        return cls(
            grouping_method="watershed",
            threshold_method=threshold_method,
            min_region_size=min_region_size,
            min_distance=min_distance,
        )

    @classmethod
    def morphological(
        cls,
        threshold_method: str = "baseline_n_sigma",
        *,
        kernel_size: int = 3,
        pixel_per_um: float | None = None,
    ) -> "SpatialCategorizer":
        """Morphological open + close; larger kernel_size = more aggressive cleanup.

        With pixel_per_um (objective scale), objects smaller than MIN_OBJECT_UM2 are dropped afterwards.
        """
        return cls(
            grouping_method="morphological",
            threshold_method=threshold_method,
            kernel_size=kernel_size,
            pixel_per_um=pixel_per_um,
        )

    def fit(self, image_segment: np.ndarray, spike_frame_idx: int) -> "SpatialCategorizer":
        """Threshold from frames before spike_frame_idx, then categorize every frame; returns self."""
        self.source_frames = [image_segment[i] for i in range(image_segment.shape[0])]

        self._calculate_global_threshold(spike_frame_idx)

        self.categorized_frames = []
        self.frame_regions = []

        for frame_idx, frame in enumerate(self.source_frames):
            categorized, stats = self._dispatch_frame(frame, frame_idx)
            self.categorized_frames.append(categorized)
            self.frame_regions.append(stats)

        return self

    # =======================================================================
    #
    #   STEP 1 -- THRESHOLD
    #
    # =======================================================================

    def _calculate_global_threshold(self, spike_frame_idx: int) -> None:
        """Calculate the bright threshold from the baseline (pre-spike) frames."""
        self.threshold_used = self.compute_baseline_threshold(self.source_frames[:spike_frame_idx])

    @staticmethod
    def compute_baseline_threshold(baseline_frames: list[np.ndarray]) -> float:
        """Bright threshold: mean + BASELINE_SIGMA_MULT * std of the baseline pixels.

        Pixels outside the BASELINE_TRIM_PCT percentiles are dropped first.
        """
        baseline_pixels = np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in baseline_frames])
        lo, hi = np.percentile(baseline_pixels, BASELINE_TRIM_PCT)
        kept = baseline_pixels[(baseline_pixels >= lo) & (baseline_pixels <= hi)]
        return float(kept.mean() + BASELINE_SIGMA_MULT * kept.std())

    # =======================================================================
    #
    #   STEP 2 -- GROUP
    #
    # =======================================================================

    def categorize_frame(self, frame: np.ndarray, frame_idx: int, threshold: float) -> np.ndarray:
        """Categorize one frame (0 = background, 1 = bright) with a known threshold, without fit()."""
        self.threshold_used = threshold
        categorized, _ = self._dispatch_frame(frame, frame_idx)
        return categorized

    def _dispatch_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[np.ndarray, dict]:
        """Run the configured grouping method on one frame -> (categorized, stats)."""
        thresh_bright = self.threshold_used

        if self.grouping_method == "connected":
            return self._apply_connected(frame, frame_idx, thresh_bright)
        if self.grouping_method == "watershed":
            return self._apply_watershed(frame, frame_idx, thresh_bright)
        if self.grouping_method == "morphological":
            return self._apply_morphological(frame, frame_idx, thresh_bright)
        msg = f"Unknown grouping_method: {self.grouping_method}"
        raise ValueError(msg)

    # --- 2a. connected -----------------------------------------------------

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

    # --- 2b. watershed -----------------------------------------------------

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

    # --- 2c. morphological -------------------------------------------------

    def _apply_morphological(self, frame: np.ndarray, frame_idx: int, thresh_bright: float) -> tuple[np.ndarray, dict]:
        """Morphological cleanup: opening then closing with a (2k+1) x (2k+1) square (k = kernel_size, min 1).

        Same result as scipy binary_erosion/dilation with iterate_structure(full 3x3, kernel_size), via numba.
        """
        bright_mask = frame > thresh_bright
        bright_cleaned = binary_open_close_square(bright_mask, max(self.kernel_size, 1))

        if self.pixel_per_um is not None:
            labeled, _ = label(bright_cleaned, structure=np.ones((3, 3), dtype=bool))
            object_um2 = np.bincount(labeled.ravel()) / self.pixel_per_um ** 2
            keep = object_um2 >= MIN_OBJECT_UM2
            keep[0] = False  # background label
            bright_cleaned = keep[labeled]

        categorized = np.zeros_like(frame, dtype=int)
        categorized[bright_cleaned] = CATEGORY_BRIGHT

        return categorized, {"frame_idx": frame_idx, "threshold": self.threshold_used}

    # =======================================================================
    #
    #   STEP 3 -- COLLECT
    #
    # =======================================================================

    def get_results(self) -> dict:
        """source_frames, categorized_frames, frame_regions, threshold_used, grouping_method, threshold_method."""
        return {
            "source_frames": self.source_frames,
            "categorized_frames": self.categorized_frames,
            "frame_regions": self.frame_regions,
            "threshold_used": self.threshold_used,
            "grouping_method": self.grouping_method,
            "threshold_method": self.threshold_method,
        }

    def get_export_data(self) -> dict:
        """threshold_method + categorized_frames for export."""
        return {
            "threshold_method": self.threshold_method,
            "categorized_frames": self.categorized_frames,
        }
