"""
Region analysis for categorized images.

For the spike frame, finds the largest bright connected component, then
collects every dim connected component spatially related to it, merging
them into one combined dim region.
"""

import numpy as np
from skimage.measure import find_contours, label, regionprops

# Category constants
CATEGORY_BACKGROUND = 0
CATEGORY_DIM = 1
CATEGORY_BRIGHT = 2

# Pixel scaling constants (pixel/um)
PIXEL_SCALE = {
    "10X": 0.75,
    "40X": 3.0,
    "60X": 4.5,
}

# Dim-region search window = the bright region's own bbox, expanded by this
# many multiples of its own span on each side (no extra pixel constant to tune).
DIM_SEARCH_MARGIN = 0.2


class RegionAnalyzer:
    """
    Find the largest bright region and its related dim region in a single (spike) frame.

    Finds the largest bright connected component, then collects every dim
    connected component whose bounding box overlaps a window around the
    bright region (the bright bbox expanded by DIM_SEARCH_MARGIN * its own
    span on each side), merging them into one combined dim region.

    Result dict per region:
        centroid   : (row, col) in pixels
        area_px    : area in pixels
        area_um2   : area in µm²
        x_span_px  : horizontal span in pixels
        y_span_px  : vertical span in pixels
        x_span_um  : horizontal span in µm
        y_span_um  : vertical span in µm
        contour    : bright -> (N, 2) array of [row, col] contour points, or None
                     dim    -> list of such arrays (one per disjoint piece), or None

    Example:
        >>> categorizer = SpatialCategorizer.morphological()
        >>> categorizer.fit(image_segment, spike_frame_idx=spike_frame_idx)
        >>> analyzer = RegionAnalyzer(categorizer.categorized_frames[spike_frame_idx], obj="10X")
        >>> results = analyzer.get_results()
    """

    def __init__(self, spike_frame: np.ndarray, obj: str = "10X") -> None:
        """
        Analyze the spike frame immediately on construction.

        Args:
            spike_frame: 2D array (0=background, 1=dim, 2=bright) for the spike frame only.
            obj: Objective magnification ("10X", "40X", "60X")
        """
        if obj not in PIXEL_SCALE:
            msg = f"Unknown objective: {obj}. Choose from {list(PIXEL_SCALE.keys())}"
            raise ValueError(msg)

        self.obj = obj
        self.pixel_per_um = PIXEL_SCALE[obj]
        self.um_per_pixel = 1.0 / self.pixel_per_um

        self.bright_largest = self._find_largest_bright(spike_frame)

        if self.bright_largest is None:
            self.dim_largest = None
        else:
            self.dim_largest = self._find_related_dim(spike_frame, self.bright_largest["_bbox"])

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _find_largest_bright(self, frame: np.ndarray) -> dict | None:
        """Find the largest bright connected component and measure it."""
        labeled = label(frame == CATEGORY_BRIGHT)
        candidates = list(regionprops(labeled))

        if not candidates:
            return None

        region = max(candidates, key=lambda r: r.area)
        min_row, min_col, max_row, max_col = region.bbox

        x_span_px = max_col - min_col
        y_span_px = max_row - min_row

        region_mask = labeled == region.label
        raw_contours = find_contours(region_mask, level=0.5)
        contour = max(raw_contours, key=len) if raw_contours else None

        return {
            "centroid":  region.centroid,
            "area_px":   region.area,
            "area_um2":  self._area_to_um2(region.area),
            "x_span_px": x_span_px,
            "y_span_px": y_span_px,
            "x_span_um": self._px_to_um(x_span_px),
            "y_span_um": self._px_to_um(y_span_px),
            "contour":   contour,
            "_bbox":     (min_row, min_col, max_row, max_col),  # internal: dim search-window construction
        }

    def _find_related_dim(self, frame: np.ndarray, bright_bbox: tuple[int, int, int, int]) -> dict | None:
        """Merge every dim component whose bbox overlaps the bright region's search window."""
        bright_min_row, bright_min_col, bright_max_row, bright_max_col = bright_bbox
        x_span_px = bright_max_col - bright_min_col
        y_span_px = bright_max_row - bright_min_row

        window_min_row = bright_min_row - DIM_SEARCH_MARGIN * y_span_px
        window_max_row = bright_max_row + DIM_SEARCH_MARGIN * y_span_px
        window_min_col = bright_min_col - DIM_SEARCH_MARGIN * x_span_px
        window_max_col = bright_max_col + DIM_SEARCH_MARGIN * x_span_px

        labeled = label(frame == CATEGORY_DIM)
        dim_mask = np.zeros_like(frame, dtype=bool)

        for region in regionprops(labeled):
            min_row, min_col, max_row, max_col = region.bbox
            overlaps = (
                min_row < window_max_row
                and max_row > window_min_row
                and min_col < window_max_col
                and max_col > window_min_col
            )
            if overlaps:
                dim_mask |= labeled == region.label

        if not np.any(dim_mask):
            return None

        rows, cols = np.nonzero(dim_mask)
        x_span_px = int(cols.max()) + 1 - int(cols.min())
        y_span_px = int(rows.max()) + 1 - int(rows.min())
        area_px = int(dim_mask.sum())

        return {
            "centroid":  (float(rows.mean()), float(cols.mean())),
            "area_px":   area_px,
            "area_um2":  self._area_to_um2(area_px),
            "x_span_px": x_span_px,
            "y_span_px": y_span_px,
            "x_span_um": self._px_to_um(x_span_px),
            "y_span_um": self._px_to_um(y_span_px),
            "contour":   find_contours(dim_mask, level=0.5),
        }

    # ── Unit conversion helpers ────────────────────────────────────────────────

    def _px_to_um(self, pixels: float) -> float:
        return pixels * self.um_per_pixel

    def _area_to_um2(self, area_px: float) -> float:
        return area_px * (self.um_per_pixel ** 2)

    # ── Result accessors ──────────────────────────────────────────────────────

    def get_results(self) -> dict:
        """Get dim and bright largest-region results for the spike frame."""
        return {
            "dim_largest":    self.dim_largest,
            "bright_largest": self.bright_largest,
        }

    def get_summary(self) -> dict:
        """Summary for the spike frame."""
        return {
            "obj":               self.obj,
            "has_dim_region":    self.dim_largest is not None,
            "has_bright_region": self.bright_largest is not None,
        }

    def get_export_data(self) -> dict:
        """Data for export (contours and internal fields stripped by exporter)."""
        return {
            "objective":      self.obj,
            "um_per_pixel":   self.um_per_pixel,
            "region_summary": self.get_summary(),
            "region_data":    self.get_results(),
        }
