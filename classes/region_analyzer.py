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

    Also exposes find_largest_regions(frame) for finding the largest bright/dim
    component independently in any frame (no spatial relation between them) —
    used for per-frame summary panels rather than the spike-frame's related dim.

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

        self.bright_largest = self._find_largest_category(spike_frame, CATEGORY_BRIGHT)

        if self.bright_largest is None:
            self.dim_largest = None
        else:
            self.dim_largest = self._find_related_dim(spike_frame, self.bright_largest["_bbox"])

    # ── Core analysis ─────────────────────────────────────────────────────────

    def find_largest_regions(self, frame: np.ndarray) -> dict:
        """Find the largest bright and largest dim component independently in a frame.

        Unlike the spike-frame's `dim_largest` (a merge of every dim component
        related to the bright region), this looks at bright/dim each on their
        own — no spatial relation between them. Used for the per-frame summary
        panels in the spatiotemporal figure (see classes/plot_results.py).

        Args:
            frame: 2D array (0=background, 1=dim, 2=bright) for any frame.

        Returns:
            dict with "bright"/"dim" keys, each a region dict (see class docstring)
            or None if that category has no connected component in this frame.
        """
        return {
            "bright": self._find_largest_category(frame, CATEGORY_BRIGHT),
            "dim":    self._find_largest_category(frame, CATEGORY_DIM),
        }

    def _find_largest_category(self, frame: np.ndarray, category: int) -> dict | None:
        """Find the largest connected component of the given category and measure it."""
        labeled = label(frame == category)
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
            "_mask":     region_mask,  # internal: reused by get_temporal_traces
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
            "_mask":     dim_mask,  # internal: reused by get_temporal_traces
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

    def get_temporal_traces(self, segment: np.ndarray) -> dict:
        """Mean intensity per frame within the spike frame's fixed bright/dim masks.

        Args:
            segment: 3D array (frames, height, width) sharing the spike frame's
                (height, width) — e.g. the z-scored median segment the categorizer
                was fit on.

        Returns:
            dict with "bright_trace"/"dim_trace"/"total_trace": 1D arrays (n_frames,).
            bright_trace/dim_trace are NaN-filled if the corresponding region was not
            detected in the spike frame; total_trace = bright_trace + dim_trace (NaN
            if either is NaN).
        """
        n_frames = segment.shape[0]

        bright_trace = np.full(n_frames, np.nan)
        if self.bright_largest is not None:
            mask = self.bright_largest["_mask"]
            bright_trace = np.array([frame[mask].mean() for frame in segment])

        dim_trace = np.full(n_frames, np.nan)
        if self.dim_largest is not None:
            mask = self.dim_largest["_mask"]
            dim_trace = np.array([frame[mask].mean() for frame in segment])

        total_trace = bright_trace + dim_trace

        return {"bright_trace": bright_trace, "dim_trace": dim_trace, "total_trace": total_trace}

    def get_export_data(self) -> dict:
        """Data for export (contours and internal fields stripped by exporter)."""
        return {
            "objective":      self.obj,
            "um_per_pixel":   self.um_per_pixel,
            "region_summary": self.get_summary(),
            "region_data":    self.get_results(),
        }
