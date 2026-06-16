"""
Region analysis for categorized images.

For each frame, finds the single largest connected component in the bright and dim
categories, then measures its area, centroid, spans, and contour.
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


class RegionAnalyzer:
    """
    Find and measure the largest bright and dim regions in categorized frames.

    For each frame, identifies the single largest connected component per
    category and calculates its area, centroid, spans, and contour.

    Result dict per region:
        centroid   : (row, col) in pixels
        area_px    : area in pixels
        area_um2   : area in µm²
        x_span_px  : horizontal span in pixels (west + east from centroid)
        y_span_px  : vertical span in pixels (north + south from centroid)
        x_span_um  : horizontal span in µm
        y_span_um  : vertical span in µm
        contour    : (N, 2) array of [row, col] contour points, or None

    Example:
        >>> categorizer = SpatialCategorizer.morphological()
        >>> categorizer.fit(image_segment)
        >>> analyzer = RegionAnalyzer(categorizer.categorized_frames, obj="10X")
        >>> spike = analyzer.get_frame_results(len(categorizer.categorized_frames) // 2)
    """

    def __init__(self, categorized_frames: list[np.ndarray], obj: str = "10X", min_area: int = 0) -> None:
        """
        Analyze categorized frames immediately on construction.

        Args:
            categorized_frames: List of 2D arrays (0=background, 1=dim, 2=bright)
            obj: Objective magnification ("10X", "40X", "60X")
            min_area: Minimum region area in pixels; smaller components are ignored
        """
        if obj not in PIXEL_SCALE:
            msg = f"Unknown objective: {obj}. Choose from {list(PIXEL_SCALE.keys())}"
            raise ValueError(msg)

        self.obj = obj
        self.pixel_per_um = PIXEL_SCALE[obj]
        self.um_per_pixel = 1.0 / self.pixel_per_um
        self.min_area = min_area

        self.dim_largest: list[dict | None] = []
        self.bright_largest: list[dict | None] = []

        for frame in categorized_frames:
            self.dim_largest.append(self._find_largest(frame, CATEGORY_DIM))
            self.bright_largest.append(self._find_largest(frame, CATEGORY_BRIGHT))

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _find_largest(self, frame: np.ndarray, category: int) -> dict | None:
        """Find the largest connected component of a category and measure it."""
        mask = frame == category
        labeled = label(mask)
        candidates = [r for r in regionprops(labeled) if r.area >= self.min_area]

        if not candidates:
            return None

        region = max(candidates, key=lambda r: r.area)
        centroid_row, centroid_col = region.centroid
        rows, cols = region.coords[:, 0], region.coords[:, 1]

        north = float(np.max(centroid_row - rows[rows < centroid_row])) if np.any(rows < centroid_row) else 0.0
        south = float(np.max(rows[rows > centroid_row] - centroid_row)) if np.any(rows > centroid_row) else 0.0
        west  = float(np.max(centroid_col - cols[cols < centroid_col])) if np.any(cols < centroid_col) else 0.0
        east  = float(np.max(cols[cols > centroid_col] - centroid_col)) if np.any(cols > centroid_col) else 0.0

        x_span_px = west + east
        y_span_px = north + south

        region_mask = labeled == region.label
        raw_contours = find_contours(region_mask, level=0.5)
        contour = max(raw_contours, key=len) if raw_contours else None

        return {
            "centroid":  (centroid_row, centroid_col),
            "area_px":   region.area,
            "area_um2":  self._area_to_um2(region.area),
            "x_span_px": x_span_px,
            "y_span_px": y_span_px,
            "x_span_um": self._px_to_um(x_span_px),
            "y_span_um": self._px_to_um(y_span_px),
            "contour":   contour,
            "_label":    region.label,  # internal: used by exporter for boundary TIFF
        }

    # ── Unit conversion helpers ────────────────────────────────────────────────

    def _px_to_um(self, pixels: float) -> float:
        return pixels * self.um_per_pixel

    def _area_to_um2(self, area_px: float) -> float:
        return area_px * (self.um_per_pixel ** 2)

    # ── Result accessors ──────────────────────────────────────────────────────

    def get_frame_results(self, frame_idx: int) -> dict:
        """Get dim and bright largest-region results for a single frame."""
        return {
            "dim_largest":    self.dim_largest[frame_idx],
            "bright_largest": self.bright_largest[frame_idx],
        }

    def get_results(self) -> dict:
        """Get all per-frame results."""
        return {
            "dim_largest":    self.dim_largest,
            "bright_largest": self.bright_largest,
            "obj":            self.obj,
            "um_per_pixel":   self.um_per_pixel,
        }

    def get_summary(self) -> dict:
        """Summary statistics across all frames."""
        return {
            "obj":                    self.obj,
            "n_frames":               len(self.dim_largest),
            "total_dim_regions":      sum(1 for r in self.dim_largest    if r is not None),
            "total_bright_regions":   sum(1 for r in self.bright_largest if r is not None),
        }

    def get_export_data(self) -> dict:
        """Data for export (contours and internal fields stripped by exporter)."""
        return {
            "objective":      self.obj,
            "um_per_pixel":   self.um_per_pixel,
            "region_summary": self.get_summary(),
            "region_data":    self.get_results(),
        }
