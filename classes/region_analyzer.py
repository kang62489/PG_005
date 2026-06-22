"""
Region analysis for categorized images.

For the spike frame, finds the largest bright connected component, then
combines every dim-category pixel in the frame into one dim region.
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


def _nanargmax_relative(trace: np.ndarray, spike_frame_idx: int) -> int | None:
    """Index (relative to the spike frame) of trace's peak, or None if trace is all-NaN."""
    if np.all(np.isnan(trace)):
        return None
    return int(np.nanargmax(trace)) - spike_frame_idx


class RegionAnalyzer:
    """
    Find the largest bright region and the combined dim region in a single (spike) frame.

    Finds the largest bright connected component; if found, combines every
    dim-category pixel in the frame into one dim region (no spatial relation
    to the bright region is required).

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

    def __init__(self, spike_frame: np.ndarray, obj: str = "10X", min_area_um2: float = 0.0) -> None:
        """
        Analyze the spike frame immediately on construction.

        Args:
            spike_frame: 2D array (0=background, 1=dim, 2=bright) for the spike frame only.
            obj: Objective magnification ("10X", "40X", "60X")
            min_area_um2: Connected components smaller than this (in um^2) are ignored
                when picking the largest bright component. Default 0.0 disables filtering.
                Does not affect dim-region detection, which combines all dim pixels.
        """
        if obj not in PIXEL_SCALE:
            msg = f"Unknown objective: {obj}. Choose from {list(PIXEL_SCALE.keys())}"
            raise ValueError(msg)

        self.obj = obj
        self.pixel_per_um = PIXEL_SCALE[obj]
        self.um_per_pixel = 1.0 / self.pixel_per_um
        self.min_area_um2 = min_area_um2
        self._min_area_px = min_area_um2 * (self.pixel_per_um ** 2)

        self.bright_largest = self._find_largest_category(spike_frame, CATEGORY_BRIGHT)

        if self.bright_largest is None:
            self.dim_largest = None
        else:
            self.dim_largest = self._find_all_dim(spike_frame)

    # ── Core analysis ─────────────────────────────────────────────────────────

    def _find_largest_category(self, frame: np.ndarray, category: int) -> dict | None:
        """Find the largest connected component of the given category and measure it."""
        labeled = label(frame == category)
        candidates = [r for r in regionprops(labeled) if r.area >= self._min_area_px]

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
            "_bbox":     (min_row, min_col, max_row, max_col),  # internal: stripped before export
            "_mask":     region_mask,  # internal: reused by get_temporal_traces
        }

    def _find_all_dim(self, frame: np.ndarray) -> dict | None:
        """Combine every dim-category pixel in the frame into a single region."""
        dim_mask = frame == CATEGORY_DIM

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

    def area_in_combined_region(self, frame: np.ndarray, category: int) -> float:
        """Area (µm²) of `category` pixels in `frame`, restricted to the spike frame's bright+dim area.

        Args:
            frame: 2D array (0=background, 1=dim, 2=bright) for any frame.
            category: CATEGORY_BRIGHT or CATEGORY_DIM.

        Returns:
            Area in µm² (0.0 if neither bright_largest nor dim_largest was found).
        """
        area_mask = np.zeros(frame.shape, dtype=bool)
        if self.bright_largest is not None:
            area_mask |= self.bright_largest["_mask"]
        if self.dim_largest is not None:
            area_mask |= self.dim_largest["_mask"]

        pixel_count = int(np.count_nonzero(area_mask & (frame == category)))
        return self._area_to_um2(pixel_count)

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
            detected in the spike frame (so peak-finding correctly reports "no peak"
            instead of a fake peak at index 0); total_trace = bright_trace + dim_trace,
            therefore NaN throughout if either region is missing.
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

    def get_peak_latency_ms(self, segment: np.ndarray, spike_frame_idx: int, frame_duration_ms: float) -> float | None:
        """Peak-to-peak latency (dim peak minus bright peak) in milliseconds.

        Args:
            segment: 3D array (frames, height, width) — see get_temporal_traces.
            spike_frame_idx: index of the spike frame within the segment.
            frame_duration_ms: milliseconds per frame.

        Returns:
            Latency in ms, or None if either trace's peak could not be located.
        """
        traces = self.get_temporal_traces(segment)
        bright_peak_rel = _nanargmax_relative(traces["bright_trace"], spike_frame_idx)
        dim_peak_rel = _nanargmax_relative(traces["dim_trace"], spike_frame_idx)
        if bright_peak_rel is None or dim_peak_rel is None:
            return None
        return (dim_peak_rel - bright_peak_rel) * frame_duration_ms

    def get_export_data(self) -> dict:
        """Data for export (contours and internal fields stripped by exporter)."""
        return {
            "objective":      self.obj,
            "um_per_pixel":   self.um_per_pixel,
            "region_summary": self.get_summary(),
            "region_data":    self.get_results(),
        }
