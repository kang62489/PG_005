"""
Region analysis for categorized images.

This module provides post-processing analysis using skimage.measure
to calculate region properties (centroid, spans) and find contours.
"""

from typing import TYPE_CHECKING

import numpy as np
from skimage.measure import find_contours, label, regionprops

if TYPE_CHECKING:
    from skimage.measure._regionprops import RegionProperties

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
    Analyze regions in categorized images.

    Uses skimage.measure.regionprops() for area/centroid calculations
    and find_contours() for contour extraction.

    Example:
        >>> from classes.spatial_categorization import SpatialCategorizer
        >>> from classes.region_analyzer import RegionAnalyzer
        >>>
        >>> categorizer = SpatialCategorizer.morphological()
        >>> categorizer.fit(image_segment)
        >>>
        >>> analyzer = RegionAnalyzer(obj="10X")
        >>> analyzer.fit(categorizer.categorized_frames)
        >>> results = analyzer.get_results()
    """

    def __init__(self, obj: str = "10X", min_area: int = 0) -> None:
        """
        Initialize the RegionAnalyzer.

        Args:
            obj: Objective magnification ("10X", "40X", "60X")
            min_area: Minimum region area in pixels (smaller regions ignored)
        """
        if obj not in PIXEL_SCALE:
            msg = f"Unknown objective: {obj}. Choose from {list(PIXEL_SCALE.keys())}"
            raise ValueError(msg)

        self.obj = obj
        self.pixel_per_um = PIXEL_SCALE[obj]
        self.um_per_pixel = 1.0 / self.pixel_per_um
        self.min_area = min_area

        # Results (populated after fit)
        # Per-region analysis (individual connected components)
        self.dim_regions: list[list[dict]] = []
        self.bright_regions: list[list[dict]] = []
        self.dim_contours: list[list[np.ndarray]] = []
        self.bright_contours: list[list[np.ndarray]] = []

        # Category-level analysis (ALL pixels in category combined)
        self.dim_category: list[dict] = []  # {centroid}
        self.bright_category: list[dict] = []

        # Largest region analysis (largest connected component per category)
        self.bright_largest: list[dict | None] = []

    def fit(self, categorized_frames: list[np.ndarray]) -> "RegionAnalyzer":
        """
        Analyze all categorized frames.

        Args:
            categorized_frames: List of 2D arrays with values 0=background, 1=dim, 2=bright

        Returns:
            self (for method chaining)
        """
        self.dim_regions = []
        self.bright_regions = []
        self.dim_contours = []
        self.bright_contours = []
        self.dim_category = []
        self.bright_category = []
        self.bright_largest = []

        for frame in categorized_frames:
            # Analyze individual regions (connected components)
            dim_info, dim_cont, dim_props = self._analyze_regions(frame, CATEGORY_DIM)
            self.dim_regions.append(dim_info)
            self.dim_contours.append(dim_cont)

            bright_info, bright_cont, bright_props = self._analyze_regions(frame, CATEGORY_BRIGHT)
            self.bright_regions.append(bright_info)
            self.bright_contours.append(bright_cont)

            # Analyze category-level (ALL pixels combined)
            self.dim_category.append(self._analyze_category_combined(frame, CATEGORY_DIM))
            self.bright_category.append(self._analyze_category_combined(frame, CATEGORY_BRIGHT))

            # Find largest bright region using cached props
            self.bright_largest.append(self._get_largest_region(bright_info, bright_props))

        return self

    def _analyze_regions(
        self, frame: np.ndarray, category: int
    ) -> tuple[list[dict], list[np.ndarray], list["RegionProperties"]]:
        """
        Analyze individual connected regions of a specific category.

        Args:
            frame: Categorized 2D array
            category: Category value to analyze (1=dim, 2=bright)

        Returns:
            (regions_info, contours, region_props_list) - per connected component
        """
        mask = frame == category
        labeled = label(mask)

        # Get region properties for each connected component
        all_props = regionprops(labeled)
        regions_info = []
        region_props_list = []

        for region in all_props:
            if region.area < self.min_area:
                continue

            regions_info.append({
                "centroid": region.centroid,  # (y, x) = (row, col)
                "bbox": region.bbox,  # (min_row, min_col, max_row, max_col)
                "label": region.label,
                "area": region.area,  # Temporary: needed to find largest region
            })
            region_props_list.append(region)

        # Get contours
        contours = find_contours(mask, level=0.5)

        return regions_info, contours, region_props_list

    def _analyze_category_combined(self, frame: np.ndarray, category: int) -> dict:
        """
        Analyze ALL pixels of a category combined (not per-region).

        Args:
            frame: Categorized 2D array
            category: Category value to analyze (1=dim, 2=bright)

        Returns:
            dict with centroid (or None if no pixels)
        """
        mask = frame == category
        total_pixels = np.sum(mask)

        if total_pixels == 0:
            return {
                "centroid": None,
            }

        # Calculate centroid of ALL pixels in this category
        coords = np.argwhere(mask)  # (N, 2) with [row, col]
        centroid_y = np.mean(coords[:, 0])
        centroid_x = np.mean(coords[:, 1])

        return {
            "centroid": (centroid_y, centroid_x),  # (y, x) = (row, col)
        }

    def _get_largest_region(
        self, regions: list[dict], region_props_list: list["RegionProperties"]
    ) -> dict | None:
        """
        Get the largest region by area and calculate its spans.

        Args:
            regions: List of region dicts from _analyze_regions()
            region_props_list: List of regionprops objects (same order as regions)

        Returns:
            The region dict with largest area including span measurements, or None if empty
        """
        if not regions:
            return None

        # Find largest region by area
        largest_idx = max(range(len(regions)), key=lambda i: regions[i]["area"])
        largest = regions[largest_idx]
        largest_region_props = region_props_list[largest_idx]

        # Calculate spans using cached props
        span_data = self._calculate_region_spans(largest_region_props)

        # Return dict without area field
        return {
            "centroid": largest["centroid"],
            "bbox": largest["bbox"],
            "label": largest["label"],
            "x_span_pixels": span_data["x_span_pixels"],
            "y_span_pixels": span_data["y_span_pixels"],
            "x_span_um": span_data["x_span_um"],
            "y_span_um": span_data["y_span_um"],
        }

    def get_largest_region_contour(self, frame: np.ndarray, category: int, label_id: int) -> np.ndarray | None:
        """
        Get the contour for a specific labeled region.

        Args:
            frame: Categorized 2D array
            category: Category value (1=dim, 2=bright)
            label_id: The label of the region to get contour for

        Returns:
            Contour array or None if not found
        """
        mask = frame == category
        labeled = label(mask)

        # Create mask for only this specific region
        region_mask = labeled == label_id

        if not np.any(region_mask):
            return None

        # Get contour for this region
        contours = find_contours(region_mask, level=0.5)

        # Return the first (and should be only) contour
        return contours[0] if contours else None

    def _calculate_region_spans(self, region_props: "RegionProperties") -> dict:
        """
        Calculate orthogonal spans from centroid of a region.
        Uses region.coords (all pixel coordinates in the region).

        Args:
            region_props: A regionprops object from skimage.measure

        Returns:
            dict with x_span_pixels, y_span_pixels, x_span_um, y_span_um, centroid
        """
        centroid = region_props.centroid  # (row, col) = (y, x)
        centroid_row, centroid_col = centroid

        # Get all pixel coordinates in this region (already available from regionprops)
        coords = region_props.coords  # (N, 2) array of [row, col]

        rows = coords[:, 0]
        cols = coords[:, 1]

        # Calculate max distances in 4 orthogonal directions
        north_dist = np.max(centroid_row - rows[rows < centroid_row]) if np.any(rows < centroid_row) else 0
        south_dist = np.max(rows[rows > centroid_row] - centroid_row) if np.any(rows > centroid_row) else 0
        east_dist = np.max(cols[cols > centroid_col] - centroid_col) if np.any(cols > centroid_col) else 0
        west_dist = np.max(centroid_col - cols[cols < centroid_col]) if np.any(cols < centroid_col) else 0

        # Calculate spans
        x_span_pixels = west_dist + east_dist
        y_span_pixels = north_dist + south_dist

        return {
            "x_span_pixels": float(x_span_pixels),
            "y_span_pixels": float(y_span_pixels),
            "x_span_um": self.pixel_to_um(x_span_pixels),
            "y_span_um": self.pixel_to_um(y_span_pixels),
            "centroid": centroid,  # (row, col)
        }

    def pixel_to_um(self, pixels: float) -> float:
        """Convert pixel distance to micrometers."""
        return pixels * self.um_per_pixel

    def area_pixel_to_um2(self, area_pixels: float) -> float:
        """Convert pixel area to square micrometers."""
        return area_pixels * (self.um_per_pixel ** 2)

    def get_frame_results(self, frame_idx: int) -> dict:
        """
        Get results for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            dict with per-region and category-level results
        """
        return {
            # Per-region (individual connected components)
            "dim_regions": self.dim_regions[frame_idx],
            "bright_regions": self.bright_regions[frame_idx],
            "dim_contours": self.dim_contours[frame_idx],
            "bright_contours": self.bright_contours[frame_idx],
            # Category-level (ALL pixels combined)
            "dim_category": self.dim_category[frame_idx],
            "bright_category": self.bright_category[frame_idx],
            # Largest bright region
            "bright_largest": self.bright_largest[frame_idx],
        }

    def get_results(self) -> dict:
        """
        Get all results.

        Returns:
            dict with all per-frame region info, contours, and category-level stats
        """
        return {
            # Per-region (individual connected components)
            "dim_regions": self.dim_regions,
            "bright_regions": self.bright_regions,
            "dim_contours": self.dim_contours,
            "bright_contours": self.bright_contours,
            # Category-level (ALL pixels combined)
            "dim_category": self.dim_category,
            "bright_category": self.bright_category,
            # Largest bright region
            "bright_largest": self.bright_largest,
            # Metadata
            "obj": self.obj,
            "um_per_pixel": self.um_per_pixel,
        }

    def get_export_data(self) -> dict:
        """Get region analysis data for export (contours removed)."""
        region_data = self.get_results()
        region_data.pop("dim_contours", None)
        region_data.pop("bright_contours", None)

        return {
            "objective": self.obj,
            "um_per_pixel": self.um_per_pixel,
            "region_summary": self.get_summary(),
            "region_data": region_data,
        }

    def get_summary(self) -> dict:
        """
        Get summary statistics across all frames.

        Returns:
            dict with total counts, etc.
        """
        return {
            "obj": self.obj,
            "n_frames": len(self.dim_regions),
            "total_dim_regions": sum(len(frame) for frame in self.dim_regions),
            "total_bright_regions": sum(len(frame) for frame in self.bright_regions),
        }
