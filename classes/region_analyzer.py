"""
Region analysis for categorized images.

This module provides post-processing analysis using skimage.measure
to calculate region properties (area, centroid) and find contours.
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
        self.dim_category: list[dict] = []  # {total_area_pixels, total_area_um2, centroid}
        self.bright_category: list[dict] = []

        # Largest region analysis (largest connected component per category)
        self.dim_largest: list[dict | None] = []
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
        self.dim_largest = []
        self.bright_largest = []

        for frame in categorized_frames:
            # Analyze individual regions (connected components)
            dim_info, dim_cont = self._analyze_regions(frame, CATEGORY_DIM)
            self.dim_regions.append(dim_info)
            self.dim_contours.append(dim_cont)

            bright_info, bright_cont = self._analyze_regions(frame, CATEGORY_BRIGHT)
            self.bright_regions.append(bright_info)
            self.bright_contours.append(bright_cont)

            # Analyze category-level (ALL pixels combined)
            self.dim_category.append(self._analyze_category_combined(frame, CATEGORY_DIM))
            self.bright_category.append(self._analyze_category_combined(frame, CATEGORY_BRIGHT))

            # Find largest region per category
            self.dim_largest.append(self._get_largest_region(dim_info))
            self.bright_largest.append(self._get_largest_region(bright_info))

        return self

    def _analyze_regions(
        self, frame: np.ndarray, category: int
    ) -> tuple[list[dict], list[np.ndarray]]:
        """
        Analyze individual connected regions of a specific category.

        Args:
            frame: Categorized 2D array
            category: Category value to analyze (1=dim, 2=bright)

        Returns:
            (regions_info, contours) - per connected component
        """
        mask = frame == category
        labeled = label(mask)

        # Get region properties for each connected component
        regions_info = []
        for region in regionprops(labeled):
            if region.area < self.min_area:
                continue

            regions_info.append({
                "area_pixels": region.area,
                "area_um2": self.area_pixel_to_um2(region.area),
                "centroid": region.centroid,  # (y, x) = (row, col)
                "bbox": region.bbox,  # (min_row, min_col, max_row, max_col)
                "label": region.label,
            })

        # Get contours
        contours = find_contours(mask, level=0.5)

        return regions_info, contours

    def _analyze_category_combined(self, frame: np.ndarray, category: int) -> dict:
        """
        Analyze ALL pixels of a category combined (not per-region).

        Args:
            frame: Categorized 2D array
            category: Category value to analyze (1=dim, 2=bright)

        Returns:
            dict with total_area_pixels, total_area_um2, centroid (or None if no pixels)
        """
        mask = frame == category
        total_pixels = np.sum(mask)

        if total_pixels == 0:
            return {
                "total_area_pixels": 0,
                "total_area_um2": 0.0,
                "centroid": None,
            }

        # Calculate centroid of ALL pixels in this category
        coords = np.argwhere(mask)  # (N, 2) with [row, col]
        centroid_y = np.mean(coords[:, 0])
        centroid_x = np.mean(coords[:, 1])

        return {
            "total_area_pixels": int(total_pixels),
            "total_area_um2": self.area_pixel_to_um2(total_pixels),
            "centroid": (centroid_y, centroid_x),  # (y, x) = (row, col)
        }

    def _get_largest_region(self, regions: list[dict]) -> dict | None:
        """
        Get the largest region by area.

        Args:
            regions: List of region dicts from _analyze_regions()

        Returns:
            The region dict with largest area_pixels, or None if empty
        """
        if not regions:
            return None
        return max(regions, key=lambda r: r["area_pixels"])

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
            # Largest region per category
            "dim_largest": self.dim_largest[frame_idx],
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
            # Largest region per category
            "dim_largest": self.dim_largest,
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
            dict with total counts, average areas, etc.
        """
        all_dim_areas = [r["area_um2"] for frame in self.dim_regions for r in frame]
        all_bright_areas = [r["area_um2"] for frame in self.bright_regions for r in frame]

        return {
            "obj": self.obj,
            "n_frames": len(self.dim_regions),
            "total_dim_regions": sum(len(frame) for frame in self.dim_regions),
            "total_bright_regions": sum(len(frame) for frame in self.bright_regions),
            "dim_area_um2_mean": np.mean(all_dim_areas) if all_dim_areas else 0,
            "dim_area_um2_std": np.std(all_dim_areas) if all_dim_areas else 0,
            "bright_area_um2_mean": np.mean(all_bright_areas) if all_bright_areas else 0,
            "bright_area_um2_std": np.std(all_bright_areas) if all_bright_areas else 0,
        }
