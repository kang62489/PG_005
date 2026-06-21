"""
Results exporter for saving analysis outputs.

Exports analysis results to:
- SQLite database (metadata, spike-frame region measurements, region_analysis JSON)
- TIFF files (spike-centered median stack, categorized frames for ImageJ overlay)
- PNG figures (spatiotemporal summary plot)

Filenames use a compact code: {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_{TYPE}.
See ResultsExporter's class docstring for the full folder layout.
"""

## Modules
# Standard library imports
import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party imports
import numpy as np
import tifffile

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_SITE_NUMBER = re.compile(r"(\d+)$")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def optimize_region_data(region_data: dict) -> dict:
    """Strip contours and internal fields from region data before JSON serialization.

    Keeps only scalar measurements for the spike frame's bright/dim largest
    regions — contour arrays and the internal _bbox/_mask fields are removed to
    keep the stored JSON small.

    Args:
        region_data: Full region analysis dict from RegionAnalyzer.get_results()
            (single bright_largest/dim_largest dicts, or None if not found).

    Returns:
        Serialization-safe dict with both regions' scalar measurements.
    """
    _strip = {"contour", "_bbox", "_mask"}

    def _clean(region: dict | None) -> dict | None:
        return {k: v for k, v in region.items() if k not in _strip} if region is not None else None

    return {
        "bright_largest": _clean(region_data.get("bright_largest")),
        "dim_largest":    _clean(region_data.get("dim_largest")),
    }


class ResultsExporter:
    """
    Export analysis results to files and SQLite database.

    Output structure (flat — exp_date is already encoded in every filename, so
    no per-date folder layer is needed):
        results/
        ├── results.db
        ├── median/
        │   └── {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_MED.tif
        ├── categorized/
        │   └── {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_CAT.tif
        └── region_sta/
            └── {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_SPATIAL.png

    region_sta/ is created on demand by export_figure() — export_all() only creates
    median/ and categorized/.

    A{n} is a batch-local sequential animal index (see build_animal_index_map),
    not the real ANIMAL_ID; S{slice} is the SLICE value verbatim; C{site} is
    derived from AT (e.g. SITE_1/CELL_1 -> C1).
    """

    def __init__(self, results_root: Path = Path(__file__).parent.parent / "results") -> None:
        """
        Initialize the ResultsExporter.

        Args:
            results_root: Root directory for results (default: "results")
        """
        self.results_root = Path(results_root)
        self.db_path = self.results_root / "results.db"
        self._init_db()

    @staticmethod
    def build_animal_index_map(animal_ids: Iterable[str]) -> dict[str, int]:
        """Map each distinct animal ID to a 1-based sequential index, sorted for stable output.

        A{n} is a batch-local scan index (not the real ANIMAL_ID) — call this once
        per ana-list run with every row's ANIMAL_ID before exporting.
        """
        return {animal_id: i + 1 for i, animal_id in enumerate(sorted(set(animal_ids)))}

    @staticmethod
    def derive_site_code(at: str) -> str:
        """Derive a compact site code from an AT value, e.g. 'SITE_1'/'CELL_1' -> 'C1'."""
        match = _SITE_NUMBER.search(at)
        if match is None:
            msg = f"Could not derive a site code from AT={at!r} (no trailing digits)"
            raise ValueError(msg)
        return f"C{match.group(1)}"

    @staticmethod
    def build_export_stem(
        exp_date: str,
        img_serial: str,
        animal_idx: int,
        slice_val: str,
        at: str,
        detrend_mode: str,
        normalization: str,
        file_type: str,
    ) -> str:
        """Build the compact export filename stem (no extension).

        Code: {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_{TYPE}
        A{n} is a batch-local sequential animal index (not the real ANIMAL_ID), so the
        code stays short and scannable across a whole ana-list run.
        """
        site_code = ResultsExporter.derive_site_code(at)
        return f"{exp_date}-{img_serial}_A{animal_idx}S{slice_val}{site_code}_{detrend_mode}_{normalization}_{file_type}"

    def _init_db(self) -> None:
        """Create database and tables if not exist."""
        self.results_root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_date TEXT NOT NULL,
                abf_serial TEXT NOT NULL,
                img_serial TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                objective TEXT,
                um_per_pixel REAL,
                threshold_method TEXT,
                n_spikes_detected INTEGER,
                n_spikes_analyzed INTEGER,
                has_bright_region INTEGER,
                has_dim_region INTEGER,
                region_analysis TEXT,
                ANIMAL_ID TEXT,
                SLICE TEXT,
                AT TEXT,
                med_filename TEXT,
                centroid_y REAL,
                centroid_x REAL,
                x_span_pixels REAL,
                y_span_pixels REAL,
                x_span_um REAL,
                y_span_um REAL,
                bright_area_um2 REAL,
                dim_area_um2 REAL,
                dim_x_span_um REAL,
                dim_y_span_um REAL,
                peak_latency_ms REAL,
                zscore_min REAL,
                zscore_max REAL,
                UNIQUE(exp_date, abf_serial, img_serial)
            )
        """)
        conn.commit()
        conn.close()

    def export_all(
        self,
        # Experiment identifiers
        exp_date: str,
        abf_serial: str,
        img_serial: str,
        # Filename-code metadata
        animal_idx: int,
        animal_id: str,
        slice_val: str,
        at: str,
        # Processing metadata
        detrend_mode: str,
        normalization: str,
        # ABF metadata
        num_found_spikes: int,
        n_spikes_analyzed: int,
        # Categorization metadata
        threshold_method: str,
        objective: str,
        um_per_pixel: float,
        # Data to save
        median_stack: np.ndarray,
        categorized_frames: list[np.ndarray],
        zscore_range: tuple[float, float],
        # Analysis results
        region_summary: dict,
        region_data: dict,
        peak_latency_ms: float | None,
    ) -> dict[str, Path]:
        """
        Export all results and update database.

        Args:
            exp_date: Experiment date string
            abf_serial: ABF file serial number
            img_serial: Image file serial number
            animal_idx: Batch-local sequential animal index (see build_animal_index_map)
            animal_id: Real ANIMAL_ID (e.g. "neoChAT-677"), stored in the DB record
            slice_val: SLICE value verbatim (e.g. "2R")
            at: AT location (e.g. "SITE_1"/"CELL_1")
            detrend_mode: Detrend mode used ("BIEXP"/"MOV")
            normalization: Normalization used ("GAUSS"/"ALS")
            num_found_spikes: Total number of spikes detected
            n_spikes_analyzed: Number of spikes analyzed
            threshold_method: Threshold method used for categorization
            objective: Microscope objective used
            um_per_pixel: Micrometers per pixel scale
            median_stack: Spike-centered median z-score stack
            categorized_frames: Categorized frames (0=bg, 1=dim, 2=bright)
            zscore_range: (min, max) z-score across median_stack, from spike_centered_median()
            region_summary: Summary dict from RegionAnalyzer.get_summary()
            region_data: Spike-frame region dict from RegionAnalyzer.get_results()
            peak_latency_ms: Dim-minus-bright peak latency, from RegionAnalyzer.get_peak_latency_ms()

        Returns:
            dict with keys "median", "categorized" → Path to each subfolder
        """
        dirs = {
            "median": self.results_root / "median",
            "categorized": self.results_root / "categorized",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self._export_median_stack(
            dirs["median"], median_stack, exp_date, img_serial, animal_idx, slice_val, at, detrend_mode, normalization
        )
        self._export_categorized_stack(
            dirs["categorized"], categorized_frames, exp_date, img_serial, animal_idx, slice_val, at, detrend_mode, normalization
        )

        med_stem = self.build_export_stem(exp_date, img_serial, animal_idx, slice_val, at, detrend_mode, normalization, "MED")

        self._upsert_record(
            exp_date=exp_date,
            abf_serial=abf_serial,
            img_serial=img_serial,
            num_found_spikes=num_found_spikes,
            n_spikes_analyzed=n_spikes_analyzed,
            threshold_method=threshold_method,
            objective=objective,
            um_per_pixel=um_per_pixel,
            region_summary=region_summary,
            region_data=region_data,
            zscore_range=zscore_range,
            animal_id=animal_id,
            slice_val=slice_val,
            at=at,
            peak_latency_ms=peak_latency_ms,
            med_filename=f"{med_stem}.tif",
        )

        return dirs

    def export_figure(self, category: str, figure: "Figure", filename: str) -> Path:
        """
        Save a figure under results_root/{category}/, creating the folder on demand.

        Args:
            category: Subfolder name (e.g. "region_sta")
            figure: Matplotlib figure to save
            filename: Name of the output file

        Returns:
            Path the figure was saved to
        """
        out_dir = self.results_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        figure.savefig(out_path)
        return out_path

    def _export_median_stack(
        self,
        files_dir: Path,
        median_stack: np.ndarray,
        exp_date: str,
        img_serial: str,
        animal_idx: int,
        slice_val: str,
        at: str,
        detrend_mode: str,
        normalization: str,
    ) -> None:
        """Save the spike-centered median stack as a float32 TIFF."""
        stem = self.build_export_stem(exp_date, img_serial, animal_idx, slice_val, at, detrend_mode, normalization, "MED")
        tifffile.imwrite(files_dir / f"{stem}.tif", median_stack.astype(np.float32))

    def _export_categorized_stack(
        self,
        files_dir: Path,
        categorized_frames: list[np.ndarray],
        exp_date: str,
        img_serial: str,
        animal_idx: int,
        slice_val: str,
        at: str,
        detrend_mode: str,
        normalization: str,
    ) -> None:
        """Save the categorized stack (0=bg, 1=dim, 2=bright) as a uint8 TIFF."""
        stem = self.build_export_stem(exp_date, img_serial, animal_idx, slice_val, at, detrend_mode, normalization, "CAT")
        tifffile.imwrite(files_dir / f"{stem}.tif", np.array(categorized_frames, dtype=np.uint8))

    def _upsert_record(
        self,
        exp_date: str,
        abf_serial: str,
        img_serial: str,
        num_found_spikes: int,
        n_spikes_analyzed: int,
        threshold_method: str,
        objective: str,
        um_per_pixel: float,
        region_summary: dict,
        region_data: dict,
        zscore_range: tuple[float, float],
        animal_id: str,
        slice_val: str,
        at: str,
        peak_latency_ms: float | None,
        med_filename: str,
    ) -> None:
        """Insert or update experiment record in SQLite.

        Note: region_data is optimized to strip contours/masks before JSON
        serialization (see optimize_region_data), keeping the blob small.
        """
        optimized_region_data = optimize_region_data(region_data)

        bright_largest = optimized_region_data.get("bright_largest")
        if bright_largest:
            centroid_y, centroid_x = bright_largest["centroid"]
            x_span_pixels = bright_largest["x_span_px"]
            y_span_pixels = bright_largest["y_span_px"]
            x_span_um = bright_largest["x_span_um"]
            y_span_um = bright_largest["y_span_um"]
            bright_area_um2 = bright_largest["area_um2"]
        else:
            centroid_y = centroid_x = x_span_pixels = y_span_pixels = x_span_um = y_span_um = bright_area_um2 = None

        dim_largest = optimized_region_data.get("dim_largest")
        if dim_largest:
            dim_area_um2 = dim_largest["area_um2"]
            dim_x_span_um = dim_largest["x_span_um"]
            dim_y_span_um = dim_largest["y_span_um"]
        else:
            dim_area_um2 = dim_x_span_um = dim_y_span_um = None

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                exp_date, abf_serial, img_serial, timestamp,
                objective, um_per_pixel, threshold_method,
                n_spikes_detected, n_spikes_analyzed,
                has_bright_region, has_dim_region,
                region_analysis, ANIMAL_ID, SLICE, AT, med_filename,
                centroid_y, centroid_x, x_span_pixels, y_span_pixels, x_span_um, y_span_um,
                bright_area_um2, dim_area_um2, dim_x_span_um, dim_y_span_um, peak_latency_ms,
                zscore_min, zscore_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_date,
                abf_serial,
                img_serial,
                datetime.now(UTC).isoformat(),
                objective,
                um_per_pixel,
                threshold_method,
                num_found_spikes,
                n_spikes_analyzed,
                region_summary["has_bright_region"],
                region_summary["has_dim_region"],
                json.dumps(optimized_region_data, cls=NumpyEncoder),
                animal_id,
                slice_val,
                at,
                med_filename,
                centroid_y,
                centroid_x,
                x_span_pixels,
                y_span_pixels,
                x_span_um,
                y_span_um,
                bright_area_um2,
                dim_area_um2,
                dim_x_span_um,
                dim_y_span_um,
                peak_latency_ms,
                zscore_range[0],
                zscore_range[1],
            ),
        )
        conn.commit()
        conn.close()
