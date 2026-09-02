"""
Results exporter for saving analysis outputs.

Exports analysis results to:
- SQLite database (metadata, critical-frame cluster measurements)
- TIFF files (spike-centered median stack, categorized frames for ImageJ overlay)
- PNG figures (spatiotemporal summary plot)

Filenames use a compact code: {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_{TYPE}.
See ResultsExporter's class docstring for the full folder layout.
"""

## Modules
# Standard library imports
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np
import polars as pl
import tifffile

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_SITE_NUMBER = re.compile(r"(\d+)$")


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
        ├── spatial/
        │   └── {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_SPATIAL.png
        └── latency/
            └── {exp_date}-{img_serial}_A{n}S{slice}C{site}_{detrend}_{normalization}_LATENCY.png

    spatial/ and latency/ are created on demand by export_figure() — export_all() only creates
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
    def build_animal_index_map(ref_df: pl.DataFrame) -> dict[str, dict[str, int]]:
        """Map each DOR to a 1-based sequential index per distinct ANIMAL_ID recorded that day.

        A{n} resets every DOR (date prefix of Filename, e.g. "2024_02_15-0042.tif" ->
        "2024_02_15") since exp_date already appears in every export filename. Call
        this once per ana-list run with ref_df's Filename + ANIMAL_ID columns.
        """
        dor_df = ref_df.with_columns(ref_df["Filename"].str.split("-").list.first().alias("_dor"))
        index_map: dict[str, dict[str, int]] = {}
        for (dor,), group in dor_df.group_by(["_dor"], maintain_order=True):
            animal_ids = sorted(set(group["ANIMAL_ID"].to_list()))
            index_map[dor] = {animal_id: i + 1 for i, animal_id in enumerate(animal_ids)}
        return index_map

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
                n_clusters INTEGER,
                has_region INTEGER,
                critical_frame_offset INTEGER,
                critical_frame_area_pct REAL,
                critical_frame_area_um2 REAL,
                max_area_offset INTEGER,
                max_area_um2 REAL,
                max_area_eq_radius_um REAL,
                max_area_x_span_um REAL,
                max_area_y_span_um REAL,
                decay_peak_offset INTEGER,
                decay_fit_r2 REAL,
                lasting_time_ms REAL,
                ANIMAL_ID TEXT,
                SLICE TEXT,
                AT TEXT,
                med_filename TEXT,
                centroid_y REAL,
                centroid_x REAL,
                R_lat_px REAL,
                R_lat_um REAL,
                peak_latency_ms REAL,
                zscore_min REAL,
                zscore_max REAL,
                UNIQUE(exp_date, abf_serial, img_serial)
            )
        """)
        self._ensure_columns(
            conn,
            {
                "max_area_x_span_um": "REAL",
                "max_area_y_span_um": "REAL",
                "decay_peak_offset": "INTEGER",
                "decay_fit_r2": "REAL",
                "lasting_time_ms": "REAL",
            },
        )
        conn.commit()
        conn.close()

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, columns: dict[str, str]) -> None:
        """Add missing columns to an existing experiments table."""
        existing = {row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
        for column_name, column_type in columns.items():
            if column_name not in existing:
                conn.execute(f"ALTER TABLE experiments ADD COLUMN {column_name} {column_type}")

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
        lasting_time_ms: float | None,
        significant: bool = True,
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
            detrend_mode: Detrend mode used ("BIEXP")
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
            region_data: Critical-frame cluster dict from RegionAnalyzer.get_results()
            peak_latency_ms: Peak-timing latency, from RegionAnalyzer.get_peak_latency_ms()
            lasting_time_ms: Decay time constant, from RegionAnalyzer.get_lasting_time_ms()
            significant: When False, skips MED/CAT TIFF writes (no ACh detected).

        Returns:
            dict with keys "median", "categorized" → Path to each subfolder
        """
        dirs = {
            "median": self.results_root / "median",
            "categorized": self.results_root / "categorized",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        if significant:
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
            lasting_time_ms=lasting_time_ms,
            med_filename=f"{med_stem}.tif",
        )

        return dirs

    def export_figure(self, category: str, figure: "Figure", filename: str) -> Path:
        """
        Save a figure under results_root/{category}/, creating the folder on demand.

        Args:
            category: Subfolder name (e.g. "spatial")
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
        lasting_time_ms: float | None,
        med_filename: str,
    ) -> None:
        """Insert or update experiment record in SQLite."""
        clusters = region_data["clusters"]
        if clusters:
            # Clusters are sorted largest-first by _run_cluster_seeker.
            largest = clusters[0]
            centroid_y, centroid_x = largest["centroid"]
            r_lat_px = largest["R_lat_px"]
            r_lat_um = largest["R_lat_um"]
        else:
            centroid_y = centroid_x = r_lat_px = r_lat_um = None

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                exp_date, abf_serial, img_serial, timestamp,
                objective, um_per_pixel, threshold_method,
                n_spikes_detected, n_spikes_analyzed,
                n_clusters, has_region,
                critical_frame_offset, critical_frame_area_pct, critical_frame_area_um2,
                max_area_offset, max_area_um2, max_area_eq_radius_um, max_area_x_span_um, max_area_y_span_um,
                decay_peak_offset, decay_fit_r2, lasting_time_ms,
                ANIMAL_ID, SLICE, AT, med_filename,
                centroid_y, centroid_x, R_lat_px, R_lat_um, peak_latency_ms,
                zscore_min, zscore_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                region_summary["n_clusters"],
                region_summary["has_region"],
                region_data["critical_frame_offset"],
                region_data["critical_frame_area_pct"],
                region_data["critical_frame_area_um2"],
                region_data["max_area_offset"],
                region_data["max_area_um2"],
                region_data["max_area_eq_radius_um"],
                region_data["max_area_x_span_um"],
                region_data["max_area_y_span_um"],
                region_data["decay_peak_offset"],
                region_data["decay_fit_r2"],
                lasting_time_ms,
                animal_id,
                slice_val,
                at,
                med_filename,
                centroid_y,
                centroid_x,
                r_lat_px,
                r_lat_um,
                peak_latency_ms,
                zscore_range[0],
                zscore_range[1],
            ),
        )
        conn.commit()
        conn.close()
