"""
Results exporter for saving analysis outputs.

Exports analysis results to:
- SQLite database (metadata, summaries, region analysis)
- TIFF files (z-score stack, categorized frames)
- NPZ files (image segments, ABF segments)
- PNG figure (region plot snapshot)
"""

## Modules
# Standard library imports
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party imports
import numpy as np
import tifffile

# Local imports
from config_paths import PATHS

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


class ResultsExporter:
    """
    Export analysis results to files and SQLite database.

    Output structure:
        results/
        ├── results.db
        └── {exp_date}/
            └── abf{abf_serial}_img{img_serial}/
                ├── zscore_stack.tif
                ├── img_segments.npz
                ├── categorized_stack.tif
                ├── abf_segments.npz
                └── region_plot.png
    """

    def __init__(self, results_root: Path | None = None) -> None:
        """
        Initialize the ResultsExporter.

        Args:
            results_root: Root directory for results (default: from config_paths)
        """
        if results_root is None:
            results_root = PATHS["results"]
        self.results_root = Path(results_root)
        self.db_path = self.results_root / "results.db"
        self._init_db()

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
                n_frames INTEGER,
                total_dim_regions INTEGER,
                total_bright_regions INTEGER,
                dim_area_um2_mean REAL,
                dim_area_um2_std REAL,
                bright_area_um2_mean REAL,
                bright_area_um2_std REAL,
                region_analysis TEXT,
                data_dir TEXT,
                notes TEXT,
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
        # ABF metadata
        num_found_spikes: int,
        n_spikes_analyzed: int,
        # Processing metadata
        threshold_method: str,
        objective: str,
        um_per_pixel: float,
        # Data to save
        zscore_stack: np.ndarray,
        img_segments_zscore: list[np.ndarray],
        categorized_frames: list[np.ndarray],
        lst_time_segments: list[np.ndarray],
        lst_abf_segments: list[np.ndarray],
        # Analysis results
        region_summary: dict,
        region_data: dict,
    ) -> Path:
        """
        Export all results and update database.

        Args:
            exp_date: Experiment date string
            abf_serial: ABF file serial number
            img_serial: Image file serial number
            num_found_spikes: Total number of spikes detected
            n_spikes_analyzed: Number of spikes analyzed
            threshold_method: Threshold method used for categorization
            objective: Microscope objective used
            um_per_pixel: Micrometers per pixel scale
            zscore_stack: Spike-centered median z-score stack
            img_segments_zscore: List of z-score normalized image segments
            categorized_frames: List of categorized frames
            lst_time_segments: List of time segments from ABF
            lst_abf_segments: List of voltage segments from ABF
            region_summary: Summary statistics from region analysis
            region_data: Detailed region analysis results (with contours removed)

        Returns:
            Path to the experiment data directory
        """
        # Create experiment-specific directory: {date}/abf{}_img{}/
        exp_subdir = f"abf{abf_serial}_img{img_serial}"
        exp_dir = self.results_root / exp_date / exp_subdir
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save binary files
        self._export_zscore_stack(exp_dir, zscore_stack)
        self._export_img_segments(exp_dir, img_segments_zscore)
        # NOTE: categorized_stack.tif removed - use PNG plots instead
        self._export_abf_segments(exp_dir, lst_time_segments, lst_abf_segments)

        # Insert/update database record
        data_dir = f"{exp_date}/{exp_subdir}"
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
            data_dir=data_dir,
        )

        return exp_dir

    def export_figure(self, exp_dir: Path, figure: "Figure", filename: str = "plot.png") -> None:
        """
        Save figure to experiment directory.

        Args:
            exp_dir: Experiment data directory
            figure: Matplotlib figure to save (QPixmap from window.grab())
            filename: Name of the output file (default: "plot.png")
        """
        figure.save(str(exp_dir / filename))

    def _export_zscore_stack(self, exp_dir: Path, zscore_stack: np.ndarray) -> None:
        """Save z-score stack as TIFF."""
        tifffile.imwrite(exp_dir / "zscore_stack.tif", zscore_stack.astype(np.float32))

    def _export_img_segments(self, exp_dir: Path, img_segments_zscore: list) -> None:
        """Save image segments as compressed NPZ."""
        np.savez_compressed(exp_dir / "img_segments.npz", segments=np.array(img_segments_zscore, dtype=object))

    def _export_categorized_stack(self, exp_dir: Path, categorized_frames: list[np.ndarray]) -> None:
        """Save categorized frames as TIFF."""
        categorized = np.array(categorized_frames, dtype=np.uint8)
        tifffile.imwrite(exp_dir / "categorized_stack.tif", categorized)

    def _export_abf_segments(
        self, exp_dir: Path, lst_time_segments: list[np.ndarray], lst_abf_segments: list[np.ndarray]
    ) -> None:
        """Save ABF segments as compressed NPZ."""
        np.savez_compressed(
            exp_dir / "abf_segments.npz",
            time_segments=np.array(lst_time_segments, dtype=object),
            abf_segments=np.array(lst_abf_segments, dtype=object),
        )

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
        data_dir: str,
    ) -> None:
        """Insert or update experiment record in SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                exp_date, abf_serial, img_serial, timestamp,
                objective, um_per_pixel, threshold_method,
                n_spikes_detected, n_spikes_analyzed,
                n_frames, total_dim_regions, total_bright_regions,
                dim_area_um2_mean, dim_area_um2_std,
                bright_area_um2_mean, bright_area_um2_std,
                region_analysis, data_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                region_summary["n_frames"],
                region_summary["total_dim_regions"],
                region_summary["total_bright_regions"],
                region_summary["dim_area_um2_mean"],
                region_summary["dim_area_um2_std"],
                region_summary["bright_area_um2_mean"],
                region_summary["bright_area_um2_std"],
                json.dumps(region_data, cls=NumpyEncoder),
                data_dir,
            ),
        )
        conn.commit()
        conn.close()
