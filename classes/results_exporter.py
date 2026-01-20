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

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .abf_clip import AbfClip
    from .region_analyzer import RegionAnalyzer
    from .spatial_categorization import SpatialCategorizer


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

    def __init__(self, results_root: Path = Path(__file__).parent.parent / "results") -> None:
        """
        Initialize the ResultsExporter.

        Args:
            results_root: Root directory for results (default: "results")
        """
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
        abf_clip: "AbfClip",
        categorizer: "SpatialCategorizer",
        region_analyzer: "RegionAnalyzer",
        zscore_stack: np.ndarray,
        img_segments_zscore: list,
    ) -> Path:
        """
        Export all results and update database.

        Args:
            abf_clip: AbfClip instance with experiment data
            categorizer: SpatialCategorizer instance
            region_analyzer: RegionAnalyzer instance
            zscore_stack: Spike-centered median z-score stack
            img_segments_zscore: List of z-score normalized image segments

        Returns:
            Path to the experiment data directory
        """
        # Create experiment-specific directory: {date}/abf{}_img{}/
        exp_subdir = f"abf{abf_clip.abf_serial}_img{abf_clip.img_serial}"
        exp_dir = self.results_root / abf_clip.exp_date / exp_subdir
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save binary files
        self._export_zscore_stack(exp_dir, zscore_stack)
        self._export_img_segments(exp_dir, img_segments_zscore)
        self._export_categorized_stack(exp_dir, categorizer)
        self._export_abf_segments(exp_dir, abf_clip)

        # Insert/update database record
        data_dir = f"{abf_clip.exp_date}/{exp_subdir}"
        self._upsert_record(abf_clip, categorizer, region_analyzer, data_dir)

        return exp_dir

    def export_figure(self, exp_dir: Path, figure: "Figure") -> None:
        """
        Save figure to experiment directory.

        Args:
            exp_dir: Experiment data directory
            figure: Matplotlib figure to save
        """
        figure.save(str(exp_dir / "region_plot.png"))

    def _export_zscore_stack(self, exp_dir: Path, zscore_stack: np.ndarray) -> None:
        """Save z-score stack as TIFF."""
        tifffile.imwrite(exp_dir / "zscore_stack.tif", zscore_stack.astype(np.float32))

    def _export_img_segments(self, exp_dir: Path, img_segments_zscore: list) -> None:
        """Save image segments as compressed NPZ."""
        np.savez_compressed(exp_dir / "img_segments.npz", segments=np.array(img_segments_zscore, dtype=object))

    def _export_categorized_stack(self, exp_dir: Path, categorizer: "SpatialCategorizer") -> None:
        """Save categorized frames as TIFF."""
        categorized = np.array(categorizer.categorized_frames, dtype=np.uint8)
        tifffile.imwrite(exp_dir / "categorized_stack.tif", categorized)

    def _export_abf_segments(self, exp_dir: Path, abf_clip: "AbfClip") -> None:
        """Save ABF segments as compressed NPZ."""
        np.savez_compressed(
            exp_dir / "abf_segments.npz",
            time_segments=np.array(abf_clip.lst_time_segments, dtype=object),
            abf_segments=np.array(abf_clip.lst_abf_segments, dtype=object),
        )

    def _upsert_record(
        self, abf_clip: "AbfClip", categorizer: "SpatialCategorizer", region_analyzer: "RegionAnalyzer", data_dir: str
    ) -> None:
        """Insert or update experiment record in SQLite."""
        summary = region_analyzer.get_summary()
        region_data = region_analyzer.get_results()

        # Remove contours (not JSON serializable)
        region_data.pop("dim_contours", None)
        region_data.pop("bright_contours", None)

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
                abf_clip.exp_date,
                abf_clip.abf_serial,
                abf_clip.img_serial,
                datetime.now(UTC).isoformat(),
                region_analyzer.obj,
                region_analyzer.um_per_pixel,
                categorizer.threshold_method,
                abf_clip.num_found_spikes,
                len(abf_clip.df_picked_spikes),
                summary["n_frames"],
                summary["total_dim_regions"],
                summary["total_bright_regions"],
                summary["dim_area_um2_mean"],
                summary["dim_area_um2_std"],
                summary["bright_area_um2_mean"],
                summary["bright_area_um2_std"],
                json.dumps(region_data, cls=NumpyEncoder),
                data_dir,
            ),
        )
        conn.commit()
        conn.close()
