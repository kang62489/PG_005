"""
Results exporter for saving analysis outputs.

Exports analysis results to:
- SQLite database (metadata, summaries, optimized region analysis with largest regions only)
- TIFF files (z-score stack, categorized frames for ImageJ overlay)
- PNG figures (spatial plot, region plot)

Note: region_analysis in database is optimized to store only largest regions per frame,
reducing database size from ~700MB to <1MB while retaining essential data.
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def optimize_region_data(region_data: dict) -> dict:
    """Optimize region data by keeping only largest bright regions.

    Reduces JSON size from ~5MB to ~5KB per experiment by removing:
    - dim_regions (all regions)
    - bright_regions (all regions)
    - dim_category (category stats)
    - bright_category (category stats)
    - dim_largest (not needed)

    Keeps only:
    - bright_largest (largest bright region per frame with spans)
    - obj (microscope objective)
    - um_per_pixel (scale factor)

    Args:
        region_data: Full region analysis dict from RegionAnalyzer.get_results()

    Returns:
        Optimized dict with only largest bright regions
    """
    return {
        "bright_largest": region_data.get("bright_largest"),
        "obj": region_data.get("obj"),
        "um_per_pixel": region_data.get("um_per_pixel"),
    }


class ResultsExporter:
    """
    Export analysis results to files and SQLite database.

    Output structure:
        results/
        ├── results.db
        └── files/
            ├── {date}-img{img}-abf{abf}_zscore.tif
            ├── {date}-img{img}-abf{abf}_categorized.tif
            ├── {date}-img{img}-abf{abf}_spatial_plot.png
            └── {date}-img{img}-abf{abf}_region_plot.png
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
                region_analysis TEXT,
                data_dir TEXT,
                SLICE INTEGER,
                AT TEXT,
                centroid_y REAL,
                centroid_x REAL,
                x_span_pixels REAL,
                y_span_pixels REAL,
                x_span_um REAL,
                y_span_um REAL,
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
        categorized_frames: list[np.ndarray],
        # Analysis results
        region_summary: dict,
        region_data: dict,
        # Optional metadata
        slice_num: int | None = None,
        at: str | None = None,
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
            categorized_frames: List of categorized frames (0=bg, 1=dim, 2=bright)
            region_summary: Summary statistics from region analysis
            region_data: Detailed region analysis results (with contours removed)
            slice_num: Slice number (optional)
            at: AT location (optional)

        Returns:
            Path to the files directory
        """
        files_dir = self.results_root / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        exp_prefix = f"{exp_date}-img{img_serial}-abf{abf_serial}"

        self._export_zscore_stack(files_dir, zscore_stack, exp_prefix)
        self._export_categorized_stack(files_dir, categorized_frames, exp_prefix)

        # Insert/update database record
        data_dir = "files"
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
            slice_num=slice_num,
            at=at,
        )

        return files_dir

    def export_figure(self, exp_dir: Path, figure: "Figure", filename: str = "plot.png") -> None:
        """
        Save figure to experiment directory.

        Args:
            exp_dir: Experiment data directory
            figure: Matplotlib figure to save (QPixmap from window.grab())
            filename: Name of the output file (default: "plot.png")
        """
        figure.save(str(exp_dir / filename))

    def _export_zscore_stack(self, files_dir: Path, zscore_stack: np.ndarray, exp_prefix: str) -> None:
        """Save z-score stack as TIFF."""
        tifffile.imwrite(files_dir / f"{exp_prefix}_zscore.tif", zscore_stack.astype(np.float32))

    def _export_categorized_stack(self, files_dir: Path, categorized_frames: list[np.ndarray], exp_prefix: str) -> None:
        """Save categorized frames as uint8 TIFF for ImageJ overlay (0=bg, 1=dim, 2=bright)."""
        categorized = np.array(categorized_frames, dtype=np.uint8)
        tifffile.imwrite(files_dir / f"{exp_prefix}_categorized.tif", categorized)

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
        slice_num: int | None = None,
        at: str | None = None,
    ) -> None:
        """Insert or update experiment record in SQLite.

        Note: region_data is automatically optimized to store only largest regions,
        reducing database size by ~99.9% while retaining essential information.
        """
        # Optimize region_data to keep only largest regions (reduces JSON from ~5MB to ~5KB)
        optimized_region_data = optimize_region_data(region_data)

        # Extract spike frame (center frame) region data for individual columns
        n_frames = region_summary["n_frames"]
        spike_frame_idx = n_frames // 2
        bright_largest_list = optimized_region_data.get("bright_largest", [])
        spike_frame_region = bright_largest_list[spike_frame_idx] if spike_frame_idx < len(bright_largest_list) else None

        if spike_frame_region:
            centroid_y, centroid_x = spike_frame_region["centroid"]
            x_span_pixels = spike_frame_region["x_span_pixels"]
            y_span_pixels = spike_frame_region["y_span_pixels"]
            x_span_um = spike_frame_region["x_span_um"]
            y_span_um = spike_frame_region["y_span_um"]
        else:
            centroid_y = centroid_x = x_span_pixels = y_span_pixels = x_span_um = y_span_um = None

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                exp_date, abf_serial, img_serial, timestamp,
                objective, um_per_pixel, threshold_method,
                n_spikes_detected, n_spikes_analyzed,
                n_frames, total_dim_regions, total_bright_regions,
                region_analysis, data_dir, SLICE, AT,
                centroid_y, centroid_x, x_span_pixels, y_span_pixels, x_span_um, y_span_um
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(optimized_region_data, cls=NumpyEncoder),
                data_dir,
                slice_num,
                at,
                centroid_y,
                centroid_x,
                x_span_pixels,
                y_span_pixels,
                x_span_um,
                y_span_um,
            ),
        )
        conn.commit()
        conn.close()
