"""
spike_analysis.py  --  Spike-aligned image analysis pipeline.
=============================================================
Reads an analysis list (ana_list_*.txt), loads the appropriate processed TIFF
(*_GAUSS.tif or *_ALS.tif) together with its paired ABF file, runs spike
detection and alignment, spatial categorization, and region analysis.

For each entry, the objective (10X / 40X / 60X) is looked up automatically
from rec_data.db using the raw TIFF filename.

Ana list format (column names declared on the 'Picked:' line):
  [raw_tiff_name, gauss_exist, als_exist, paired_abf, abf_exist]

Detrend mode selects which processed TIFF prefix is loaded:
  BIEXP -> *_BIEXP_GAUSS.tif  or  *_BIEXP_ALS.tif
  MOV   -> *_MOV_GAUSS.tif    or  *_MOV_ALS.tif

Usage:
    python spike_analysis.py --ana_list data/ana_list_20260601_000.txt
                             [--detrend BIEXP] [--use_als]
                             [--db data/rec_data.db]
"""
# Standard library imports
import argparse
import sqlite3
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile
from rich.console import Console

# Local application imports
from classes import AbfClip, RegionAnalyzer, SpatialCategorizer
from functions import list_parser, spike_centered_median, zscore_img_segs

console = Console()


# ── DB metadata lookup ────────────────────────────────────────────────────────


def _lookup_obj(db_path: Path, raw_tiff_name: str) -> str | None:
    """Look up the objective for a raw TIFF filename in rec_data.db.

    Derives the table name from the date prefix of the filename
    (e.g. '2025_12_15-0042.tif' → table 'REC_2025_12_15').

    Returns the OBJ string (e.g. '10X') or None if not found.
    """
    date = Path(raw_tiff_name).stem.split("-")[0]  # "2025_12_15"
    table = f"REC_{date}"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT OBJ FROM {table} WHERE Filename = ?", (raw_tiff_name,))
        row = cur.fetchone()
        conn.close()
    except sqlite3.OperationalError:
        return None
    return row[0] if row else None


# ── Ana list parsing ──────────────────────────────────────────────────────────


def parse_ana_list(
    ana_list_path: Path,
    db_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = False,
) -> tuple[list[dict], Path, str, str]:
    """Parse an ana list file and return entries with resolved paths and metadata.

    Args:
        ana_list_path: Path to the ana list file (ana_list_*.txt).
        db_path:       Path to rec_data.db for objective lookup.
        detrend_mode:  Which detrend variant to load — "BIEXP" or "MOV".
        use_als:       If True, load *_ALS.tif; otherwise load *_GAUSS.tif.

    Returns:
        (entries, results_dir, detrend_mode, normalization) where each entry is:
            {"proc_tiff_path": Path, "raw_abf_path": Path, "obj": str}
        Only entries passing all existence and DB-lookup guards are included.
    """
    table, io_dirs = list_parser(ana_list_path)

    missing = [k for k in ("dir_proc_tiffs", "dir_raw_abfs", "dir_results") if k not in io_dirs]
    if missing:
        msg = f"Missing footer keys in {ana_list_path}: {', '.join(missing)}"
        raise ValueError(msg)
    proc_dir = Path(io_dirs["dir_proc_tiffs"])
    raw_abfs_dir = Path(io_dirs["dir_raw_abfs"])
    results_dir = Path(io_dirs["dir_results"])

    exist_col = "als_exist" if use_als else "gauss_exist"
    suffix = "_ALS.tif" if use_als else "_GAUSS.tif"

    entries: list[dict] = []
    for row in table.iter_rows(named=True):
        if row[exist_col] != "YES":
            console.print(f"[yellow]Skipped {row['raw_tiff_name']}: {exist_col}={row[exist_col]}[/yellow]")
            continue
        if row["abf_exist"] != "YES":
            console.print(f"[yellow]Skipped {row['raw_tiff_name']}: abf_exist={row['abf_exist']}[/yellow]")
            continue

        obj = _lookup_obj(db_path, row["raw_tiff_name"])
        if obj is None:
            console.print(f"[yellow]Skipped {row['raw_tiff_name']}: not found in rec_data.db[/yellow]")
            continue

        stem = Path(row["raw_tiff_name"]).stem
        entries.append({
            "proc_tiff_path": proc_dir / f"{stem}_{detrend_mode}{suffix}",
            "raw_abf_path":   raw_abfs_dir / row["paired_abf"],
            "obj":            obj,
        })

    normalization = "ALS" if use_als else "GAUSS"
    return entries, results_dir, detrend_mode, normalization


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike-aligned image analysis pipeline")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_list_*.txt)")
    parser.add_argument("--detrend", choices=["BIEXP", "MOV"], default="BIEXP", help="Detrend mode (default: BIEXP)")
    parser.add_argument("--use_als", action="store_true", help="Load *_ALS.tif instead of *_GAUSS.tif")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db (default: data/rec_data.db)")
    args = parser.parse_args()

    entries, results_dir, detrend_mode, normalization = parse_ana_list(
        args.ana_list, args.db, args.detrend, args.use_als
    )
    console.print(f"Found {len(entries)} entries in {args.ana_list.name}")

    for entry in entries:
        console.print(f"\n[cyan]{entry['proc_tiff_path'].name}  +  {entry['raw_abf_path'].name}  [{entry['obj']}][/cyan]")
        clip = AbfClip(
            proc_tiff_path=entry["proc_tiff_path"],
            raw_abf_path=entry["raw_abf_path"],
            results_dir=results_dir,
            detrend_mode=detrend_mode,
            normalization=normalization,
        )

        if not clip.lst_img_frame_ranges:
            console.print("[yellow]No valid segments — skipping z-score step.[/yellow]")
            continue

        # Coverts each segment to z-score normalized values using the baseline frames before the spike frame.
        lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
        console.print(f"[green]Z-score normalized {len(lst_zscore)} segment(s)[/green]")

        # Merge all segments into a single median segment, aligned by the spike frame.
        median_segment, zscore_range = spike_centered_median(lst_zscore)

        # Free memory from lst_zscore since it's no longer needed after computing the median.
        del lst_zscore
        console.print(f"[green]Median shape: {median_segment.shape}, z-score range: [{zscore_range[0]:.2f}, {zscore_range[1]:.2f}][/green]")

        median_path = results_dir / f"{clip.proc_tiff_path.stem}_MED.tif"
        tifffile.imwrite(median_path, median_segment.astype(np.float32))
        console.print(f"[green]Saved median → {median_path.name}[/green]")

        # Real analysis starts here: categorize the z scores for further region analysis (using skimage.measure).
        categorizer = SpatialCategorizer.morphological(threshold_method="otsu_double")
        categorizer.fit(median_segment)
        console.print(f"[green]Categorized {len(categorizer.categorized_frames)} frame(s), thresholds: {categorizer.thresholds_used}[/green]")

        cat_path = results_dir / f"{clip.proc_tiff_path.stem}_CAT.tif"
        tifffile.imwrite(cat_path, np.array(categorizer.categorized_frames, dtype=np.uint8))
        console.print(f"[green]Saved categorized → {cat_path.name}[/green]")

        region_analyzer = RegionAnalyzer(categorizer.categorized_frames, obj=entry["obj"])
        spike_frame_idx = len(categorizer.categorized_frames) // 2
        spike = region_analyzer.get_frame_results(spike_frame_idx)

        bright = spike["bright_largest"]
        dim = spike["dim_largest"]
        if bright:
            console.print(
                f"[magenta]Bright (spike frame): area={bright['area_um2']:.1f} µm²  "
                f"x-span={bright['x_span_um']:.1f} µm  y-span={bright['y_span_um']:.1f} µm[/magenta]"
            )
        else:
            console.print("[yellow]No bright region in spike frame[/yellow]")
        if dim:
            console.print(
                f"[cyan]Dim (spike frame):    area={dim['area_um2']:.1f} µm²  "
                f"x-span={dim['x_span_um']:.1f} µm  y-span={dim['y_span_um']:.1f} µm[/cyan]"
            )
        else:
            console.print("[yellow]No dim region in spike frame[/yellow]")

