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
                             [--db data/rec_data.db] [--exp_db data/exp_info.db]
"""
# Standard library imports
import argparse
import time
from pathlib import Path

# Third-party imports
import polars as pl
from rich.console import Console

# Local application imports
from classes import AbfClip, RegionAnalyzer, ResultsExporter, SpatialCategorizer
from functions import (
    count_unique_cells,
    list_parser,
    lookup_rec_from_db,
    plot_spatiotemporal_summary,
    spike_centered_median,
    write_cell_summary_xlsx,
    zscore_img_segs,
)

console = Console()


# ── Ana list parsing ──────────────────────────────────────────────────────────


def parse_ana_list(
    ana_list_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = False,
) -> tuple[pl.DataFrame, Path, str, str]:
    """Parse an ana list file and filter rows by existence flags.

    Args:
        ana_list_path: Path to the ana list file (ana_list_*.txt).
        detrend_mode:  Which detrend variant to load — "BIEXP" or "MOV".
        use_als:       If True, load *_ALS.tif; otherwise load *_GAUSS.tif.

    Returns:
        (entries, results_dir, detrend_mode, normalization). entries keeps every
        original ana-list column plus proc_tiff_path/raw_abf_path, restricted to
        rows passing the gauss_exist/als_exist and abf_exist guards. Resolving OBJ
        from rec_data.db is left to the caller (see lookup_rec_from_db).
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
    normalization = "ALS" if use_als else "GAUSS"

    passing_rows: list[dict] = []
    for row in table.iter_rows(named=True):
        if row[exist_col] != "YES":
            console.log(f"[yellow]Skipped {row['raw_tiff_name']}: {exist_col}={row[exist_col]}[/yellow]")
            continue
        if row["abf_exist"] != "YES":
            console.log(f"[yellow]Skipped {row['raw_tiff_name']}: abf_exist={row['abf_exist']}[/yellow]")
            continue

        stem = Path(row["raw_tiff_name"]).stem
        row["proc_tiff_path"] = str(proc_dir / f"{stem}_{detrend_mode}{suffix}")
        row["raw_abf_path"] = str(raw_abfs_dir / row["paired_abf"])
        passing_rows.append(row)

    entries = pl.DataFrame(passing_rows) if passing_rows else pl.DataFrame()
    return entries, results_dir, detrend_mode, normalization


# ── Pipeline runner ───────────────────────────────────────────────────────────


def run(
    ana_list_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = False,
    db_path: Path = Path("data/rec_data.db"),
    exp_db_path: Path = Path("data/exp_info.db"),
    emitter=None,
) -> None:
    """Run the full spike-aligned analysis pipeline for every entry in an ana list."""
    run_t0 = time.time()
    entries, results_dir, detrend_mode, normalization = parse_ana_list(ana_list_path, detrend_mode, use_als)
    ref_df = lookup_rec_from_db(entries, db_path, exp_db_path)
    cell_df = count_unique_cells(ref_df)
    console.log(f"Found {len(entries)} entries in {ana_list_path.name} -> {len(cell_df)} unique cells")

    xlsx_dir = results_dir / "xlsx"
    xlsx_dir.mkdir(parents=True, exist_ok=True)
    cell_summary_path = xlsx_dir / f"{ana_list_path.stem}_cells.xlsx"
    write_cell_summary_xlsx(cell_df, cell_summary_path)
    console.log(f"Saved cell summary -> {cell_summary_path.name}")

    animal_index_map = ResultsExporter.build_animal_index_map(ref_df["ANIMAL_ID"].unique())
    exporter = ResultsExporter(results_root=results_dir)

    total = len(entries)
    for i, row in enumerate(entries.iter_rows(named=True), 1):
        entry_t0 = time.time()
        match = ref_df.filter(pl.col("Filename") == row["raw_tiff_name"])
        if match.is_empty():
            console.log(f"[yellow]Skipped {row['raw_tiff_name']}: not found in rec_data.db[/yellow]")
            continue
        obj = match["OBJ"].item()

        proc_tiff_path = Path(row["proc_tiff_path"])
        raw_abf_path = Path(row["raw_abf_path"])
        if emitter:
            emitter({"type": "progress", "i": i, "total": total, "file": proc_tiff_path.name})
        console.log(f"\n[cyan]{proc_tiff_path.name}  +  {raw_abf_path.name}  [{obj}]  [{i}/{total}][/cyan]")
        clip = AbfClip(
            proc_tiff_path=proc_tiff_path,
            raw_abf_path=raw_abf_path,
            results_dir=results_dir,
            detrend_mode=detrend_mode,
            normalization=normalization,
        )

        if not clip.lst_img_frame_ranges:
            console.log("[yellow]No valid segments — skipping z-score step.[/yellow]")
            continue

        if emitter:
            emitter({"type": "step", "msg": "Z-score normalizing segments..."})
        # Coverts each segment to z-score normalized values using the baseline frames before the spike frame.
        lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
        console.log(f"[green]Z-score normalized {len(lst_zscore)} segment(s)  ({time.time() - entry_t0:.1f}s)[/green]")

        # Merge all segments into a single median segment, aligned by the spike frame.
        median_segment, zscore_range = spike_centered_median(lst_zscore)

        # Free memory from lst_zscore since it's no longer needed after computing the median.
        del lst_zscore
        console.log(f"[green]Median shape: {median_segment.shape}, z-score range: [{zscore_range[0]:.2f}, {zscore_range[1]:.2f}][/green]")

        if emitter:
            emitter({"type": "step", "msg": "Categorizing spike frame..."})
        # Real analysis starts here: categorize the z scores for further region analysis (using skimage.measure).
        spike_frame_idx = median_segment.shape[0] // 2
        categorizer = SpatialCategorizer.morphological(threshold_method="base977_otsu")
        categorizer.fit(median_segment, spike_frame_idx=spike_frame_idx)
        console.log(
            f"[green]Categorized {len(categorizer.categorized_frames)} frame(s), thresholds: {categorizer.thresholds_used}"
            f"  ({time.time() - entry_t0:.1f}s)[/green]"
        )

        region_analyzer = RegionAnalyzer(categorizer.categorized_frames[spike_frame_idx], obj=obj, min_area_um2=400)
        spike_frame = region_analyzer.get_results()

        bright_area = spike_frame["bright_largest"]
        dim_area = spike_frame["dim_largest"]
        if bright_area:
            console.log(
                f"[magenta]Bright (spike frame): area={bright_area['area_um2']:.1f} µm²  "
                f"x-span={bright_area['x_span_um']:.1f} µm  y-span={bright_area['y_span_um']:.1f} µm[/magenta]"
            )
        else:
            console.log("[yellow]No bright region in spike frame[/yellow]")
        if dim_area:
            console.log(
                f"[cyan]Dim (spike frame):    area={dim_area['area_um2']:.1f} µm²  "
                f"x-span={dim_area['x_span_um']:.1f} µm  y-span={dim_area['y_span_um']:.1f} µm[/cyan]"
            )
        else:
            console.log("[yellow]No dim region in spike frame[/yellow]")

        if emitter:
            emitter({"type": "step", "msg": "Exporting results..."})
        export_data = clip.get_export_data()
        animal_id = match["ANIMAL_ID"].item()
        slice_val = match["SLICE"].item()
        at = match["AT"].item()
        frame_duration_ms = clip.ts_imgs * 1000
        peak_latency_ms = region_analyzer.get_peak_latency_ms(median_segment, spike_frame_idx, frame_duration_ms)

        dirs = exporter.export_all(
            exp_date=export_data["exp_date"],
            abf_serial=export_data["abf_serial"],
            img_serial=export_data["img_serial"],
            animal_idx=animal_index_map[animal_id],
            animal_id=animal_id,
            slice_val=slice_val,
            at=at,
            detrend_mode=detrend_mode,
            normalization=normalization,
            num_found_spikes=export_data["num_found_spikes"],
            n_spikes_analyzed=export_data["n_spikes_analyzed"],
            threshold_method=categorizer.threshold_method,
            objective=obj,
            um_per_pixel=region_analyzer.um_per_pixel,
            median_stack=median_segment,
            categorized_frames=categorizer.categorized_frames,
            zscore_range=zscore_range,
            region_summary=region_analyzer.get_summary(),
            region_data=spike_frame,
            peak_latency_ms=peak_latency_ms,
        )

        title_info = {
            "animal_id": animal_id,
            "slice": slice_val,
            "at": at,
            "obj": obj,
            "tiff_serial": export_data["img_serial"],
            "abf_serial": export_data["abf_serial"],
        }
        fig = plot_spatiotemporal_summary(
            categorizer, region_analyzer, median_segment, spike_frame_idx, frame_duration_ms, title_info
        )
        stem = ResultsExporter.build_export_stem(
            export_data["exp_date"], export_data["img_serial"], animal_index_map[animal_id],
            slice_val, at, detrend_mode, normalization, "SPATIAL",
        )
        exporter.export_figure("region_sta", fig, f"{stem}.png")

        dir_names = "/, ".join(d.name for d in dirs.values())
        console.log(f"[green]✓ Exported {dir_names}/, region_sta/  (entry: {time.time() - entry_t0:.1f}s)[/green]")

    console.log(f"\n[bold green]All done!  (total: {time.time() - run_t0:.1f}s)[/bold green]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike-aligned image analysis pipeline")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_list_*.txt)")
    parser.add_argument("--detrend", choices=["BIEXP", "MOV"], default="BIEXP", help="Detrend mode (default: BIEXP)")
    parser.add_argument("--use_als", action="store_true", help="Load *_ALS.tif instead of *_GAUSS.tif")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db (default: data/rec_data.db)")
    parser.add_argument("--exp_db", type=Path, default=Path("data/exp_info.db"), help="Path to exp_info.db (default: data/exp_info.db)")
    args = parser.parse_args()

    run(args.ana_list, args.detrend, args.use_als, args.db, args.exp_db)

