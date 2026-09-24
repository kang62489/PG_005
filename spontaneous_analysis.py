"""
spontaneous_analysis.py  --  Spontaneous ACh hotspot -> zone analysis (10X, *_BIEXP_ALS.tif).

  Step 1. Select  : proc list -> recordings with an ALS tiff; OBJ / SENSOR from rec_data.db, 10X only
  Step 2. Analyze : per recording, SpontaneousZoneAnalyzer detect -> group -> map
  Step 3. Export  : per recording {stem}_ZONES.xlsx / _ZONE_MASK.tif / _ZONES.npz + zone-map PNGs
  Step 4. Summary : spontaneous_summary.xlsx (one row per recording + pooled zone table)
                    + spontaneous_stats.png (zone area / frequency per sensor)

All outputs go to {results_dir}/spontaneous/.

Usage:
    python spontaneous_analysis.py --proc_list data/proc_20260924_000.txt
        [--results_dir results] [--sigma 1.5] [--proj mean|max] [--color gray|red|green|blue]
        [--all_obj] [--debug]
"""

## Modules
# Standard library imports
import argparse
import time
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import tifffile
from rich.console import Console

# Local imports
from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import timed
from functions import (
    check_cuda,
    list_parser,
    lookup_rec_from_db,
    plot_single_zone,
    plot_zone_overlay,
    plot_zone_stats,
    zone_colors,
)

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

TARGET_OBJ = "10X"
PROC_SUFFIX = "_BIEXP_ALS.tif"
MAP_DPI = 120


# ===========================================================================
#
#   STEP 1 -- SELECT: proc list -> 10X ALS recordings with sensor info
#
# ===========================================================================

def select_recordings(proc_list_path: Path, db_path: Path, exp_db_path: Path, all_obj: bool) -> pl.DataFrame:
    """Rows of (raw_tiff_name, proc_tiff_path, OBJ, SENSOR) for recordings that pass the filters."""
    table, io_dirs = list_parser(proc_list_path)
    proc_dir = Path(io_dirs["dir_proc_tiffs"])

    rec_info = lookup_rec_from_db(table.select("raw_tiff_name"), db_path, exp_db_path)
    selected: list[dict] = []
    for name in table["raw_tiff_name"].to_list():
        proc_tiff_path = proc_dir / f"{Path(name).stem}{PROC_SUFFIX}"
        match = rec_info.filter(pl.col("Filename") == name) if not rec_info.is_empty() else rec_info
        if match.is_empty():
            console.log(f"[yellow]Skipped {name}: not in rec_data.db[/yellow]")
            continue
        obj, sensor = match["OBJ"].item(), match["SENSOR"].item()
        if not proc_tiff_path.exists():
            console.log(f"[yellow]Skipped {name}: {proc_tiff_path.name} not found[/yellow]")
            continue
        if obj != TARGET_OBJ and not all_obj:
            console.log(f"[yellow]Skipped {name}: OBJ={obj} (only {TARGET_OBJ})[/yellow]")
            continue
        selected.append({"raw_tiff_name": name, "proc_tiff_path": str(proc_tiff_path), "OBJ": obj, "SENSOR": sensor})

    return pl.DataFrame(selected)


# ===========================================================================
#
#   STEP 3 -- EXPORT: zone-map PNGs
#
# ===========================================================================

def export_zone_maps(analyzer: SpontaneousZoneAnalyzer, background: np.ndarray, bg_color: str,
                     title_tag: str, out_dir: Path) -> int:
    """Write 01_all_zones.png + one NN_zoneMM.png per zone; return zone count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in out_dir.glob("*.png"):
        stale_png.unlink()

    zone_ids = sorted(analyzer.zone_masks)
    colors = zone_colors(zone_ids)
    width = len(str(len(zone_ids) + 1))
    zone_width = len(str(max(zone_ids, default=0)))

    n_by_source = analyzer.zone_stats["source"].str.split(" #").str[0].value_counts()
    overlay_title = (f"{len(zone_ids)} zones -- {n_by_source.get('trace_corr', 0)} trace-corr, "
                     f"{n_by_source.get('proximity', 0)} proximity, {n_by_source.get('isolated', 0)} isolated "
                     f"({title_tag})")
    fig = plot_zone_overlay(analyzer.zone_masks, analyzer.zone_centroids, background, bg_color, overlay_title)
    fig.savefig(out_dir / f"{1:0{width}d}_all_zones.png", dpi=MAP_DPI)

    for i, zone_id in enumerate(zone_ids, start=2):
        fig = plot_single_zone(zone_id, analyzer.zone_masks[zone_id], colors[zone_id],
                               analyzer.zone_centroids.get(zone_id), background, bg_color,
                               f"zone {zone_id} ({title_tag})")
        fig.savefig(out_dir / f"{i:0{width}d}_zone{zone_id:0{zone_width}d}.png", dpi=MAP_DPI)

    return len(zone_ids)


# ===========================================================================
#
#   RUN
#
# ===========================================================================

def run(proc_list_path: Path, results_dir: Path = Path("results"), sigma: float = 1.5, proj: str = "mean",
        color: str = "gray", all_obj: bool = False, debug: bool = False, cuda_available: bool = False,
        db_path: Path = Path("data/rec_data.db"), exp_db_path: Path = Path("data/exp_info.db")) -> None:
    """Run the spontaneous zone analysis for every selected recording in a proc list."""
    run_t0 = time.time()
    out_root = results_dir / "spontaneous"
    out_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1. Select
    # -----------------------------------------------------------------------
    console.rule("[bold]Step 1 - Select")
    recordings = select_recordings(proc_list_path, db_path, exp_db_path, all_obj)
    console.log(f"{len(recordings)} recording(s) selected from {proc_list_path.name}")

    summary_rows: list[dict] = []
    pooled_zones: list[pd.DataFrame] = []

    for i, row in enumerate(recordings.iter_rows(named=True), 1):
        entry_t0 = time.time()
        proc_tiff_path = Path(row["proc_tiff_path"])
        stem = proc_tiff_path.stem
        console.rule(f"[bold]{stem}  [{row['OBJ']}, {row['SENSOR']}]  [{i}/{len(recordings)}]")

        # -------------------------------------------------------------------
        # Step 2. Analyze
        # -------------------------------------------------------------------
        with timed("load   read tiff + float16 copy"):
            stack = tifffile.imread(proc_tiff_path)
            analyzer = SpontaneousZoneAnalyzer(stack, obj=row["OBJ"], sigma_ratio=sigma, cuda_available=cuda_available)
        analyzer.run()

        # -------------------------------------------------------------------
        # Step 3. Export
        # -------------------------------------------------------------------
        with timed("export xlsx / mask tif / npz"):
            paths = analyzer.save(out_root, stem, debug=debug)
        with timed(f"export background ({proj} projection)"):
            background = stack.max(axis=0) if proj == "max" else stack.mean(axis=0)
            del stack
        map_dir = out_root / "zone_maps" / stem
        with timed(f"export zone-map PNGs ({len(analyzer.zone_masks) + 1})"):
            n_zones = export_zone_maps(analyzer, background, color, f"{stem}, {row['SENSOR']}, {proj} proj", map_dir)
        for path in paths.values():
            console.log(f"[green]saved[/green] {path.resolve()}")
        console.log(f"[green]saved[/green] {n_zones + 1} PNGs -> {map_dir.resolve()}")

        zone_stats = analyzer.zone_stats
        summary_rows.append({
            "recording": stem,
            "sensor": row["SENSOR"],
            "obj": row["OBJ"],
            "n_frames": analyzer.n_frames,
            "background_threshold": analyzer.threshold,
            "n_zones": len(zone_stats),
            "n_trace_corr": int(zone_stats["source"].str.startswith("trace_corr").sum()),
            "n_proximity": int(zone_stats["source"].str.startswith("proximity").sum()),
            "n_isolated": int(zone_stats["source"].str.startswith("isolated").sum()),
            "median_area_um2": float(zone_stats["area_um2"].median()),
            "median_freq_hz": float(zone_stats["mean_freq_hz"].median()),
            "median_period_s": float(zone_stats["mean_period_s"].median()),
            "n_high_freq_zones": int(zone_stats["high_freq_flag"].sum()),
        })
        pooled_zones.append(zone_stats.assign(recording=stem, sensor=row["SENSOR"]))
        console.log(f"[bold magenta]{'entry total':<40} {time.time() - entry_t0:6.1f}s[/bold magenta]")

    # -----------------------------------------------------------------------
    # Step 4. Summary
    # -----------------------------------------------------------------------
    console.rule("[bold]Step 4 - Summary")
    if summary_rows:
        summary_path = out_root / "spontaneous_summary.xlsx"
        pooled = pd.concat(pooled_zones, ignore_index=True)
        pooled = pooled[["recording", "sensor", *[c for c in pooled.columns if c not in ("recording", "sensor")]]]
        with pd.ExcelWriter(summary_path) as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="recordings", index=False)
            pooled.to_excel(writer, sheet_name="zones", index=False)
        console.log(f"[green]saved[/green] {summary_path.resolve()}")

        stats_path = out_root / "spontaneous_stats.png"
        fig = plot_zone_stats(pooled, f"Spontaneous ACh zones -- {len(summary_rows)} recording(s), {len(pooled)} zones")
        fig.savefig(stats_path, dpi=150, bbox_inches="tight")
        console.log(f"[green]saved[/green] {stats_path.resolve()}")

    console.rule(f"[dim]Total time: {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spontaneous ACh hotspot -> zone analysis")
    parser.add_argument("--proc_list", required=True, type=Path, help="Proc list (proc_*.txt) naming the recordings")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Outputs go to <results_dir>/spontaneous/")
    parser.add_argument("--sigma", type=float, default=1.5, help="Threshold = background peak + this many sigmas")
    parser.add_argument("--proj", choices=["mean", "max"], default="mean", help="Zone-map background projection")
    parser.add_argument("--color", choices=["gray", "red", "green", "blue"], default="gray", help="Background tint")
    parser.add_argument("--all_obj", action="store_true", help=f"Also analyze non-{TARGET_OBJ} recordings (testing only)")
    parser.add_argument("--debug", action="store_true", help="Also save the raw per-frame detections CSV")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=Path("data/exp_info.db"), help="Path to exp_info.db")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda()  # must run before anything imports numba
    console.log(_cuda_msg)
    run(args.proc_list, args.results_dir, args.sigma, args.proj, args.color, args.all_obj, args.debug,
        _cuda_available, args.db, args.exp_db)
