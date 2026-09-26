"""
spontaneous_analysis.py  --  Spontaneous ACh hotspot -> zone analysis (10X, *_BIEXP_ALS.tif).

  Step 1. Select  : proc list -> recordings with an ALS tiff; OBJ / SENSOR from rec_data.db, 10X only
  Step 2. Analyze : per recording, SpontaneousZoneAnalyzer detect -> group -> map
  Step 3. Export  : per recording {stem}_ZONES.xlsx + {stem}_ZONE_MAPS.tif (z-scored RGB stack: max projection
                    + all zones, then one page per detection frame) + footprints/{stem}_ZONES.npz
                    + mask/{stem}_ZONE_MASK.tif (off with --no_mask)
  Step 4. Summary : spontaneous_summary.xlsx (one row per recording + pooled zone table)

All outputs go to {results_dir}/spontaneous/.

Usage:
    python spontaneous_analysis.py --proc_list data/proc_20260924_000.txt
        [--results_dir results] [--sigma 2.0] [--no_mask] [--all_obj] [--debug]
"""

## Modules
# Standard library imports
import argparse
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rich.console import Console

# Local imports
from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import CROSSOVER_RATIO, MIN_EVENTS_FOR_FREQ, timed
from functions import (
    check_cuda,
    img_zscore_convert,
    list_parser,
    lookup_rec_from_db,
    plot_frame_zones,
    plot_zone_overview,
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
MAP_Z_MIN = 1.0  # zone-map gray range starts at background peak + this many sigmas (black)
MAP_TITLE_WIDTH = 100  # characters per title line before the zone list wraps


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
#   STEP 3 -- EXPORT: zone-map TIFF stack
#
# ===========================================================================

def _figure_to_rgb(fig) -> np.ndarray:
    """Render a Figure at MAP_DPI -> (H, W, 3) uint8 RGB array."""
    fig.set_dpi(MAP_DPI)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def export_zone_maps(analyzer: SpontaneousZoneAnalyzer, title_tag: str, out_path: Path) -> int:
    """Write one RGB TIFF stack: page 1 = max projection + all zones, then one page per detection frame.

    Every page is z-scored against the fitted background and shares one gray range: z = MAP_Z_MIN
    -> median of the detections' max z, so single bright specks can't stretch it. Returns page count.
    """
    stack = analyzer.stack_f16
    center, sigma, thr = analyzer.bg_center, analyzer.bg_sigma, analyzer.threshold
    det = analyzer.detections
    det_frames = det["frame"].to_numpy() if not det.empty else np.array([], dtype=int)
    frames = np.unique(det_frames)  # 1-based

    max_proj = stack.max(axis=0)
    det_raw_max = np.array([stack[f - 1][fp[:, 0], fp[:, 1]].max() for f, fp in zip(det_frames, analyzer.footprints,
                                                                                    strict=True)], dtype=np.float32)
    det_max_z = img_zscore_convert(det_raw_max, center, sigma)  # max z inside each detection
    vmin = MAP_Z_MIN
    vmax = float(np.median(det_max_z)) if det_max_z.size else float((max_proj.max() - center) / sigma)
    thr_text = f"thr = {center:.3f} + {analyzer.sigma_ratio} × {sigma:.3f} = {thr:.3f}"

    zone_ids = sorted(analyzer.zone_masks)
    colors = zone_colors(zone_ids)
    label_to_zone = {label: row.zone_id for row in analyzer.zones.itertuples() for label in row.joint_labels}
    zone_source = dict(zip(analyzer.zones["zone_id"], analyzer.zones["source"], strict=True))
    n_by_source = analyzer.zone_stats["source"].str.split(" #").str[0].value_counts()
    overview_title = (f"{title_tag} | {len(zone_ids)} zones -- {n_by_source.get('trace_corr', 0)} trace-corr, "
                      f"{n_by_source.get('proximity', 0)} proximity, {n_by_source.get('isolated', 0)} isolated\n"
                      f"{thr_text}")
    first = _figure_to_rgb(plot_zone_overview(
        img_zscore_convert(max_proj.astype(np.float32), center, sigma), analyzer.zone_masks,
        analyzer.zone_centroids, colors, vmin, vmax, overview_title, analyzer.um_per_px))

    def frame_pages() -> Iterator[np.ndarray]:
        yield first
        for frame in frames:
            z_frame = img_zscore_convert(stack[frame - 1].astype(np.float32), center, sigma)
            rows = np.flatnonzero(det_frames == frame)
            hotspot_mask = np.zeros(z_frame.shape, dtype=bool)
            for i in rows:
                hotspot_mask[analyzer.footprints[i][:, 0], analyzer.footprints[i][:, 1]] = True
            frame_zone_ids = sorted({label_to_zone[label] for label in det["joint_label"].iloc[rows]})
            zones_text = ", ".join(f"{z} ({zone_source[z]})" for z in frame_zone_ids)
            title = (f"frame {frame} ({frame / analyzer.fps:.2f} s) | {thr_text} | max z = {det_max_z[rows].max():.2f}\n"
                     + textwrap.fill(f"zones {zones_text}", MAP_TITLE_WIDTH))
            yield _figure_to_rgb(plot_frame_zones(z_frame, frame_zone_ids, analyzer.zone_masks,
                                                  analyzer.zone_centroids, colors, hotspot_mask, vmin, vmax, title,
                                                  analyzer.um_per_px))

    n_pages = 1 + frames.size
    tifffile.imwrite(out_path, frame_pages(), shape=(n_pages, *first.shape), dtype=np.uint8, photometric="rgb",
                     compression="zlib")  # pages streamed one at a time: one series (pages, H, W, 3)
    return n_pages


# ===========================================================================
#
#   RUN
#
# ===========================================================================

def run(proc_list_path: Path, results_dir: Path = Path("results"), sigma: float = CROSSOVER_RATIO, save_mask: bool = True,
        all_obj: bool = False, debug: bool = False, cuda_available: bool = False, db_path: Path = Path("data/rec_data.db"), exp_db_path: Path = Path("data/exp_info.db")) -> None:
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
            del stack  # the analyzer keeps its float16 copy
        analyzer.run()

        # -------------------------------------------------------------------
        # Step 3. Export
        # -------------------------------------------------------------------
        with timed("export xlsx / npz" + (" / mask tif" if save_mask else "")):
            paths = analyzer.save(out_root, stem, save_mask=save_mask, debug=debug)
        map_path = out_root / f"{stem}_ZONE_MAPS.tif"
        with timed("export zone-map TIFF"):
            n_pages = export_zone_maps(analyzer, f"{stem}, {row['SENSOR']}", map_path)
        for path in paths.values():
            console.log(f"[green]saved[/green] {path.resolve()}")
        console.log(f"[green]saved[/green] {n_pages}-page zone-map TIFF ({map_path.stat().st_size / 1e6:.1f} MB) "
                    f"-> {map_path.resolve()}")

        zone_stats = analyzer.zone_stats
        freq = zone_stats.loc[zone_stats["n_events"] >= MIN_EVENTS_FOR_FREQ, "mean_freq_hz"]
        period = zone_stats.loc[zone_stats["n_events"] >= MIN_EVENTS_FOR_FREQ, "mean_period_s"]
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
            "n_low_freq_zones": int((zone_stats["n_events"] == 1).sum()),  # < 1 event per recording
            "n_freq_zones": len(freq),  # zones with >= MIN_EVENTS_FOR_FREQ events, used below
            "median_freq_hz": float(freq.median()),
            "freq_q1_hz": float(freq.quantile(0.25)),
            "freq_q3_hz": float(freq.quantile(0.75)),
            "freq_cv": float(freq.std() / freq.mean()),
            "median_period_s": float(period.median()),
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

    console.rule(f"[dim]Total time: {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spontaneous ACh hotspot -> zone analysis")
    parser.add_argument("--proc_list", required=True, type=Path, help="Proc list (proc_*.txt) naming the recordings")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Outputs go to <results_dir>/spontaneous/")
    parser.add_argument("--sigma", type=float, default=CROSSOVER_RATIO,
                        help="Threshold = background peak + this many sigmas")
    parser.add_argument("--no_mask", action="store_true", help="Skip saving the per-frame hotspot mask (mask/)")
    parser.add_argument("--all_obj", action="store_true", help=f"Also analyze non-{TARGET_OBJ} recordings (testing only)")
    parser.add_argument("--debug", action="store_true", help="Also save the raw per-frame detections CSV")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=Path("data/exp_info.db"), help="Path to exp_info.db")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda()  # must run before anything imports numba
    console.log(_cuda_msg)
    run(args.proc_list, args.results_dir, args.sigma, not args.no_mask, args.all_obj,
        args.debug, _cuda_available, args.db, args.exp_db)
