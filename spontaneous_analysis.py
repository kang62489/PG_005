"""
spontaneous_analysis.py  --  Spontaneous ACh hotspot -> zone analysis (10X, *_BIEXP_ALS.tif).

  Step 1. Select  : proc list -> recordings with an ALS tiff; OBJ / SENSOR from rec_data.db, 10X only
  Step 2. Analyze : per recording, SpontaneousZoneAnalyzer detect -> group -> map
          Coverage: union of all compartments (NR zones excluded) inside the striatum / striatum area
                    (striatum outline from the Striatum Boundary export bd_{date}_{serial}.json; NaN if none)
  Step 3. Export  : per recording {stem}_ZONES.xlsx + {stem}_ZONE_MAPS.tif (z-scored RGB stack: max projection
                    + compartments + NR zones (light gray) + striatum outline, then one page per detection frame;
                    frame pages rendered on a process pool, --map_workers)
                    + footprints/{stem}_ZONES.npz + mask/{stem}_HOTSPOT_MASK.tif (off with --no_mask)
  Step 4. Summary : spontaneous_summary.xlsx (one row per recording + pooled compartment table)

Zone = any mapped region; compartment = recurring zone (the only ones in stats); NR zone = non-recurring zone.

All outputs go to {results_dir}/spontaneous/.

Usage:
    python spontaneous_analysis.py --proc_list data/proc_20260924_000.txt
        [--results_dir results] [--sigma 2.0] [--no_mask] [--all_obj] [--debug]
        [--stbd data/bd_20260924_000.json]  (default: bd_{date}_{serial}.json next to the proc list)
        [--map_workers N]  (default: usable CPUs, max MAP_WORKERS_MAX; 1 = no pool)
"""

## Modules
# Standard library imports
import argparse
import itertools
import multiprocessing as mp
import os
import textwrap
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
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
    bd_export_path,
    check_cuda,
    direction_labels,
    frame_zone_figures,
    img_zscore_convert,
    list_parser,
    load_st_bd,
    lookup_rec_from_db,
    outline_mask,
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
MAP_TITLE_WIDTH = 100  # characters per title line before the compartment / NR zone list wraps
NR_COLOR = (0.85, 0.85, 0.85)  # NR zones (NR1, NR2, ...) on the zone maps: light gray
MAP_WORKERS_MAX = 16  # auto worker count for zone-map rendering = usable CPUs, capped here (--map_workers overrides)
MAP_CHUNK_PAGES = 8  # frame pages per pool task (each task builds its own Figure once)


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
#   STEP 2b -- COVERAGE: union of all compartments (NR zones excluded) inside the striatum / striatum area
#
# ===========================================================================

def striatum_of(stbd: dict[str, dict], raw_stem: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Striatum mask of one recording from the bd export; None (+ warning) if missing or a different frame size."""
    entry = stbd.get(raw_stem)
    if entry is None or "striatum_outline_px" not in entry:
        console.log(f"[yellow]No striatum outline for {raw_stem} -- coverage NaN[/yellow]")
        return None
    if tuple(entry["image_shape"]) != shape:
        console.log(f"[yellow]Striatum drawn on {entry['image_shape']}, image is {list(shape)} -- "
                    f"coverage NaN[/yellow]")
        return None
    return outline_mask(entry["striatum_outline_px"], shape)


def zone_coverage(zone_masks: dict[int, np.ndarray], striatum: np.ndarray | None, um_per_px: float) -> dict:
    """Summary columns: striatum area, area of the compartment union inside it (um^2), and their ratio (0-1)."""
    if striatum is None:
        return {"striatum_area_um2": np.nan, "compartment_area_in_striatum_um2": np.nan, "striatum_coverage": np.nan}
    union = np.zeros_like(striatum)
    for mask in zone_masks.values():
        union |= mask
    striatum_px, covered_px = int(striatum.sum()), int((union & striatum).sum())
    return {
        "striatum_area_um2": striatum_px * um_per_px**2,
        "compartment_area_in_striatum_um2": covered_px * um_per_px**2,
        "striatum_coverage": covered_px / striatum_px,
    }


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


def _gray_range(analyzer: SpontaneousZoneAnalyzer) -> tuple[np.ndarray, np.ndarray, float]:
    """(max projection, max z inside each detection, gray top = median of those max z) for the zone maps."""
    stack, center, sigma = analyzer.stack_f16, analyzer.bg_center, analyzer.bg_sigma
    det = analyzer.detections
    det_frames = det["frame"].to_numpy() if not det.empty else np.array([], dtype=int)
    max_proj = stack.max(axis=0)
    det_raw_max = np.array([stack[f - 1][fp[:, 0], fp[:, 1]].max() for f, fp in zip(det_frames, analyzer.footprints,
                                                                                    strict=True)], dtype=np.float32)
    det_max_z = img_zscore_convert(det_raw_max, center, sigma)
    vmax = float(np.median(det_max_z)) if det_max_z.size else float((max_proj.max() - center) / sigma)
    return max_proj, det_max_z, vmax


def _thr_text(analyzer: SpontaneousZoneAnalyzer) -> str:
    """'thr = c + k × σ = thr' for map titles."""
    return (f"thr = {analyzer.bg_center:.3f} + {analyzer.sigma_ratio} × {analyzer.bg_sigma:.3f} = "
            f"{analyzer.threshold:.3f}")


# --- 3a. page rendering (serial / process pool) ----------------------------


def map_workers_auto() -> int:
    """CPUs this process may use (the SLURM allocation on Linux, not the whole node), capped at MAP_WORKERS_MAX."""
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(n_cpu, MAP_WORKERS_MAX))


def _render_pages(specs, ctx: dict) -> Iterator[np.ndarray]:
    """Frame specs -> RGB pages, rendered on one reused Figure (frame_zone_figures).

    spec = (raw float16 frame, hotspot footprints, NR flag per footprint, zone ids hit, title).
    """
    center, sigma, shape = ctx["center"], ctx["sigma"], ctx["shape"]

    def inputs() -> Iterator[tuple]:
        for frame_raw, footprints, nr_flags, frame_zone_ids, title in specs:
            z_frame = img_zscore_convert(frame_raw.astype(np.float32), center, sigma)
            hotspot_mask = np.zeros(shape, dtype=bool)
            nr_hotspot_mask = np.zeros(shape, dtype=bool)  # hotspots of NR-zone units -> '////'
            for fp, is_nr in zip(footprints, nr_flags, strict=True):
                hotspot_mask[fp[:, 0], fp[:, 1]] = True
                if is_nr:
                    nr_hotspot_mask[fp[:, 0], fp[:, 1]] = True
            yield z_frame, frame_zone_ids, hotspot_mask, nr_hotspot_mask, title

    for fig in frame_zone_figures(inputs(), shape, ctx["masks"], ctx["centroids"], ctx["colors"],
                                  ctx["vmin"], ctx["vmax"], ctx["um_per_px"]):
        yield _figure_to_rgb(fig)  # one reused Figure: render before pulling the next page


_worker_state: dict = {}  # per pool worker: {"key", "feed", "pages"} -- one Figure kept across chunks of a recording


def _render_chunk(specs: list[tuple], ctx: dict) -> list[np.ndarray]:
    """Process-pool task: one chunk of frame specs -> its RGB pages.

    The worker's page generator (and its Figure, ~0.3 s to build) lives on between chunks of the same recording
    (ctx["key"]); each next() pulls exactly one spec from the feed, so the feed never runs dry.
    """
    if _worker_state.get("key") != ctx["key"]:
        feed: deque = deque()
        _worker_state.update(key=ctx["key"], feed=feed, pages=_render_pages(iter(feed.popleft, None), ctx))
    _worker_state["feed"].extend(specs)
    return [next(_worker_state["pages"]) for _ in specs]


def _render_pages_parallel(specs, ctx: dict, pool: ProcessPoolExecutor, n_workers: int) -> Iterator[np.ndarray]:
    """Like _render_pages, but MAP_CHUNK_PAGES-page chunks on the pool; pages come back in frame order.

    At most 2 chunks per worker are in flight, so finished-but-unwritten pages stay bounded in memory.
    """
    pending: deque = deque()
    for chunk in itertools.batched(specs, MAP_CHUNK_PAGES):
        pending.append(pool.submit(_render_chunk, list(chunk), ctx))
        if len(pending) >= 2 * n_workers:
            yield from pending.popleft().result()
    while pending:
        yield from pending.popleft().result()


# --- 3b. TIFF stack ---------------------------------------------------------


def export_zone_maps(analyzer: SpontaneousZoneAnalyzer, title_tag: str, out_path: Path,
                     striatum_outline: np.ndarray | None = None, axis_labels: tuple[str, str] | None = None,
                     pool: ProcessPoolExecutor | None = None, n_workers: int = 1) -> int:
    """Write one RGB TIFF stack: page 1 = max projection + all zones, then one page per detection frame.

    All zones = compartments (1, 2, ...) + NR zones (NR1, NR2, ...; light gray, for reference, not in stats).
    Every page is z-scored against the fitted background and shares one gray range: z = MAP_Z_MIN
    -> median of the detections' max z, so single bright specks can't stretch it. Returns page count.
    pool: frame pages rendered on its n_workers processes (same pixels); None -> in this process.
    """
    stack = analyzer.stack_f16
    center, sigma = analyzer.bg_center, analyzer.bg_sigma
    det = analyzer.detections
    det_frames = det["frame"].to_numpy() if not det.empty else np.array([], dtype=int)
    frames = np.unique(det_frames)  # 1-based

    max_proj, det_max_z, vmax = _gray_range(analyzer)
    vmin = MAP_Z_MIN
    thr_text = _thr_text(analyzer)

    # compartments first (ids 1..N, tab20 colors), then NR1.. in light gray (reference only)
    map_masks = {**analyzer.zone_masks, **analyzer.non_recur_masks}
    map_centroids = {**analyzer.zone_centroids, **analyzer.non_recur_centroids}
    colors = {**zone_colors(list(analyzer.zone_masks)), **dict.fromkeys(analyzer.non_recur_masks, NR_COLOR)}
    map_zones = pd.concat([analyzer.zones[["zone_id", "source", "joint_labels"]], analyzer.non_recur_zones])
    label_to_zone = {label: row.zone_id for row in map_zones.itertuples() for label in row.joint_labels}
    zone_source = dict(zip(map_zones["zone_id"], map_zones["source"], strict=True))
    nr_labels = {label for labels in analyzer.non_recur_zones["joint_labels"] for label in labels}
    overview_title = (f"{title_tag}\n{len(analyzer.zone_masks)} compartments + "
                      f"{len(analyzer.non_recur_masks)} NR zones (non-recurring, not in stats)\n{thr_text}")
    first = _figure_to_rgb(plot_zone_overview(
        img_zscore_convert(max_proj.astype(np.float32), center, sigma), map_masks, map_centroids, colors, vmin,
        vmax, overview_title, analyzer.um_per_px, striatum_outline, axis_labels))

    def frame_specs() -> Iterator[tuple]:
        for frame in frames:
            rows = np.flatnonzero(det_frames == frame)
            footprints = [analyzer.footprints[i] for i in rows]
            nr_flags = [det["joint_label"].iloc[i] in nr_labels for i in rows]
            hit = {label_to_zone[label] for label in det["joint_label"].iloc[rows]
                   if label in label_to_zone}  # hotspots of dropped units have no zone
            frame_zone_ids = [z for z in map_masks if z in hit]  # map order: compartments, then NR zones
            compartments = ", ".join(f"{z} ({zone_source[z]})" for z in frame_zone_ids if z in analyzer.zone_masks)
            nr_zones = ", ".join(f"{z} ({zone_source[z]})" for z in frame_zone_ids if z in analyzer.non_recur_masks)
            zones_text = f"compartments {compartments or '-'}" + (f" | NR zones {nr_zones}" if nr_zones else "")
            title = (f"frame {frame} ({frame / analyzer.fps:.2f} s) | {thr_text} | "
                     f"max z = {det_max_z[rows].max():.2f}\n"
                     + textwrap.fill(zones_text, MAP_TITLE_WIDTH))
            yield stack[frame - 1], footprints, nr_flags, frame_zone_ids, title

    ctx = {"key": (str(out_path), time.time_ns()),  # one export call -> pool workers rebuild their Figure once
           "center": center, "sigma": sigma, "shape": max_proj.shape, "masks": map_masks, "centroids": map_centroids,
           "colors": colors, "vmin": vmin, "vmax": vmax, "um_per_px": analyzer.um_per_px}

    def frame_pages() -> Iterator[np.ndarray]:
        yield first
        if pool is None:
            yield from _render_pages(frame_specs(), ctx)
        else:
            yield from _render_pages_parallel(frame_specs(), ctx, pool, n_workers)

    n_pages = 1 + frames.size
    tifffile.imwrite(out_path, frame_pages(), shape=(n_pages, *first.shape), dtype=np.uint8, photometric="rgb",
                     compression="zlib")  # pages streamed one at a time: one series (pages, H, W, 3)
    return n_pages


# ===========================================================================
#
#   RUN
#
# ===========================================================================

def run(proc_list_path: Path, results_dir: Path = Path("results"), sigma: float = CROSSOVER_RATIO,
        save_mask: bool = True, all_obj: bool = False, debug: bool = False, cuda_available: bool = False,
        db_path: Path = Path("data/rec_data.db"), exp_db_path: Path = Path("data/exp_info.db"),
        stbd_path: Path | None = None, map_workers: int | None = None) -> None:
    """Run the spontaneous zone analysis for every selected recording in a proc list.

    stbd_path: Striatum Boundary export; None -> bd_{date}_{serial}.json next to the proc list.
    map_workers: processes rendering the zone-map pages; None -> map_workers_auto(), 1 -> in this process.
    """
    run_t0 = time.time()
    n_workers = map_workers or map_workers_auto()
    console.log(f"zone-map rendering: {n_workers} worker(s)")
    # one pool for the whole run ('spawn': same on Linux / Windows, no fork of the numba / CUDA state)
    pool_ctx = (ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) if n_workers > 1
                else nullcontext())
    with pool_ctx as pool:
        _run_recordings(proc_list_path, results_dir, sigma, save_mask, all_obj, debug, cuda_available, db_path,
                        exp_db_path, stbd_path, pool, n_workers)
    console.rule(f"[dim]Total time: {time.time() - run_t0:.1f}s")


def _run_recordings(proc_list_path: Path, results_dir: Path, sigma: float, save_mask: bool, all_obj: bool, debug: bool,
                    cuda_available: bool, db_path: Path, exp_db_path: Path, stbd_path: Path | None,
                    pool: ProcessPoolExecutor | None, n_workers: int) -> None:
    """Steps 1-4 of run() (pool: zone-map renderer, None -> in this process)."""
    out_root = results_dir / "spontaneous"
    out_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1. Select
    # -----------------------------------------------------------------------
    console.rule("[bold]Step 1 - Select")
    recordings = select_recordings(proc_list_path, db_path, exp_db_path, all_obj)
    console.log(f"{len(recordings)} recording(s) selected from {proc_list_path.name}")
    stbd_path = stbd_path or bd_export_path(proc_list_path)
    if stbd_path.exists():
        stbd = load_st_bd(stbd_path)["recordings"]
        console.log(f"striatum boundaries: {len(stbd)} recording(s) in {stbd_path.resolve()}")
    else:
        stbd = {}
        console.log(f"[yellow]No striatum boundary file {stbd_path.resolve()} -- coverage NaN for all[/yellow]")

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
        raw_stem = Path(row["raw_tiff_name"]).stem
        striatum = striatum_of(stbd, raw_stem, (analyzer.height, analyzer.width))
        striatum_outline = np.array(stbd[raw_stem]["striatum_outline_px"]) if striatum is not None else None
        axis_labels = direction_labels(stbd[raw_stem]["dorsal"], stbd[raw_stem]["medial"]) if raw_stem in stbd else None
        coverage = zone_coverage(analyzer.zone_masks, striatum, analyzer.um_per_px)
        if striatum is not None:
            console.log(f"  coverage {coverage['striatum_coverage']:.3f} ({len(analyzer.zone_masks)} compartments, "
                        f"striatum {striatum.mean():.2f} of FOV)")

        # -------------------------------------------------------------------
        # Step 3. Export
        # -------------------------------------------------------------------
        with timed("export xlsx / npz" + (" / mask tif" if save_mask else "")):
            paths = analyzer.save(out_root, stem, save_mask=save_mask, debug=debug)
        map_path = out_root / f"{stem}_ZONE_MAPS.tif"
        with timed("export zone-map TIFF"):
            n_pages = export_zone_maps(analyzer, f"{stem}, {row['SENSOR']}", map_path, striatum_outline, axis_labels,
                                       pool, n_workers)
        for path in paths.values():
            console.log(f"[green]saved[/green] {path.resolve()}")
        console.log(f"[green]saved[/green] {n_pages}-page zone-map TIFF ({map_path.stat().st_size / 1e6:.1f} MB) "
                    f"-> {map_path.resolve()}")

        zone_stats = analyzer.zone_stats  # compartments only; NR zones stay out of every stat
        non_recur_source = analyzer.non_recur_zones["source"]
        freq = zone_stats.loc[zone_stats["n_events"] >= MIN_EVENTS_FOR_FREQ, "mean_freq_hz"]
        period = zone_stats.loc[zone_stats["n_events"] >= MIN_EVENTS_FOR_FREQ, "mean_period_s"]
        summary_rows.append({
            "recording": stem,
            "sensor": row["SENSOR"],
            "obj": row["OBJ"],
            "n_frames": analyzer.n_frames,
            "background_threshold": analyzer.threshold,
            "n_compartments": len(zone_stats),
            "n_non_recur_zone_type_1": int(non_recur_source.str.startswith("non_recur_zone_type_1").sum()),
            "n_non_recur_zone_type_2": int(non_recur_source.str.startswith("non_recur_zone_type_2").sum()),
            "n_dropped_units": len(analyzer.dropped_units),
            "n_dropped_hotspots": int(analyzer.detections["joint_label"].isin(analyzer.dropped_units).sum()),
            "median_area_um2": float(zone_stats["area_um2"].median()),
            "n_low_freq_compartments": int((zone_stats["n_events"] == 1).sum()),  # < 1 event per recording
            "n_freq_compartments": len(freq),  # compartments with >= MIN_EVENTS_FOR_FREQ events, used below
            "median_freq_hz": float(freq.median()),
            "freq_q1_hz": float(freq.quantile(0.25)),
            "freq_q3_hz": float(freq.quantile(0.75)),
            "freq_cv": float(freq.std() / freq.mean()),
            "median_period_s": float(period.median()),
            **coverage,
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
            pooled.to_excel(writer, sheet_name="compartments", index=False)
        console.log(f"[green]saved[/green] {summary_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spontaneous ACh hotspot -> zone analysis")
    parser.add_argument("--proc_list", required=True, type=Path, help="Proc list (proc_*.txt) naming the recordings")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Outputs go to <results_dir>/spontaneous/")
    parser.add_argument("--sigma", type=float, default=CROSSOVER_RATIO,
                        help="Threshold = background peak + this many sigmas")
    parser.add_argument("--no_mask", action="store_true",
                        help="Skip saving the per-frame hotspot mask (mask/{stem}_HOTSPOT_MASK.tif)")
    parser.add_argument("--all_obj", action="store_true", help=f"Also analyze non-{TARGET_OBJ} recordings (testing only)")
    parser.add_argument("--debug", action="store_true", help="Also save the raw per-frame detections CSV")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=Path("data/exp_info.db"), help="Path to exp_info.db")
    parser.add_argument("--stbd", type=Path, default=None,
                        help="Striatum Boundary export (default: bd_{date}_{serial}.json next to the proc list)")
    parser.add_argument("--map_workers", type=int, default=None,
                        help=f"Processes rendering the zone-map pages (default: usable CPUs, max {MAP_WORKERS_MAX}; "
                             "1 = no pool)")
    args = parser.parse_args()

    _cuda_available, _cuda_msg = check_cuda()  # must run before anything imports numba
    console.log(_cuda_msg)
    run(args.proc_list, args.results_dir, args.sigma, not args.no_mask, args.all_obj,
        args.debug, _cuda_available, args.db, args.exp_db, args.stbd, args.map_workers)
