"""
SRE_analysis.py  --  Spontaneous ACh flash -> zone analysis (10X, *_BIEXP_ALS.tif).

  Step 1. Input            : proc list -> 10X recordings in rec_data.db with their *_BIEXP_ALS.tif (RecordingList;
                             GAUSS with --use_gauss, all OBJ with --all_obj)
  Step 2. Detection        : th_small_obj (removal_suggestion.csv, else REMOVAL_FALLBACK) + connect_radius
                             (JOIN_SAME_FRAME False: 0, every blob its own flash)
                             -> SREAnalyzer.detect() -> footprints/{stem}_FLASH_FOOTPRINTS.npz
  Step 3. Clustering       : trace_corr tree (1 - r, average linkage) cut at its best-silhouette h -> map(method)
                             (z_peak tree kept, not run: CLUSTER_METHODS)
  Step 4. Characterization : striatum outline + axis labels from bd_{date}_{serial}.json (coverage: commented out)
  Step 5. Export           : per method {stem}_{METHOD}_ZONES.xlsx, footprints/{stem}_{METHOD}_ZONES.npz,
                             clustering/{stem}_{METHOD}_DENDROGRAM.png, {stem}_{METHOD}_ZONE_MAPS.tif
                             (pages rendered on a process pool, --map_workers);
                             clustering/{stem}_SILHOUETTE.png + _CLUSTERS.png; mask/{stem}_FLASH_MASK.tif
                             (--no_mask)

recur_zone = recurring zone (non-recurring zones are not defined for the clustering yet).

All outputs go to {results_dir}/spontaneous/.

Usage:
    python SRE_analysis.py --proc_list data/proc_20260924_000.txt
        [--results_dir results] [--use_gauss] [--sigma 2.0] [--no_mask] [--all_obj] [--debug]
        [--stbd data/bd_20260924_000.json]  (default: bd_{date}_{serial}.json next to the proc list)
        [--map_workers N]  (default: usable CPUs, max MAP_WORKERS_MAX; 1 = no pool)
"""

## Modules
# Standard library imports
import argparse
import gc
import itertools
import multiprocessing as mp
import os
import textwrap
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Third-party imports
import numpy as np
import polars as pl
import tifffile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from rich.console import Console

# Local imports
from classes import SREAnalyzer  # detection
from classes.recording_list import ALS_SUFFIX, RecordingList  # input
from classes.sre_analyzer import (
    TREE_UNITS,  # export
    cut_h_name,  # export
    timed,  # detection, export
)

# from classes.sre_analyzer import MIN_EVENTS_FOR_FREQ  # run summary (commented out for now)
from functions import (
    bd_export_path,
    check_cuda,
    direction_labels,
    frame_zone_figures,
    img_zscore_convert,
    load_st_bd,
    outline_mask,
    plot_cluster_dendrogram,  # export
    plot_cluster_scatter,  # export
    plot_silhouette_sweep,  # export
    plot_zone_overview,
    zone_colors,
)
from functions.fit_hist import CROSSOVER_RATIO  # detection

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Input -----------------------------------------------------------------
TARGET_OBJ = "10X"  # recordings analyzed (--all_obj: all)
GAUSS_SUFFIX = "_BIEXP_GAUSS.tif"  # input with --use_gauss (default: ALS_SUFFIX)

# --- Detection -------------------------------------------------------------
JOIN_SAME_FRAME = False  # True: same-frame blobs <= connect_radius apart joined; False: every blob its own flash
                         # (connect_radius 0, the trees group same-frame blobs)

# --- Clustering ------------------------------------------------------------
CLUSTER_METHODS = ("trace_corr",)  # clusterings; each gets its own zones + {stem}_{METHOD}_* exports
                                   # ("z_peak" kept in SREAnalyzer.group_z_peak, not run; add it here to run it)

# --- Export ----------------------------------------------------------------
MAP_DPI = 120  # dpi of the zone-map pages and clustering/ PNGs
MAP_Z_MIN = 1.0  # zone-map gray range starts at background peak + this many sigmas (black)
MAP_TITLE_WIDTH = 80  # characters per title line before the recur_zone list wraps (titles at ZONE_MAP_TITLE_FS)
MAP_WORKERS_MAX = 16  # auto worker count for zone-map rendering = usable CPUs, capped here (--map_workers overrides)
MAP_CHUNK_PAGES = 8  # frame pages per pool task (each task builds its own Figure once)


# ===========================================================================
#
#   FLOW: The flow of the pipeline
#
# ===========================================================================

# fn: run, input once, then per recording detection -> clustering -> characterization -> export, then the run summary.

def run(proc_list_path: Path, results_dir: Path = Path("results"), sigma: float = CROSSOVER_RATIO,
        save_mask: bool = True, all_obj: bool = False, debug: bool = False, cuda_available: bool = False,
        db_path: Path = Path("data/rec_data.db"), exp_db_path: Path = Path("data/exp_info.db"),
        stbd_path: Path | None = None, map_workers: int | None = None, use_gauss: bool = False) -> None:
    """Run input -> detection -> clustering -> characterization -> export for every recording in a proc list.

    stbd_path: Striatum Boundary export; None -> bd_{date}_{serial}.json next to the proc list.
    map_workers: processes rendering the zone-map pages; None -> map_workers_auto(), 1 -> in this process.
    use_gauss: read *_BIEXP_GAUSS.tif instead of *_BIEXP_ALS.tif.
    """
    run_t0 = time.time()
    out_root = results_dir / "spontaneous"
    out_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------
    console.rule("[bold]Input")

    # imported class: RecordingList -- proc list -> recordings in rec_data.db with their tiff, 10X only (--all_obj: all)
    recording_list = RecordingList(proc_list_path, "proc", db_path, exp_db_path)
    recordings = recording_list.select(objs=None if all_obj else [TARGET_OBJ],
                                       suffix=GAUSS_SUFFIX if use_gauss else ALS_SUFFIX)
    console.log(f"{len(recordings)} recording(s) selected from {proc_list_path.name}")

    # -----------------------------------------------------------------------
    # Detection (set once, before the loop)
    # -----------------------------------------------------------------------
    # class method: set_removal_suggestion() -- th_small_obj of every recording
    recordings = recording_list.set_removal_suggestion(recordings, results_dir)
    # class method: set_connect_radius() -- connect_radius of every recording, from its th_small_obj
    # (JOIN_SAME_FRAME False: 0, every blob its own flash)
    recordings = recording_list.set_connect_radius(recordings)
    if not JOIN_SAME_FRAME:
        recordings = recordings.with_columns(pl.lit(0).alias("connect_radius"))

    # -----------------------------------------------------------------------
    # Characterization (set once, before the loop)
    # -----------------------------------------------------------------------
    # striatum outlines of the run (coverage; also drawn on the zone maps)
    stbd_path = stbd_path or bd_export_path(proc_list_path)
    if stbd_path.exists():
        stbd = load_st_bd(stbd_path)["recordings"]
        console.log(f"striatum boundaries: {len(stbd)} recording(s) in {stbd_path.resolve()}")
    else:
        stbd = {}
        console.log(f"[yellow]No striatum boundary file {stbd_path.resolve()} -- coverage NaN for all[/yellow]")

    # -----------------------------------------------------------------------
    # Export (set once, before the loop)
    # -----------------------------------------------------------------------
    # run summary (spontaneous_summary.xlsx): commented out for now, not in the planned exports
    # summary_rows: list[dict] = []  # one row per recording -> spontaneous_summary.xlsx
    # lst_recur_zones: list[pd.DataFrame] = []  # recur_zone tables of all recordings -> spontaneous_summary.xlsx

    # local fn: open_map_pool() -- one pool for the whole run
    pool, n_workers = open_map_pool(map_workers)

    for i, row in enumerate(recordings.iter_rows(named=True), 1):
        entry_t0 = time.time()
        proc_tiff_path = Path(row["proc_tiff_path"])
        proc_stem = proc_tiff_path.stem  # 2025_12_15-0009_BIEXP_ALS: export filenames + "recording" column
        raw_stem = Path(row["raw_tiff_name"]).stem  # 2025_12_15-0009: key of the striatum boundaries (stbd)
        console.rule(f"[bold]{proc_stem}  [{row['OBJ']}, {row['SENSOR']}]  [{i}/{len(recordings)}]")

        # -------------------------------------------------------------------
        # Detection
        # -------------------------------------------------------------------
        th_small_obj, th_flag = row["th_small_obj"], row["th_small_obj_flag"]
        console.log(f"  th_small_obj {th_small_obj:,} px ({th_flag or 'permutation'}), "
                    f"connect_radius {row['connect_radius']} px")

        with timed("load   read tiff + float16 copy"):
            stack = tifffile.imread(proc_tiff_path)
            # imported class: SREAnalyzer -- keeps a float16 copy of the stack
            analyzer = SREAnalyzer(stack, obj=row["OBJ"], sigma_ratio=sigma, cuda_available=cuda_available,
                                   th_small_obj=th_small_obj, connect_radius=row["connect_radius"])
            del stack  # the analyzer keeps its float16 copy

        # class method: detect() -> analyzer.threshold, mask (flash mask), flashes + footprints
        # (same-frame blobs <= connect_radius apart joined; connect_radius 0: none joined)
        analyzer.detect()

        # class method: save_flash_footprints() -- footprints/{stem}_FLASH_FOOTPRINTS.npz (all flashes; group_trace input)
        footprint_path = analyzer.save_flash_footprints(out_root, proc_stem)
        console.log(f"[green]saved[/green] {footprint_path.resolve()}")

        # -------------------------------------------------------------------
        # Clustering
        # -------------------------------------------------------------------
        # class method: group_trace() -- footprint traces from the npz, 1 - r (Pearson, zero lag) tree cut at the best
        # silhouette
        analyzer.group_trace(footprint_path)
        # class method: group_z_peak() -- z peak (largest-z footprint pixel) Euclidean distance tree, same cut
        # (+ acceleration elbow for the plots); kept, not run: add "z_peak" to CLUSTER_METHODS to run it
        if "z_peak" in CLUSTER_METHODS:
            analyzer.group_z_peak()

        # -------------------------------------------------------------------
        # Export (every clustering of CLUSTER_METHODS)
        # -------------------------------------------------------------------
        # class method: save_flashes() -- mask/{stem}_FLASH_MASK.tif, debug: {stem}_FLASHES.csv
        with timed("export mask tif / flashes csv"):
            paths = list(analyzer.save_flashes(out_root, proc_stem, save_mask=save_mask, debug=debug).values())

        # -------------------------------------------------------------------
        # Characterization (not sorted yet)
        # -------------------------------------------------------------------
        # striatum outline + axis labels: drawn on the zone maps
        striatum = striatum_of(stbd, raw_stem, (analyzer.height, analyzer.width))
        striatum_outline = np.array(stbd[raw_stem]["striatum_outline_px"]) if striatum is not None else None
        axis_labels = direction_labels(stbd[raw_stem]["dorsal"], stbd[raw_stem]["medial"]) if raw_stem in stbd else None

        for method in CLUSTER_METHODS:
            tag = method.upper()

            # Clustering. class method: map() -- clusters of this method = recur_zones (masks, centroids, stats)
            analyzer.map(method)

            # Characterization. coverage of the striatum: commented out for now, not in the planned exports
            # coverage = zone_coverage(analyzer.zone_masks, striatum, analyzer.um_per_px)
            # if striatum is not None:
            #     console.log(f"  coverage {coverage['striatum_coverage']:.3f} ({len(analyzer.zone_masks)} "
            #                 f"recur_zones, striatum {striatum.mean():.2f} of FOV)")

            # Export. class method: save_zones() -- {stem}_{METHOD}_ZONES.xlsx (counts, recur_zones,
            # recur_zone_stats), footprints/{stem}_{METHOD}_ZONES.npz (zone footprints + contours)
            with timed(f"export {method} xlsx / npz"):
                paths += list(analyzer.save_zones(out_root, proc_stem).values())

            # Export. local fn: export_dendrogram() -- clustering/{stem}_{METHOD}_DENDROGRAM.png
            with timed(f"export {method} dendrogram"):
                paths += export_dendrogram(analyzer, proc_stem, out_root / "clustering")

            # Export. local fn: export_zone_maps() -- {stem}_{METHOD}_ZONE_MAPS.tif: all zones, then zone 1 frames,
            # zone 2 frames, ...
            map_path = out_root / f"{proc_stem}_{tag}_ZONE_MAPS.tif"
            with timed(f"export {method} zone-map TIFF"):
                n_pages = export_zone_maps(analyzer, f"{proc_stem}, {row['SENSOR']}, {method} tree", map_path,
                                           striatum_outline, axis_labels, pool, n_workers)
            console.log(f"[green]saved[/green] {n_pages}-page {method} zone-map TIFF "
                        f"({map_path.stat().st_size / 1e6:.1f} MB) -> {map_path.resolve()}")

            # Export. run summary row: commented out for now, not in the planned exports
            # zone_stats = analyzer.zone_stats
            # freq = zone_stats.loc[zone_stats["n_events"] >= MIN_EVENTS_FOR_FREQ, "mean_freq_hz"]
            # period = zone_stats.loc[zone_stats["n_events"] >= MIN_EVENTS_FOR_FREQ, "mean_period_s"]
            # summary_rows.append({
            #     "recording": proc_stem,
            #     "sensor": row["SENSOR"],
            #     "obj": row["OBJ"],
            #     "n_frames": analyzer.n_frames,
            #     "background_threshold": analyzer.threshold,
            #     "th_small_obj_flag": th_flag,  # empty = own th_small_obj from removal_suggestion.csv
            #     "n_recur_zones": len(zone_stats),
            #     "median_area_um2": float(zone_stats["area_um2"].median()),
            #     "n_low_freq_recur_zones": int((zone_stats["n_events"] == 1).sum()),  # < 1 event per recording
            #     "n_freq_recur_zones": len(freq),  # recur_zones with >= MIN_EVENTS_FOR_FREQ events, used below
            #     "median_freq_hz": float(freq.median()),
            #     "freq_q1_hz": float(freq.quantile(0.25)),
            #     "freq_q3_hz": float(freq.quantile(0.75)),
            #     "freq_cv": float(freq.std() / freq.mean()),
            #     "median_period_s": float(period.median()),
            #     **coverage,
            # })
            # lst_recur_zones.append(zone_stats.assign(recording=proc_stem, sensor=row["SENSOR"]))

        # Export. local fn: export_cluster_comparison() -- clustering/{stem}_SILHOUETTE.png / _CLUSTERS.png:
        # silhouette sweep + clusters at the z peaks | on the correlation map
        with timed("export silhouette / cluster comparison"):
            paths += export_cluster_comparison(analyzer, proc_stem, out_root / "clustering")

        for path in paths:
            console.log(f"[green]saved[/green] {path.resolve()}")

        del analyzer  # free this recording's stack + mask (~4.7 GB) before the next one is read
        gc.collect()
        console.log(f"[bold magenta]{'entry total':<40} {time.time() - entry_t0:6.1f}s[/bold magenta]")

    if pool is not None:
        pool.shutdown()  # crash / Ctrl+C: Python's exit hook shuts the pool down

    # -----------------------------------------------------------------------
    # Export (run summary): commented out for now, not in the planned exports
    # -----------------------------------------------------------------------
    # console.rule("[bold]Summary")
    #
    # if summary_rows:
    #     summary_path = out_root / "spontaneous_summary.xlsx"
    #     pooled = pd.concat(lst_recur_zones, ignore_index=True)
    #     pooled = pooled[["recording", "sensor", *[c for c in pooled.columns if c not in ("recording", "sensor")]]]
    #     with pd.ExcelWriter(summary_path) as writer:
    #         pd.DataFrame(summary_rows).to_excel(writer, sheet_name="recordings", index=False)
    #         pooled.to_excel(writer, sheet_name="recur_zones", index=False)
    #     console.log(f"[green]saved[/green] {summary_path.resolve()}")

    flagged = recordings.filter(pl.col("th_small_obj_flag") != "") if not recordings.is_empty() else recordings
    if not flagged.is_empty():
        console.log(f"[yellow]{len(flagged)} recording(s) without their own th_small_obj "
                    f"(column th_small_obj_flag):[/yellow]")
        for proc_tiff_path, th_flag in flagged.select("proc_tiff_path", "th_small_obj_flag").iter_rows():
            console.log(f"[yellow]  {Path(proc_tiff_path).stem}  {th_flag}[/yellow]")

    console.rule(f"[dim]Total time: {time.time() - run_t0:.1f}s")


# ===========================================================================
#
#   INPUT related local functions
#
# ===========================================================================

# (none yet: the proc list is read by classes/recording_list.RecordingList)


# ===========================================================================
#
#   DETECTION related local functions
#
# ===========================================================================

# (none yet: th_small_obj + connect_radius are set by classes/recording_list.RecordingList)


# ===========================================================================
#
#   CLUSTERING related local functions
#
# ===========================================================================

# (none yet)


# ===========================================================================
#
#   CHARACTERIZATION related local functions
#
# ===========================================================================

# (none yet)


# ===========================================================================
#
#   EXPORT related local functions
#
# ===========================================================================

# fn: map_workers_auto, worker count for the zone-map pool (usable CPUs, capped at MAP_WORKERS_MAX).
# fn: open_map_pool, process pool that renders the zone-map pages for the whole run (None = in this process).
# fn: _has_sweep, True if a method's tree has a silhouette sweep (>= 3 flashes).
# fn: export_dendrogram, write clustering/{stem}_{METHOD}_DENDROGRAM.png.
# fn: export_cluster_comparison, write clustering/{stem}_SILHOUETTE.png + _CLUSTERS.png (CLUSTER_METHODS trees side by side).

def map_workers_auto() -> int:
    """CPUs this process may use (the SLURM allocation on Linux, not the whole node), capped at MAP_WORKERS_MAX."""
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(n_cpu, MAP_WORKERS_MAX))


def open_map_pool(map_workers: int | None) -> tuple[ProcessPoolExecutor | None, int]:
    """(pool, n_workers) for export_zone_maps; 'spawn': same on Linux / Windows, no fork of the numba / CUDA state."""
    n_workers = map_workers or map_workers_auto()
    console.log(f"zone-map rendering: {n_workers} worker(s)")
    pool = ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn")) if n_workers > 1 else None
    return pool, n_workers


def _has_sweep(analyzer: SREAnalyzer, method: str) -> bool:
    """True if the tree of this method exists and has a silhouette sweep (>= 3 flashes)."""
    clustering = analyzer.clusterings[method]
    return clustering["tree"] is not None and not clustering["sweep"].empty


def export_dendrogram(analyzer: SREAnalyzer, stem: str, out_dir: Path) -> list[Path]:
    """Dendrogram of the last map(method) -> clustering/{stem}_{METHOD}_DENDROGRAM.png; none if < 3 flashes."""
    method = analyzer.zone_method
    if not _has_sweep(analyzer, method):
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    clustering = analyzer.clusterings[method]
    cut = clustering["cut"]
    distance_label = {"z_peak": "z-peak Euclidean distance",
                      "trace_corr": "footprint-trace 1 - r (Pearson)"}[method]
    fig = plot_cluster_dendrogram(clustering["tree"], cut["h"], cut["n_clusters"],
                                  f"{stem}: {len(analyzer.flashes)} flashes, {distance_label}, average linkage",
                                  distance_label, TREE_UNITS[method], cut_h_name(clustering["cut_rule"]))
    path = out_dir / f"{stem}_{method.upper()}_DENDROGRAM.png"
    fig.savefig(path, dpi=MAP_DPI)
    return [path]


def export_cluster_comparison(analyzer: SREAnalyzer, stem: str, out_dir: Path) -> list[Path]:
    """CLUSTER_METHODS trees (>= 3 flashes) side by side -> clustering/{stem}_SILHOUETTE.png (sweeps, cuts marked) +
    {stem}_CLUSTERS.png (zone-map colours, at the z peaks + trace_corr correlation map; other cut without zone maps:
    best silhouette or the z_peak acceleration elbow)."""
    methods = [method for method in CLUSTER_METHODS if _has_sweep(analyzer, method)]
    if not methods:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    centroids_yx = analyzer.flashes[["z_peak_y", "z_peak_x"]].to_numpy()
    sweeps, scatters = [], []
    for method in methods:
        clustering = analyzer.clusterings[method]
        cut, unit, cut_rule = clustering["cut"], TREE_UNITS[method], clustering["cut_rule"]
        # other cut of the same tree, plotted but without zones: best silhouette (tree cut at another rule) or the
        # acceleration elbow (z_peak tree cut at the best silhouette)
        other = None
        if cut_rule != "silhouette":
            other = ("silhouette", clustering["silhouette_cut"], clustering["silhouette_cluster_ids"])
        elif "acceleration_cut" in clustering:
            other = ("acceleration", clustering["acceleration_cut"], clustering["acceleration_cluster_ids"])
        cuts = [{"h_name": cut_h_name(cut_rule), **cut}]
        if other:
            cuts.append({"h_name": cut_h_name(other[0]), **other[1]})
        sweeps.append({"sweep": clustering["sweep"], "cuts": cuts, "unit": unit,
                       "title": f"{method} tree: mean silhouette vs h"})
        cut_label = "" if cut_rule == "silhouette" else f", {cut_rule} cut"
        cut_panels = [(analyzer.flashes[f"cluster_{method}"].to_numpy() + 1, cut,  # zone_id = cluster + 1
                       f"{method} tree{cut_label}: {cut['n_clusters']} clusters\n(cut at h {cut['h']:g} {unit}, "
                       f"silhouette {cut['silhouette']:.3f})")]
        if other:
            other_rule, other_cut, other_ids = other
            other_name = "best silhouette" if other_rule == "silhouette" else f"{other_rule} cut"
            cut_panels.append((other_ids + 1, other_cut,
                               f"{method} tree, {other_name}: {other_cut['n_clusters']} clusters (no zone maps)\n"
                               f"(cut at h {other_cut['h']:g} {unit}, silhouette {other_cut['silhouette']:.3f})"))
        for flash_zone_ids, panel_cut, title in cut_panels:
            panel = {"centroids_yx": centroids_yx, "flash_zone_ids": flash_zone_ids,
                     "colors": zone_colors(list(range(1, panel_cut["n_clusters"] + 1))),  # = zone-map colours
                     "title": title}
            scatters.append(panel)  # flashes at their z peaks
            if "distance" in clustering:  # trace_corr: same clusters on the correlation map (dot distance ~ 1 - r)
                scatters.append({**panel, "distance": clustering["distance"]})
    figures = {"SILHOUETTE": plot_silhouette_sweep(sweeps), "CLUSTERS": plot_cluster_scatter(
        scatters, (analyzer.height, analyzer.width))}
    paths = []
    for suffix, fig in figures.items():
        fig.suptitle(f"{stem}: {len(analyzer.flashes)} flashes")
        paths.append(out_dir / f"{stem}_{suffix}.png")
        fig.savefig(paths[-1], dpi=MAP_DPI)
    return paths


# ===========================================================================
#
#   NOT SORTED YET (-> CHARACTERIZATION)
#
# ===========================================================================

# fn: striatum_of, striatum mask of one recording from the bd export.
# fn: zone_coverage, recur_zone union inside the striatum / striatum area.

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
    """Summary columns: striatum area, area of the recur_zone union inside it (um^2), and their ratio (0-1)."""
    if striatum is None:
        return {"striatum_area_um2": np.nan, "recur_zone_area_in_striatum_um2": np.nan, "striatum_coverage": np.nan}
    union = np.zeros_like(striatum)
    for mask in zone_masks.values():
        union |= mask
    striatum_px, covered_px = int(striatum.sum()), int((union & striatum).sum())
    return {
        "striatum_area_um2": striatum_px * um_per_px**2,
        "recur_zone_area_in_striatum_um2": covered_px * um_per_px**2,
        "striatum_coverage": covered_px / striatum_px,
    }


# ===========================================================================
#
#   NOT SORTED YET (-> EXPORT)
#
# ===========================================================================

# fn: _figure_to_rgb, _gray_range, _thr_text, zone-map page helpers.
# fn: _render_pages, _render_chunk, _render_pages_parallel, frame pages rendered in / out of the pool.
# fn: export_zone_maps, write {stem}_{METHOD}_ZONE_MAPS.tif.

def _figure_to_rgb(fig) -> np.ndarray:
    """Render a Figure at MAP_DPI -> (H, W, 3) uint8 RGB array."""
    fig.set_dpi(MAP_DPI)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def _gray_range(analyzer: SREAnalyzer) -> tuple[np.ndarray, np.ndarray, float]:
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


def _thr_text(analyzer: SREAnalyzer) -> str:
    """'thr = c + k × σ = thr' for map titles."""
    return (f"thr = {analyzer.bg_center:.3f} + {analyzer.sigma_ratio} × {analyzer.bg_sigma:.3f} = "
            f"{analyzer.threshold:.3f}")


# --- page rendering (serial / process pool) --------------------------------


def _render_pages(specs, ctx: dict) -> Iterator[np.ndarray]:
    """Frame specs -> RGB pages, rendered on one reused Figure (frame_zone_figures).

    spec = (raw float16 frame, flash footprints, zone ids hit, title).
    """
    center, sigma, shape = ctx["center"], ctx["sigma"], ctx["shape"]
    no_hatch = np.zeros(shape, dtype=bool)  # frame_zone_figures' NR-flash hatch layer: no NR zones any more

    def inputs() -> Iterator[tuple]:
        for frame_raw, footprints, frame_zone_ids, title in specs:
            z_frame = img_zscore_convert(frame_raw.astype(np.float32), center, sigma)
            flash_mask = np.zeros(shape, dtype=bool)
            for fp in footprints:
                flash_mask[fp[:, 0], fp[:, 1]] = True
            yield z_frame, frame_zone_ids, flash_mask, no_hatch, title

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


# --- TIFF stack -------------------------------------------------------------


def export_zone_maps(analyzer: SREAnalyzer, title_tag: str, out_path: Path,
                     striatum_outline: np.ndarray | None = None, axis_labels: tuple[str, str] | None = None,
                     pool: ProcessPoolExecutor | None = None, n_workers: int = 1) -> int:
    """Write one RGB TIFF stack: page 1 = max projection + all recur_zones, then zone by zone (1, 2, ...) every
    frame with a flash of that zone, showing only that zone's outline + flashes. A frame hit by 2 zones appears twice.

    Every page is z-scored against the fitted background and shares one gray range: z = MAP_Z_MIN
    -> median of the detections' max z, so single bright specks can't stretch it. Returns page count.
    pool: frame pages rendered on its n_workers processes (same pixels); None -> in this process.
    """
    stack = analyzer.stack_f16
    center, sigma = analyzer.bg_center, analyzer.bg_sigma
    det = analyzer.detections
    det_frames = det["frame"].to_numpy() if not det.empty else np.array([], dtype=int)

    max_proj, det_max_z, vmax = _gray_range(analyzer)
    vmin = MAP_Z_MIN
    thr_text = _thr_text(analyzer)

    # recur_zones: ids 1..N, tab20 colors
    map_masks, map_centroids = analyzer.zone_masks, analyzer.zone_centroids
    colors = zone_colors(list(map_masks))
    map_zones = analyzer.zones
    label_to_zone = {label: row.zone_id for row in map_zones.itertuples() for label in row.joint_labels}

    # page order: (zone id, frame) zone by zone, frames ascending
    det_zones = np.array([label_to_zone.get(label) for label in det["joint_label"]], dtype=object)
    zone_frames = [(z, frame) for z in map_masks for frame in np.unique(det_frames[det_zones == z])]

    overview_title = f"{title_tag}\n{len(map_masks)} recur_zones\n{thr_text}"
    first = _figure_to_rgb(plot_zone_overview(
        img_zscore_convert(max_proj.astype(np.float32), center, sigma), map_masks, map_centroids, colors, vmin,
        vmax, overview_title, analyzer.um_per_px, striatum_outline, axis_labels))

    def frame_specs() -> Iterator[tuple]:
        for zone_id, frame in zone_frames:
            in_frame = det_frames == frame
            rows = np.flatnonzero(in_frame & (det_zones == zone_id))  # only this zone's flashes are drawn
            footprints = [analyzer.footprints[i] for i in rows]
            others = [z for z in map_masks if z != zone_id and z in set(det_zones[in_frame])]
            title = (f"zone {zone_id} | frame {frame} ({frame / analyzer.fps:.2f} s) | "
                     f"max z = {det_max_z[rows].max():.2f}\n"
                     + textwrap.fill(f"other zones in this frame: {', '.join(map(str, others)) or '-'}",
                                     MAP_TITLE_WIDTH))
            yield stack[frame - 1], footprints, [zone_id], title

    ctx = {"key": (str(out_path), time.time_ns()),  # one export call -> pool workers rebuild their Figure once
           "center": center, "sigma": sigma, "shape": max_proj.shape, "masks": map_masks, "centroids": map_centroids,
           "colors": colors, "vmin": vmin, "vmax": vmax, "um_per_px": analyzer.um_per_px}

    def frame_pages() -> Iterator[np.ndarray]:
        yield first
        if pool is None:
            yield from _render_pages(frame_specs(), ctx)
        else:
            yield from _render_pages_parallel(frame_specs(), ctx, pool, n_workers)

    n_pages = 1 + len(zone_frames)
    tifffile.imwrite(out_path, frame_pages(), shape=(n_pages, *first.shape), dtype=np.uint8, photometric="rgb",
                     compression="zlib")  # pages streamed one at a time: one series (pages, H, W, 3)
    return n_pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spontaneous ACh flash -> zone analysis")
    parser.add_argument("--proc_list", required=True, type=Path, help="Proc list (proc_*.txt) naming the recordings")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Outputs go to <results_dir>/spontaneous/")
    parser.add_argument("--use_gauss", action="store_true", help="Load *_GAUSS.tif instead of *_ALS.tif (default: ALS)")
    parser.add_argument("--sigma", type=float, default=CROSSOVER_RATIO,
                        help="Threshold = background peak + this many sigmas")
    parser.add_argument("--no_mask", action="store_true",
                        help="Skip saving the per-frame flash mask (mask/{stem}_FLASH_MASK.tif)")
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
        args.debug, _cuda_available, args.db, args.exp_db, args.stbd, args.map_workers, args.use_gauss)
