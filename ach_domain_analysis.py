"""
ach_domain_analysis.py  --  Spike-aligned image analysis pipeline.
===================================================================
For every ana-list entry (processed TIFF + paired ABF, OBJ looked up in rec_data.db):

  Step 1. Clip        : spikes in the ABF -> one image/Vm segment per spike
  Step 2. Reliability : per-segment hotspot check -> RELIABILITY.png + VM_SUCCESS_FAIL.png
  Step 3. Median      : spike-centered median of the detected segments
  Step 4. Categorize  : bright / background per frame
  Step 5. Region+Flow : critical-frame clusters, hotspot area decay, TV-L1 flow (+ CAT keep mask)
  Step 6. Export      : results.db row, MED/CAT TIFFs, SPATIAL.png + FLOW.png

Ana list format (column names declared on the 'Picked:' line):
  [raw_tiff_name, gauss_exist, als_exist, paired_abf, abf_exist]

Detrend mode selects which processed TIFF prefix is loaded:
  BIEXP -> *_BIEXP_GAUSS.tif  or  *_BIEXP_ALS.tif

Usage:
    python ach_domain_analysis.py --ana_list data/ana_list_20260601_000.txt
                             [--detrend BIEXP] [--use_gauss]
                             [--db data/rec_data.db] [--exp_db data/exp_info.db]
"""
# Standard library imports
import argparse
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

# Third-party imports
import numpy as np
import polars as pl
from numba import config as numba_config
from rich.console import Console
from tabulate import tabulate

# Local application imports
from classes import (
    AbfClip,
    RegionAnalyzer,
    ResultsExporter,
    SpatialCategorizer,
    SpikeReliabilityChecker,
)
from functions import (
    compute_region_stats,
    count_unique_cells,
    get_cell_recording_status,
    list_parser,
    load_img_segs,
    lookup_rec_from_db,
    plot_flow_panels,
    plot_spatiotemporal_summary,
    spike_centered_median,
    write_cell_summary_xlsx,
)

console = Console()


# ── Ana list parsing ──────────────────────────────────────────────────────────


def parse_ana_list(
    ana_list_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = True,
) -> tuple[pl.DataFrame, Path, str, str]:
    """Parse an ana list file and filter rows by existence flags.

    Args:
        ana_list_path: Path to the ana list file (ana_*.txt).
        detrend_mode:  Which detrend variant to load — "BIEXP".
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


def _format_neuron_line(label: str, pairs: list[str], ratio: str) -> str:
    """Format one neuron's recording list, wrapping one (filename, detected) pair per line.

    A single-recording neuron stays on one line; multi-recording neurons get
    each pair on its own line, indented to align under the opening bracket.
    """
    prefix = f"{label} ["
    if len(pairs) <= 1:
        body = pairs[0] if pairs else ""
        return f"{prefix}{body}] {ratio}"

    indent = " " * len(prefix)
    lines = [f"{prefix}{pairs[0]},"]
    lines += [f"{indent}{p}," for p in pairs[1:-1]]
    lines.append(f"{indent}{pairs[-1]}] {ratio}")
    return "\n".join(lines)


_STATS_BLOCK_MARKER = "=" * 80 + "\nRegion Analysis Statistics"


def _strip_existing_report(text: str) -> str:
    """Drop a previously written Region Analysis Statistics block, if present.

    Lets write_stats_report() overwrite the block on re-run instead of stacking
    a duplicate copy every time the same ana list is processed again.
    """
    idx = text.find(_STATS_BLOCK_MARKER)
    return text[:idx].rstrip() if idx != -1 else text.rstrip()


def build_stats_report(db_path: Path, run_keys: set[tuple[str, str]] | None = None) -> str:
    """Format the region-analysis summary block appended to the ana list after a run.

    run_keys: optional set of (exp_date, img_serial) pairs to restrict stats to
    the current ana-list run rather than the full accumulated DB.

    Returns "" if results.db has no rows yet (nothing to report).
    """
    stats = compute_region_stats(db_path, run_keys)
    if stats.is_empty():
        return ""

    n_detected = stats["n_detected"][0]
    n_total = stats["n_total"][0]

    def _fmt(val: float | None, decimals: int = 2) -> str:
        return f"{val:.{decimals}f}" if val is not None else "N/A"

    rows_by_metric = {r["metric"]: r for r in stats.to_dicts()}
    area_metrics = ["spike_frame_hotspot_um2", "spike_plus1_frame_hotspot_um2"]
    temporal_metrics = ["lasting_time_ms"]

    area_rows = [
        {
            "Metric": r["metric"],
            "Mean": _fmt(r["mean"]),
            "Std": _fmt(r["std"]),
            "CV%": _fmt(r["cv_pct"], 1),
            "Median": _fmt(r["median"]),
            "IQR(Q1-Q3)": f"{_fmt(r['iqr_q1'], 1)}-{_fmt(r['iqr_q3'], 1)}" if r["iqr_q1"] is not None else "N/A",
            "GeoMean": _fmt(r["geomean"]),
            "GeoStd*": _fmt(r["geostd_factor"]),
            "n": r["n_detected"],
        }
        for metric in area_metrics
        if (r := rows_by_metric.get(metric)) is not None
    ]
    temporal_rows = [
        {
            "Metric": r["metric"],
            "Mean": _fmt(r["mean"]),
            "Std": _fmt(r["std"]),
            "n": r["n_detected"],
        }
        for metric in temporal_metrics
        if (r := rows_by_metric.get(metric)) is not None
    ]
    area_table = tabulate(area_rows, headers="keys", tablefmt="pretty") if area_rows else ""
    temporal_table = tabulate(temporal_rows, headers="keys", tablefmt="pretty") if temporal_rows else ""
    table = (
        "Spatial (skewed — Median/IQR/GeoMean shown):\n" + area_table
        + "\n\nTemporal:\n" + temporal_table
    )

    recordings = get_cell_recording_status(db_path, run_keys)
    neuron_lines = []
    for (animal_id, slice_val, at), group in recordings.group_by(["ANIMAL_ID", "SLICE", "AT"], maintain_order=True):
        site_code = ResultsExporter.derive_site_code(at)
        label = f"{animal_id}_S{slice_val}{site_code}"
        pairs = [f"({r['med_filename']}, {r['detected']})" for r in group.iter_rows(named=True)]
        ratio = f"{int(group['detected'].sum())}/{group.height}"
        neuron_lines.append(_format_neuron_line(label, pairs, ratio))

    return (
        "\n\n" + "=" * 80 + "\n"
        f"Region Analysis Statistics — generated {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        + "=" * 80 + "\n"
        f"Neurons detected ACh release: {n_detected}/{n_total}\n\n"
        f"{table}\n\n"
        "Per-Neuron Recording List (filename, detected) [detected/total]:\n"
        + "\n".join(neuron_lines) + "\n"
    )


def write_stats_report(
    ana_list_path: Path, results_db_path: Path, run_keys: set[tuple[str, str]] | None = None
) -> bool:
    """Write (or overwrite, on re-run) the region-analysis stats block in an ana list.

    run_keys: optional set of (exp_date, img_serial) pairs to restrict stats to
    the current ana-list run rather than the full accumulated DB.

    Returns False if results_db_path has no rows yet (nothing written).
    """
    report = build_stats_report(results_db_path, run_keys)
    if not report:
        return False

    original = ana_list_path.read_text(encoding="utf-8")
    kept = _strip_existing_report(original)
    ana_list_path.write_text(kept + report, encoding="utf-8")
    return True


# ===========================================================================
#
#   PIPELINE -- per entry: 1 Clip -> 2 Reliability -> 3 Median -> 4 Categorize
#                          -> 5 Region + Flow -> 6 Export
#
# ===========================================================================


def _save_entry_figures(exporter: ResultsExporter, figures: list[tuple[str, object, str]]) -> None:
    """Save (category, figure, filename) triples -- runs on a background thread."""
    for category, fig, filename in figures:
        exporter.export_figure(category, fig, filename)


def _log_skip(ana_list_path: Path, msg: str) -> None:
    """Append a '[SKIPPED] ...' line to the ana list."""
    with ana_list_path.open("a", encoding="utf-8") as f:
        f.write(f"[SKIPPED] {msg}\n")


def analyze_entry(
    row: dict,
    progress: tuple[int, int],
    df_checked_tiff: pl.DataFrame,
    animal_idx_lut: dict,
    exporter: ResultsExporter,
    results_dir: Path,
    detrend_mode: str,
    normalization: str,
    ana_list_path: Path,
    emitter=None,
) -> list[tuple[str, object, str]]:
    """Analyze one ana-list entry; returns the (category, figure, filename) triples still to be saved."""
    entry_t0 = time.time()
    i, total = progress

    # ----- STEP 1. Clip: spikes in the ABF -> one image/Vm segment per spike -----
    match = df_checked_tiff.filter(pl.col("Filename") == row["raw_tiff_name"])
    if match.is_empty():
        console.log(f"[yellow]Skipped {row['raw_tiff_name']}: not found in rec_data.db[/yellow]")
        return []
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
    if not clip.lst_img_frame_ranges:  # every spike skipped (too closely spaced for a baseline window)
        console.log("[yellow]No valid segments — skipping z-score step.[/yellow]")
        _log_skip(ana_list_path, f"{proc_tiff_path.name}: no valid segments "
                                 "(spikes too closely spaced for any baseline window)")
        return []

    # Filename / DB metadata, needed from step 2 on
    export_data = clip.get_export_data()
    animal_id = match["ANIMAL_ID"].item()
    animal_idx = animal_idx_lut[export_data["exp_date"]][animal_id]
    slice_val = match["SLICE"].item()
    at = match["AT"].item()
    frame_duration_ms = clip.ts_imgs * 1000
    name_args = (animal_idx, slice_val, at, detrend_mode, normalization)

    def export_stem(file_type: str) -> str:
        return ResultsExporter.build_export_stem(export_data["exp_date"], export_data["img_serial"], *name_args, file_type)

    # ----- STEP 2. Reliability: per-segment hotspot check + montage + success/failure Vm -----
    if emitter:
        emitter({"type": "step", "msg": "Loading raw segments..."})
    lst_segments = load_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)  # detrended, unnormalized
    console.log(f"[green]Loaded {len(lst_segments)} segment(s)  ({time.time() - entry_t0:.1f}s)[/green]")
    spike_frame_idx = lst_segments[0].shape[0] // 2  # segments are symmetric around their spike

    if emitter:
        emitter({"type": "step", "msg": "Checking per-segment reliability..."})
    reliability_checker = SpikeReliabilityChecker(obj)
    seg_results, reliability_pct = reliability_checker.check(lst_segments, spike_frame_idx)
    n_detected = sum(r["detected"] for r in seg_results)
    n_total = len(seg_results)
    console.log(
        f"[green]Reliability: {n_detected}/{n_total} segment(s) detected ({reliability_pct:.1f}%)"
        f"  ({time.time() - entry_t0:.1f}s)[/green]"
    )
    # Exported regardless of significance -- they explain a "no detection" result.
    reliability_checker.export_montage(exporter, proc_tiff_path.stem, export_data, *name_args)
    reliability_checker.export_vm_groups(exporter, proc_tiff_path.stem, clip.get_vm_segments(), export_data, *name_args)

    # ----- STEP 3. Median: detected segments only (all segments if none detected) -----
    segments_for_median = [
        seg for seg, r in zip(lst_segments, seg_results, strict=True) if r["detected"]
    ] or lst_segments
    median_segment, intensity_range = spike_centered_median(segments_for_median)
    del lst_segments
    console.log(f"[green]Median shape: {median_segment.shape}, intensity range: [{intensity_range[0]:.2f}, {intensity_range[1]:.2f}][/green]")

    # ----- STEP 4. Categorize: bright / background per frame -----
    if emitter:
        emitter({"type": "step", "msg": "Categorizing spike frame..."})
    categorizer = SpatialCategorizer.morphological(threshold_method="baseline_n_sigma")
    categorizer.fit(median_segment, spike_frame_idx=spike_frame_idx)
    console.log(
        f"[green]Categorized {len(categorizer.categorized_frames)} frame(s), threshold: {categorizer.threshold_used}"
        f"  ({time.time() - entry_t0:.1f}s)[/green]"
    )

    # ----- STEP 5. Region + Flow -----
    # --- 5a. critical-frame clusters, spike / spike+1 sizes, decay ---
    cat_stack = np.array(categorizer.categorized_frames)
    region_analyzer = RegionAnalyzer(cat_stack, median_segment, spike_frame_idx, obj=obj)
    region_results = region_analyzer.get_results()
    final_significant = n_detected > 0 and region_analyzer.significant  # 0% reliability -> no detection

    if region_results["n_clusters"] == 0:
        console.log("[yellow]No cluster detected[/yellow]")
    else:
        frame_tag = "spike" if region_results["critical_frame_offset"] == 0 else f"spike{region_results['critical_frame_offset']:+d}"
        console.log(f"[magenta]{region_results['n_clusters']} cluster(s) on {frame_tag} frame[/magenta]")
        for k, cluster in enumerate(region_results["clusters"]):
            console.log(f"[cyan]  cluster {k}: R_lat={cluster['R_lat_um']:.1f} µm  centroid={cluster['centroid']}[/cyan]")

    if final_significant:
        for frame_tag_, clusters in (
            ("spike", region_results["spike_frame_clusters"]),
            ("spike+1", region_results["spike_plus1_frame_clusters"]),
        ):
            if clusters is None:
                continue
            if not clusters:
                console.log(f"[green]{frame_tag_} frame: no hotspot clusters[/green]")
                continue
            sizes = ", ".join(f"cluster {c['cluster_id']}={c['area_um2']:.0f} µm² ({c['area_px']} px)" for c in clusters)
            console.log(f"[green]{frame_tag_} frame: {sizes}[/green]")

    if final_significant and region_results["n_clusters"] > 0:
        lasting_time_ms = region_analyzer.get_lasting_time_ms(frame_duration_ms)
        if lasting_time_ms is not None:
            console.log(
                f"[green]Lasting time (decay tau): {lasting_time_ms:.0f} ms "
                f"(R²={region_results['decay_fit_r2']:.2f})[/green]"
            )
        else:
            console.log("[yellow]Lasting time: decay fit failed / insufficient post-peak data[/yellow]")
    else:
        lasting_time_ms = None
        if not final_significant:
            console.log("[yellow]No ACh detection — skipping MED/CAT TIFFs, flow/lasting time export[/yellow]")
        _log_skip(ana_list_path, f"{proc_tiff_path.name}: no significant ACh detection "
                                 f"(reliability {reliability_pct:.1f}% -- {n_detected}/{n_total} segments detected)")

    # --- 5b. TV-L1 flow (significant recordings only) ---
    if final_significant:
        if emitter:
            emitter({"type": "step", "msg": "Computing hotspot flow..."})
        region_analyzer.compute_flow(cat_stack, median_segment)
        console.log(f"[green]Flow: {len(region_analyzer.flow_pairs)} pair(s)  ({time.time() - entry_t0:.1f}s)[/green]")

    # ----- STEP 6. Export: DB row + MED/CAT TIFFs now, figures on a background thread -----
    if emitter:
        emitter({"type": "step", "msg": "Exporting results..."})
    dirs = exporter.export_all(
        exp_date=export_data["exp_date"],
        abf_serial=export_data["abf_serial"],
        img_serial=export_data["img_serial"],
        animal_idx=animal_idx,
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
        intensity_range=intensity_range,
        region_summary=region_analyzer.get_summary(),
        region_data=region_results,
        lasting_time_ms=lasting_time_ms,
        significant=final_significant,
        reliability_pct=reliability_pct,
        n_segments_detected=n_detected,
        n_segments_total=n_total,
    )

    figures: list[tuple[str, object, str]] = []
    if final_significant:
        title_info = {
            "animal_id": animal_id,
            "slice": slice_val,
            "at": at,
            "obj": obj,
            "tiff_serial": export_data["img_serial"],
            "abf_serial": export_data["abf_serial"],
        }
        spatial_fig = plot_spatiotemporal_summary(
            categorizer, region_analyzer, spike_frame_idx, title_info, clip.get_vm_segments(), frame_duration_ms
        )
        flow_fig = plot_flow_panels(median_segment, region_analyzer.flow_pairs, title_info, frame_duration_ms)
        figures = [
            ("spatial", spatial_fig, f"{export_stem('SPATIAL')}.png"),
            ("flow", flow_fig, f"{export_stem('FLOW')}.png"),
        ]

    dir_names = "/, ".join(d.name for d in dirs.values())
    console.log(
        f"[green]Exported {dir_names}/, reliability/, spatial/, flow/  (entry: {time.time() - entry_t0:.1f}s)[/green]"
    )
    return figures


def run(
    ana_list_path: Path,
    detrend_mode: str = "BIEXP",
    use_als: bool = True,
    db_path: Path = Path("data/rec_data.db"),
    exp_db_path: Path = Path("data/exp_info.db"),
    emitter=None,
) -> None:
    """Run the full spike-aligned analysis pipeline for every entry in an ana list."""
    run_t0 = time.time()

    # On a SLURM cluster (e.g. deigo), cgroups can cap this job to far fewer CPUs than the
    # node physically has — numba's parallel=True median kernel only ever sees this count.
    cpu_affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    console.log(
        f"[bold]CPUs available to this job: {cpu_affinity}  "
        f"(numba NUMBA_NUM_THREADS={numba_config.NUMBA_NUM_THREADS})[/bold]"
    )

    entries, results_dir, detrend_mode, normalization = parse_ana_list(ana_list_path, detrend_mode, use_als)

    df_checked_tiff = lookup_rec_from_db(entries, db_path, exp_db_path)
    df_cell_group = count_unique_cells(df_checked_tiff)
    run_keys: set[tuple[str, str]] = {
        tuple(Path(name).stem.split("-", 1))  # type: ignore[misc]
        for name in entries["raw_tiff_name"].to_list()
    }
    console.log(f"Found {len(entries)} entries in {ana_list_path.name} -> {len(df_cell_group)} unique cells")

    xlsx_dir = results_dir / "spikes"
    xlsx_dir.mkdir(parents=True, exist_ok=True)
    cell_summary_path = xlsx_dir / f"{ana_list_path.stem}_cells.xlsx"
    write_cell_summary_xlsx(df_cell_group, cell_summary_path)
    console.log(f"Saved cell summary -> {cell_summary_path.name}")

    animal_idx_lut = ResultsExporter.build_animal_idx_lut(df_checked_tiff)
    exporter = ResultsExporter(results_root=results_dir)

    total = len(entries)
    figure_export_thread: threading.Thread | None = None

    for i, row in enumerate(entries.iter_rows(named=True), 1):
        # Previous entry's PNG export must finish first (matplotlib isn't thread-safe) -- 1 thread max.
        if figure_export_thread is not None:
            figure_export_thread.join()
            figure_export_thread = None
        figures = analyze_entry(
            row, (i, total), df_checked_tiff, animal_idx_lut, exporter,
            results_dir, detrend_mode, normalization, ana_list_path, emitter,
        )
        if figures:
            figure_export_thread = threading.Thread(target=_save_entry_figures, args=(exporter, figures))
            figure_export_thread.start()

    if figure_export_thread is not None:
        figure_export_thread.join()

    if write_stats_report(ana_list_path, exporter.db_path, run_keys):
        console.log(f"[green]Updated region analysis statistics -> {ana_list_path.name}[/green]")

    console.log(f"\n[bold green]All done!  (total: {time.time() - run_t0:.1f}s)[/bold green]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike-aligned image analysis pipeline")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_*.txt)")
    parser.add_argument("--detrend", choices=["BIEXP"], default="BIEXP", help="Detrend mode (default: BIEXP)")
    parser.add_argument("--use_gauss", action="store_true", help="Load *_GAUSS.tif instead of *_ALS.tif (default: ALS)")
    parser.add_argument("--db", type=Path, default=Path("data/rec_data.db"), help="Path to rec_data.db (default: data/rec_data.db)")
    parser.add_argument("--exp_db", type=Path, default=Path("data/exp_info.db"), help="Path to exp_info.db (default: data/exp_info.db)")
    args = parser.parse_args()

    run(args.ana_list, args.detrend, not args.use_gauss, args.db, args.exp_db)

