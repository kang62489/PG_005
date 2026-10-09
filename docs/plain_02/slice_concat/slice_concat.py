# ruff: noqa: INP001, E402
"""
slice_concat.py  --  Idea test: recur_zones found per recording (separated) vs on all recordings of one slice / site
concatenated. Test only -- not part of the pipeline; output goes to output/test02/slice_concat/.

Steps
-----
1. Select: proc list -> 10X recordings with the same date + SLICE + AT (site) as in rec_data.db.
2. Separated: per recording, SpontaneousZoneAnalyzer.run() as in the pipeline -> its own recur_zones.
3. Pool: the units of every recording (after step 2b), unit labels offset so they stay unique.
4. Concatenated traces: per recording, the trace of EVERY pooled flash footprint on that recording, z-scored with its
   own background (peak, sigma) -> appended in time (each recording is read twice; no concatenated stack in memory).
5. Best-r grouping (same rule as step 2d, MIN_GROUP_CORR) -> merge / fit leftovers / NR zones (pipeline step 3).
   Steps 2-5 are cached ({tag}_CACHE.pkl): re-runs only redraw.
6. Output, per circle mode (TIGHT = inner, LOOSE = furthest):
   {tag}_SEPARATED_VS_JOINED_{mode}.png: left = joined (concatenated), right = per separated recording; both the
   largest zone of each group (no NR); {tag}_JOINED_ZONE_CONTOURS_{mode}.png of the joined zones;
   {tag}_JOINED_ZONES.xlsx (units per recording in each joined recur_zone). No stats.

Usage:
    .venv/Scripts/python.exe docs/plain_02/slice_concat/slice_concat.py [date SLICE AT]
        (default: 2025_06_11 4R CELL_1)
"""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from functions import check_cuda

_CUDA, _MSG = check_cuda()  # must run before anything imports numba

import numpy as np
import pandas as pd
import polars as pl
import tifffile
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from classes import SpontaneousZoneAnalyzer
from classes.sp_zone_analyzer import (
    CROSSOVER_RATIO,
    MIN_GROUP_CORR,
    TH_FIT_ZONES,
    TH_MERGE_ZONES,
    merge_zones,
    non_recur_zones,
    spatial_fit_zones,
    unit_mask,
)
from functions import group_zones, list_parser, load_st_bd, lookup_rec_from_db, plot_zone_groups, zone_colors
from functions.plot_results import _outline_panel
from functions.zone_kernels import footprint_traces
from spontaneous_analysis import MAP_DPI, PROC_SUFFIX

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROC_LIST = ROOT / "data" / "proc_20260922_000.txt"
STBD = ROOT / "data" / "bd_20260922_000.json"
OUT_DIR = ROOT / "output" / "test02" / "slice_concat"
DB, EXP_DB = ROOT / "data" / "rec_data.db", ROOT / "data" / "exp_info.db"
DATE, SLICE, SITE = sys.argv[1:4] if len(sys.argv) > 3 else ("2025_06_11", "4R", "CELL_1")
OBJ = "10X"
SEPARATED_COLS = 5  # separated recordings per row (right side)
MODES = (("inner", "TIGHT"), ("far", "LOOSE"))


def select() -> list[tuple[str, Path]]:
    """Step 1: [(raw stem, ALS path)] of the slice / site, in recording order."""
    table, io_dirs = list_parser(PROC_LIST)
    info = lookup_rec_from_db(table.select("raw_tiff_name"), DB, EXP_DB)
    info = info.filter(pl.col("Filename").str.starts_with(DATE) & (pl.col("SLICE") == SLICE)
                       & (pl.col("AT") == SITE) & (pl.col("OBJ") == OBJ)).sort("Filename")
    proc_dir = Path(io_dirs["dir_proc_tiffs"])
    picked = [(Path(name).stem, proc_dir / f"{Path(name).stem}{PROC_SUFFIX}") for name in info["Filename"]]
    return [(stem, path) for stem, path in picked if path.exists()]


def best_r_groups(det_r: np.ndarray, unit_of_flash: np.ndarray) -> tuple[list[list], list]:
    """Step 5a: unit-unit r = best r among their flashes; complete linkage cut at MIN_GROUP_CORR (as step 2d).

    Returns (groups of >= 2 units, leftover units).
    """
    order = np.argsort(unit_of_flash, kind="stable")
    units, starts = np.unique(unit_of_flash[order], return_index=True)
    sorted_r = det_r[np.ix_(order, order)]
    best_r = np.maximum.reduceat(np.maximum.reduceat(sorted_r, starts, axis=0), starts, axis=1)
    np.fill_diagonal(best_r, 1)
    distance = 1 - best_r
    np.fill_diagonal(distance, 0)
    ids = fcluster(linkage(squareform(distance, checks=False), method="complete"), t=1 - MIN_GROUP_CORR,
                   criterion="distance")
    groups = pd.Series(units).groupby(ids).apply(list)
    return [g for g in groups if len(g) > 1], sorted(u for g in groups if len(g) == 1 for u in g)


def compute(recordings: list[tuple[str, Path]]) -> dict:
    """Steps 2-5: separated zones per recording + concatenated zones + units per recording per concatenated zone."""
    separated, det_parts, footprints, bg = {}, [], [], {}
    label_offset = 0
    for stem, path in recordings:
        analyzer = SpontaneousZoneAnalyzer(tifffile.imread(path), obj=OBJ, sigma_ratio=CROSSOVER_RATIO,
                                           cuda_available=_CUDA)
        analyzer.run()
        shape = (analyzer.height, analyzer.width)
        separated[stem] = (analyzer.zone_masks, analyzer.non_recur_masks)
        bg[stem] = (analyzer.bg_center, analyzer.bg_sigma)
        if not analyzer.detections.empty:
            det = analyzer.detections.assign(recording=stem)
            det["joint_label"] += label_offset
            label_offset = int(det["joint_label"].max()) + 1
            det_parts.append(det)
            footprints += analyzer.footprints
        print(f"  {stem}: {len(analyzer.zone_masks)} recur_zones, {len(analyzer.detections)} flashes")
        del analyzer
    detections = pd.concat(det_parts, ignore_index=True)  # row i <-> footprints[i]

    # Step 4: concatenated traces, z-scored per recording
    parts = []
    for stem, path in recordings:
        center, sigma = bg[stem]
        traces = footprint_traces(footprints, np.asarray(tifffile.imread(path), dtype=np.float16), _CUDA)
        parts.append((traces.astype(np.float32) - center) / sigma)
    concat_traces = np.concatenate(parts, axis=1)
    del parts
    print(f"concatenated traces: {concat_traces.shape[0]} flashes x {concat_traces.shape[1]} frames")
    det_r = np.nan_to_num(np.corrcoef(concat_traces))
    del concat_traces

    # Step 5: grouping + pipeline step 3 on the pooled units
    groups, leftovers = best_r_groups(det_r, detections["joint_label"].to_numpy())
    zones = [{"labels": list(labels), "source": f"trace_corr #{i}", "fitted": [], "merged_from": [],
              "mask": unit_mask(detections, footprints, labels, shape)} for i, labels in enumerate(groups)]
    merge_zones(zones, TH_MERGE_ZONES)
    _, unfitted = spatial_fit_zones(zones, leftovers, detections, footprints, shape, TH_FIT_ZONES)
    _, _, non_recur = non_recur_zones(zones, unfitted, detections, footprints, shape)
    print(f"concatenated: {len(groups)} trace-corr groups, {len(leftovers)} leftover units -> {len(zones)} "
          f"recur_zones (+ {len(non_recur)} NR zones)")

    unit_rec = detections.groupby("joint_label")["recording"].first()
    rows = []
    for k, zone in enumerate(zones, 1):
        counts = unit_rec.loc[zone["labels"]].value_counts()
        rows.append({"recur_zone_id": k, "area_px": int(zone["mask"].sum()), "n_units": len(zone["labels"]),
                     "n_recordings": len(counts), **{stem: int(counts.get(stem, 0)) for stem, _ in recordings}})
    return {"shape": shape, "separated": separated,
            "concatenated": ({k: zone["mask"] for k, zone in enumerate(zones, 1)},
                             {f"NR{k}": zone["mask"] for k, zone in enumerate(non_recur, 1)}),
            "rows": rows}


def separated_vs_concatenated(res: dict, mode: str, title: str, outline: np.ndarray | None) -> Figure:
    """Left (2 rows tall): concatenated, right: per separated recording -- both the largest zone of each group, no NR."""
    shape, separated = res["shape"], res["separated"]
    concat = res["concatenated"][0]
    n_rows = -(-len(separated) // SEPARATED_COLS)
    fig = Figure(figsize=(3.6 * (SEPARATED_COLS + 2), 3.8 * n_rows + 0.6), layout="constrained")
    fig.suptitle(title, fontsize=13)
    grid = fig.add_gridspec(n_rows, SEPARATED_COLS + 2)
    largest = [ids[0] for _, ids in group_zones(concat, mode)[0]]
    _outline_panel(fig.add_subplot(grid[:, :2]), largest, concat, zone_colors(list(concat)), set(),
                   f"JOINED ({len(separated)} recordings): {len(concat)} zones -> {len(largest)} largest",
                   shape, outline)
    for k, (stem, (masks, _)) in enumerate(separated.items()):
        largest = [ids[0] for _, ids in group_zones(masks, mode)[0]]
        _outline_panel(fig.add_subplot(grid[k // SEPARATED_COLS, 2 + k % SEPARATED_COLS]), largest, masks,
                       zone_colors(list(masks)), set(), f"{stem}: {len(masks)} zones -> {len(largest)} largest",
                       shape, outline)
    return fig


def main() -> None:
    """Steps 1-6."""
    print(_MSG)
    t0 = time.time()
    recordings = select()
    tag = f"{DATE}_{SLICE}_{SITE}"
    print(f"{tag}: {len(recordings)} recordings {[s for s, _ in recordings]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"{tag}_CACHE.pkl"
    if cache.exists():
        res = pickle.loads(cache.read_bytes())
        print(f"loaded {cache}")
    else:
        res = compute(recordings)
        cache.write_bytes(pickle.dumps(res))

    outline = load_st_bd(STBD)["recordings"].get(recordings[0][0], {}).get("striatum_outline_px")
    outline = np.array(outline) if outline is not None else None
    concat, concat_nr = res["concatenated"]
    out_paths = []
    for mode, suffix in MODES:
        out_paths.append(OUT_DIR / f"{tag}_SEPARATED_VS_JOINED_{suffix}.png")
        separated_vs_concatenated(res, mode, f"{tag}: joined (left) vs separated recordings (right); "
                                             f"largest zone of each group, {suffix} grouping, no NR", outline).savefig(
            out_paths[-1], dpi=MAP_DPI)
        zone_groups, circles = group_zones(concat, mode)
        out_paths.append(OUT_DIR / f"{tag}_JOINED_ZONE_CONTOURS_{suffix}.png")
        plot_zone_groups(concat, concat_nr, zone_groups, circles, mode, f"{tag} joined", res["shape"],
                         outline).savefig(out_paths[-1], dpi=MAP_DPI)

    out_paths.append(OUT_DIR / f"{tag}_JOINED_ZONES.xlsx")
    with pd.ExcelWriter(out_paths[-1]) as writer:
        pd.DataFrame(res["rows"]).to_excel(writer, sheet_name="joined_recur_zones", index=False)
        pd.DataFrame([{"recording": s, "n_recur_zones": len(m), "n_nr_zones": len(nr)}
                      for s, (m, nr) in res["separated"].items()]).to_excel(writer, sheet_name="separated",
                                                                           index=False)
    for path in out_paths:
        print(f"saved {path}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
