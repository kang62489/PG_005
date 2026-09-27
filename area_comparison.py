"""
area_comparison.py  --  MED hotspot size vs spontaneous zone sizes, same 10X recordings.
======================================================================================
  Step 1.  MED    : larger of spike / spike+1 frame -> total hotspot area, pixel centroid, hotspot_origin
  Step 2.  Zones  : per zone, median / mean per-frame area inside its footprint (µm²), footprint centroid;
                    per (zone, active frame) event -> area_um2
  Step 3.  Match  : MED -> nearest zone centroid < MATCH_MAX_DIST_PX; Zones row -> matched_med
  Step 3b. Rank   : MED row -> n_unmatched_zones, pct_zones_smaller
  Step 4.  Groups : A estim MED / B spont MED / C unmatched zones -> n, median, IQR; Mann-Whitney A-C, B-C, A-B
  Step 5.  Export : {db dir}/area_comparison.xlsx (MED, Zones, Zone events, Groups, Group tests)

Usage:
    python area_comparison.py --db results/results_20260922/results.db [--spont_dir results/spontaneous]
"""

## Modules
# Standard library imports
import argparse
import ast
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
import tifffile
from rich.console import Console
from scipy.stats import mannwhitneyu

# Local imports
from classes import RegionAnalyzer
from classes.region_analyzer import PIXEL_SCALE
from functions import write_stats_xlsx
from functions.database_ops import _read_experiments

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 1: MED -----------------------------------------------------------
TARGET_OBJ = "10X"  # spontaneous zones exist for 10X only

# --- Step 2: zones ---------------------------------------------------------
SPONT_SUFFIX = "_BIEXP_ALS"  # {recording}{suffix}_ZONES.xlsx / .npz / _ZONE_MASK.tif
GROUP_SHEETS = {  # zone_stats "source" prefix -> sheet holding that group's active_frames
    "trace_corr": "trace_corr_groups",
    "proximity": "proximity_groups",
    "isolated": "isolated_tracks",
}

# --- Step 3: match ---------------------------------------------------------
MATCH_MAX_DIST_PX = 50  # px (~67 µm at 10X): MED centroid -> nearest zone centroid; farther = no match

# --- Step 4: groups --------------------------------------------------------
GROUP_NAMES = {"A": "estim_induced MED", "B": "spontaneous MED", "C": "unmatched zones"}  # Groups sheet "description" column
PAIRS = [("A", "C"), ("B", "C"), ("A", "B")]  # Mann-Whitney U pairs, two-sided, raw p


# ===========================================================================
#
#   STEP 1 -- MED: larger of spike / spike+1 frame, size + centroid
#
# ===========================================================================


def med_hotspot(row: dict, db_dir: Path) -> dict:
    """One MED row: the larger of the spike / spike+1 frame (total hotspot area) and its pixel centroid."""
    med_path = db_dir / "median" / row["med_filename"]
    cat_path = db_dir / "categorized" / row["med_filename"].replace("_MED.tif", "_CAT.tif")
    med_stack = tifffile.imread(med_path)
    cat_stack = tifffile.imread(cat_path)
    analyzer = RegionAnalyzer(cat_stack, med_stack, cat_stack.shape[0] // 2, obj=row["objective"])

    frames = [("spike", analyzer.spike_frame_label_frame, analyzer.spike_frame_clusters)]
    if analyzer.spike_plus1_frame_clusters is not None:
        frames.append(("spike+1", analyzer.spike_plus1_frame_label_frame, analyzer.spike_plus1_frame_clusters))
    tag, label_frame, clusters = max(frames, key=lambda f: sum(c["area_um2"] for c in f[2]))

    rows, cols = np.nonzero(label_frame >= 0)
    return {
        "recording": f"{row['exp_date']}-{row['img_serial']}",
        "hotspot_origin": row["hotspot_origin"],
        "objective": row["objective"],
        "frame": tag,
        "n_clusters": len(clusters),
        "area_um2": float(sum(c["area_um2"] for c in clusters)),
        "centroid_y": float(rows.mean()) if len(rows) else None,
        "centroid_x": float(cols.mean()) if len(cols) else None,
        "db_spike_um2": row["spike_frame_hotspot_um2"],
        "db_spike_plus1_um2": row["spike_plus1_frame_hotspot_um2"],
    }


# ===========================================================================
#
#   STEP 2 -- ZONES: median / mean per-frame area + footprint centroid
#
# ===========================================================================


def zone_frames(xlsx_path: Path) -> dict[int, list[int]]:
    """zone id -> its unique active frames (1-based), via the zone_stats 'source' -> group sheet row."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    frames: dict[int, list[int]] = {}
    for zone in sheets["zone_stats"].itertuples():
        prefix, k = zone.source.split(" #")
        group = sheets[GROUP_SHEETS[prefix]].iloc[int(k)]
        frames[int(zone.zone_id)] = sorted(set(ast.literal_eval(group["active_frames"])))
    return frames


def zone_sizes(recording: str, spont_dir: Path, um2_per_px: float) -> tuple[list[dict], list[dict]]:
    """Per zone of one recording: n_frames, median / mean per-frame area (µm²), footprint centroid;
    plus one event row per (zone, active frame) with that frame's area."""
    stem = f"{recording}{SPONT_SUFFIX}"
    frames_by_zone = zone_frames(spont_dir / f"{stem}_ZONES.xlsx")
    if not frames_by_zone:
        return [], []
    footprints = np.load(spont_dir / "footprints" / f"{stem}_ZONES.npz")

    all_frames = sorted({f for frames in frames_by_zone.values() for f in frames})
    pages = tifffile.imread(spont_dir / "mask" / f"{stem}_ZONE_MASK.tif", key=[f - 1 for f in all_frames]) > 0
    if pages.ndim == 2:
        pages = pages[None]
    page_of = {f: i for i, f in enumerate(all_frames)}

    rows, events = [], []
    for zone_id, frames in frames_by_zone.items():
        coords = footprints[f"zone{zone_id}_footprint"]
        areas_px = np.array([pages[page_of[f]][coords[:, 0], coords[:, 1]].sum() for f in frames])
        events.extend({"recording": recording, "zone_id": zone_id, "frame": f, "area_um2": float(a * um2_per_px)}
                      for f, a in zip(frames, areas_px, strict=True))
        rows.append({
            "recording": recording,
            "zone_id": zone_id,
            "n_frames": len(frames),
            "median_area_um2": float(np.median(areas_px) * um2_per_px),
            "mean_area_um2": float(areas_px.mean() * um2_per_px),
            "footprint_area_um2": float(len(coords) * um2_per_px),
            "centroid_y": float(coords[:, 0].mean()),
            "centroid_x": float(coords[:, 1].mean()),
        })
    return rows, events


# ===========================================================================
#
#   STEP 3 -- MATCH: MED hotspot -> nearest zone of the same recording
#
# ===========================================================================


def match_zone(med: dict, zones: list[dict]) -> dict:
    """matched_zone_id / match_dist_px / zone_median_um2 of the nearest zone centroid (None if none < MATCH_MAX_DIST_PX)."""
    match = {"matched_zone_id": None, "match_dist_px": None, "zone_median_um2": None}
    if med["centroid_y"] is None or not zones:
        return match
    dists = [np.hypot(z["centroid_y"] - med["centroid_y"], z["centroid_x"] - med["centroid_x"]) for z in zones]
    nearest = int(np.argmin(dists))
    if dists[nearest] < MATCH_MAX_DIST_PX:
        match = {
            "matched_zone_id": zones[nearest]["zone_id"],
            "match_dist_px": float(dists[nearest]),
            "zone_median_um2": zones[nearest]["median_area_um2"],
        }
    return match


# ===========================================================================
#
#   STEP 4 -- GROUPS: A / B / C areas, summary + pairwise Mann-Whitney
#
# ===========================================================================


def group_areas(med_rows: list[dict], zone_rows: list[dict]) -> dict[str, np.ndarray]:
    """A / B = MED area_um2 by hotspot_origin; C = median_area_um2 of zones not matched to a MED."""
    return {
        "A": np.array([m["area_um2"] for m in med_rows if m["hotspot_origin"] == "estim_induced"]),
        "B": np.array([m["area_um2"] for m in med_rows if m["hotspot_origin"] == "spontaneous"]),
        "C": np.array([z["median_area_um2"] for z in zone_rows if not z["matched_med"]]),
    }


def group_stats(areas: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    """Per-group n / median / IQR rows, and one Mann-Whitney U row per PAIRS entry (skipped if a group is empty)."""
    summary = [{
        "group": key,
        "description": GROUP_NAMES[key],
        "n": len(vals),
        "median_um2": float(np.median(vals)) if len(vals) else None,
        "q1_um2": float(np.percentile(vals, 25)) if len(vals) else None,
        "q3_um2": float(np.percentile(vals, 75)) if len(vals) else None,
    } for key, vals in areas.items()]

    tests = []
    for a, b in PAIRS:
        if not (len(areas[a]) and len(areas[b])):
            continue
        stat, p = mannwhitneyu(areas[a], areas[b], alternative="two-sided")
        tests.append({"comparison": f"{a} vs {b}", "test": "Mann-Whitney U",
                      "n": f"{len(areas[a])}/{len(areas[b])}", "statistic": float(stat), "p": float(p)})
    return summary, tests


# ===========================================================================
#
#   RUN
#
# ===========================================================================


def run(db_path: Path, spont_dir: Path) -> Path:
    """MED rows -> zone rows for the same recordings -> area_comparison.xlsx next to the DB."""
    df = _read_experiments(db_path).filter(
        (pl.col("has_region") == 1)
        & (pl.col("n_segments_detected") > 0)  # 0 % reliability -> not significant, no MED / CAT tif
        & (pl.col("objective") == TARGET_OBJ)
    )
    console.log(f"{df.height} {TARGET_OBJ} recording(s) with a MED hotspot in {db_path}")
    um2_per_px = (1.0 / PIXEL_SCALE[TARGET_OBJ]) ** 2

    med_rows, zone_rows, event_rows = [], [], []
    for row in df.iter_rows(named=True):
        med = med_hotspot(row, db_path.parent)
        med_rows.append(med)
        if not (spont_dir / f"{med['recording']}{SPONT_SUFFIX}_ZONES.xlsx").exists():
            med.update(match_zone(med, []), n_unmatched_zones=0, pct_zones_smaller=None)
            console.log(f"[yellow]{med['recording']}: no spontaneous result -- MED only[/yellow]")
            continue
        zones, events = zone_sizes(med["recording"], spont_dir, um2_per_px)
        event_rows.extend(events)

        # --- 3. match ---
        med.update(match_zone(med, zones))
        for zone in zones:
            zone["matched_med"] = zone["zone_id"] == med["matched_zone_id"]

        # --- 3b. rank vs unmatched zones ---
        unmatched = [z["median_area_um2"] for z in zones if not z["matched_med"]]
        med["n_unmatched_zones"] = len(unmatched)
        med["pct_zones_smaller"] = (100.0 * sum(a < med["area_um2"] for a in unmatched) / len(unmatched)
                                    if unmatched else None)
        zone_rows.extend(zones)
        console.log(f"{med['recording']}  ({med['hotspot_origin']})  MED {med['frame']} {med['area_um2']:.0f} µm²,"
                    f"  {len(zones)} zone(s), matched zone {med['matched_zone_id']}")

    summary, tests = group_stats(group_areas(med_rows, zone_rows))
    for g in summary:
        console.log(f"{g['group']} {g['description']}: n={g['n']}, median {g['median_um2']}")

    out_path = db_path.parent / "area_comparison.xlsx"
    write_stats_xlsx({"MED": med_rows, "Zones": zone_rows, "Zone events": event_rows,
                      "Groups": summary, "Group tests": tests}, out_path)
    console.log(f"[green]Saved -> {out_path.resolve()}[/green]")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MED hotspot size vs spontaneous zone sizes (10X).")
    parser.add_argument("--db", required=True, type=Path, help="results.db of an ach_domain_analysis run")
    parser.add_argument("--spont_dir", type=Path, default=Path("results/spontaneous"),
                        help="spontaneous_analysis output dir (ZONES.xlsx, footprints/, mask/)")
    args = parser.parse_args()

    run(args.db, args.spont_dir)
