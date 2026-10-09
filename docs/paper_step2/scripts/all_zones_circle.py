# ruff: noqa: INP001
"""
Scratch: all-zones map with consecutive-frame linking by the circle rule (pipeline code untouched).

  Circle rule: hotspot B (frame n+1) is linked to hotspot A (frame n) if B's centroid lies inside
               the circle around A's centroid with radius = distance to A's most distal footprint pixel.

  Best-r rule (BEST_R = True): no averaged unit trace; r between two units = best r among all pairs of
               their hotspots' own traces; trace-corr = complete linkage on these best-r values, cut at 0.95.

  Step 1. Detect (same as pipeline)
  Step 2. Group: 2a connect hotspots -> 2b circle-rule linking -> 2c-2e group_tracks (pipeline) or best-r version
  Step 3. Map (same as pipeline) -> all-zones PNG + xlsx (counts + pipeline sheets)
          + zone-map TIFF stack (pipeline export_zone_maps: all zones, then one page per frame) -> output/paper_step2/

Run from the repo root:
    .venv/Scripts/python.exe docs/paper_step2/scripts/all_zones_circle.py
"""

## Modules
# Standard library imports
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import tifffile
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Local imports
from classes import SpontaneousZoneAnalyzer  # noqa: E402
from classes.sp_zone_analyzer import (  # noqa: E402
    CONNECT_RADIUS,
    MAX_CENTROID_DEVIATION,
    MAX_HOTSPOT_FRAC,
    MIN_GROUP_CORR,
    MIN_HOTSPOT_FRAC,
    _find,
    _readable_groups,
    _renumber,
    _union,
    group_tracks,
    second_grouping,
    spatiotemporally_connect_hotspots,
    track_centroids,
    zone_centroids,
    zone_stats_table,
)
from functions import check_cuda, img_zscore_convert, plot_zone_overview, zone_colors  # noqa: E402
from functions.plot_results import _label_zone, _z_page, frame_zone_figures  # noqa: E402
from functions.zone_kernels import footprint_traces  # noqa: E402
from spontaneous_analysis import _figure_to_rgb, export_zone_maps  # noqa: E402

# ===========================================================================
#   CONFIG
# ===========================================================================

PROC_DIR = Path("proc_tiffs")
OUT_DIR = Path("output/paper_step2")
RECORDINGS = ["2025_06_11-0003", "2025_12_15-0012"]  # 10X
MAP_DPI = 120
BEST_R = True  # True: best-r trace-corr; False: pipeline's averaged-trace trace-corr
NO_MIN_FRAC = True  # True: no 1 % lower size limit (mask cleanup's 4,000 px is the only one); 80 % upper kept
SAVE_TIFF = True  # zone-map TIFF stack (slow)
FIT_MERGE = True  # True: 3rd = merge trace-corr zones inside others, 4th = fit leftovers into them (no proximity)
MIN_INSIDE = 0.95  # 3rd: zone A merges into zone B if shared px / A px >= this
MIN_FIT = 0.90  # 4th: leftover unit joins a zone if its best hotspot has shared px / hotspot px >= this


def unit_mask(analyzer: SpontaneousZoneAnalyzer, units: list) -> np.ndarray:
    """Union of all hotspot footprints of the given units."""
    mask = np.zeros((analyzer.height, analyzer.width), dtype=bool)
    for i in np.flatnonzero(analyzer.detections["joint_label"].isin(units).to_numpy()):
        mask[analyzer.footprints[i][:, 0], analyzer.footprints[i][:, 1]] = True
    return mask


def fit_and_merge(analyzer: SpontaneousZoneAnalyzer) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3rd: merge trace-corr zones inside others; 4th: leftovers -> best-fitting merged zone (fit >= MIN_FIT).

    Leftovers that fit no zone are isolated (analyzer.isolated_units); 5th: isolated units overlapping each other
    (separately for A = 0 px with every zone, B = the rest) form new zones; single ones are dropped.
    Sets analyzer.zones / zone_masks / zone_centroids / zone_stats; returns (fit log, merge log).
    """
    leftover = ([u for labels in analyzer.proximity_groups["labels"] for u in labels]
                + analyzer.isolated_tracks["joint_label"].tolist())
    zones = [{"labels": list(labels), "source": f"trace_corr #{i}", "fitted": [], "merged_from": []}
             for i, labels in enumerate(analyzer.trace_corr_groups["labels"])]
    for zone in zones:
        zone["mask"] = unit_mask(analyzer, zone["labels"])

    # 3rd: merge zone A into zone B while A lies >= MIN_INSIDE inside B (best pair first, smaller into larger)
    merge_rows = []
    while True:
        best_pair, best_inside = None, MIN_INSIDE
        for a, zone_a in enumerate(zones):
            area_a = zone_a["mask"].sum()
            for b, zone_b in enumerate(zones):
                if a == b or area_a > zone_b["mask"].sum():
                    continue
                inside = (zone_a["mask"] & zone_b["mask"]).sum() / area_a
                if inside >= best_inside:
                    best_pair, best_inside = (a, b), inside
        if best_pair is None:
            break
        a, b = best_pair
        zone_a, zone_b = zones[a], zones[b]
        merge_rows.append({"merged": zone_a["source"], "into": zone_b["source"], "inside": float(best_inside),
                           "merged_px": int(zone_a["mask"].sum()), "into_px": int(zone_b["mask"].sum())})
        zone_b["labels"] += zone_a["labels"]
        zone_b["mask"] |= zone_a["mask"]
        zone_b["merged_from"].append(zone_a["source"])
        del zones[a]

    # 4th: fit each leftover unit into the merged zones (masks fixed: order-independent); the rest are isolated
    merged_masks = [zone["mask"].copy() for zone in zones]
    fit_rows, analyzer.isolated_units = [], []
    det = analyzer.detections
    for unit in sorted(leftover):
        mask = unit_mask(analyzer, [unit])
        rows = np.flatnonzero(det["joint_label"].to_numpy() == unit)
        # best fit: per zone, the best of the unit's hotspots (shared px / hotspot px)
        per_hotspot = np.array([[zone_mask[analyzer.footprints[i][:, 0], analyzer.footprints[i][:, 1]].mean()
                                 for i in rows] for zone_mask in merged_masks])  # zones x hotspots
        fits = per_hotspot.max(axis=1) if len(merged_masks) else np.array([])
        best = int(np.argmax(fits)) if fits.size else -1
        joined = best >= 0 and fits[best] >= MIN_FIT
        if joined:
            zones[best]["labels"].append(unit)
            zones[best]["fitted"].append(unit)
            zones[best]["mask"] |= mask
        else:
            analyzer.isolated_units.append(unit)
        fit_rows.append({"unit": unit, "unit_px": int(mask.sum()), "frames": det["frame"].iloc[rows].tolist(),
                         "best_zone": zones[best]["source"] if best >= 0 else None,
                         "best_fit": float(fits[best]) if best >= 0 else np.nan,
                         "best_frame": int(det["frame"].iloc[rows[per_hotspot[best].argmax()]]) if best >= 0 else None,
                         "hotspot_fits": [round(float(v), 3) for v in per_hotspot[best]] if best >= 0 else [],
                         "result": f"-> {zones[best]['source']}" if joined else "isolated"})

    # 5th: isolated -> (A) 0 px with every zone, (B) the rest; within each set, units sharing >= 1 px are chained
    # into one zone (>= 2 units); single units are dropped
    all_zones = np.zeros((analyzer.height, analyzer.width), dtype=bool)
    for zone in zones:
        all_zones |= zone["mask"]
    iso_masks = {unit: unit_mask(analyzer, [unit]) for unit in analyzer.isolated_units}
    sets = {"A": [u for u, m in iso_masks.items() if not (m & all_zones).any()]}
    sets["B"] = [u for u in analyzer.isolated_units if u not in sets["A"]]
    analyzer.dropped_units, overlap_rows = [], []
    for kind, units in sets.items():
        parent = {u: u for u in units}
        for k, a in enumerate(units):
            for b in units[k + 1:]:
                if (iso_masks[a] & iso_masks[b]).any():
                    _union(parent, a, b)
        groups: dict[int, list] = {}
        for u in units:
            groups.setdefault(_find(parent, u), []).append(u)
        n_new = 0
        for members in groups.values():
            if len(members) < 2:
                analyzer.dropped_units += members
                overlap_rows.append({"set": kind, "units": members, "result": "dropped"})
                continue
            source = f"overlap {kind} #{n_new}"
            n_new += 1
            mask = np.zeros_like(all_zones)
            for u in members:
                mask |= iso_masks[u]
            zones.append({"labels": members, "source": source, "fitted": [], "merged_from": [], "mask": mask})
            overlap_rows.append({"set": kind, "units": members, "result": f"-> {source}"})
    analyzer.overlap_log = pd.DataFrame(overlap_rows, columns=["set", "units", "result"])

    analyzer.zones = pd.DataFrame({
        "joint_labels": [zone["labels"] for zone in zones],
        "source": [zone["source"] for zone in zones],
        "fitted_units": [zone["fitted"] for zone in zones],
        "merged_from": [zone["merged_from"] for zone in zones],
    })
    analyzer.zones["zone_id"] = range(1, len(zones) + 1)
    analyzer.zone_masks = dict(zip(analyzer.zones["zone_id"], [zone["mask"] for zone in zones], strict=True))
    analyzer.zone_centroids = zone_centroids(analyzer.zones, analyzer.detections)
    analyzer.zone_stats = zone_stats_table(analyzer.zones, analyzer.zone_masks, analyzer.detections,
                                           analyzer.fps, analyzer.um_per_px)
    return pd.DataFrame(fit_rows), pd.DataFrame(merge_rows, columns=["merged", "into", "inside", "merged_px", "into_px"])


def fit_merge_counts(analyzer: SpontaneousZoneAnalyzer, fit_log: pd.DataFrame, merge_log: pd.DataFrame) -> pd.DataFrame:
    """Units and hotspots at each stage of the fit/merge version."""
    hotspots_per_unit = analyzer.detections.groupby("joint_label").size()

    def n_hotspots(units: list) -> int:
        return int(hotspots_per_unit[units].sum())

    trace_corr = [u for labels in analyzer.trace_corr_groups["labels"] for u in labels]
    leftover = fit_log["unit"].tolist()
    fitted = fit_log.loc[fit_log["result"] != "isolated", "unit"].tolist()
    isolated = fit_log.loc[fit_log["result"] == "isolated", "unit"].tolist()
    rows = [
        ("detected", np.nan, np.nan, len(analyzer.detections)),
        ("1st consecutive frames -> units", np.nan, len(hotspots_per_unit), len(analyzer.detections)),
        ("2nd in trace-corr groups", len(analyzer.trace_corr_groups), len(trace_corr), n_hotspots(trace_corr)),
        (f"3rd merges (inside >= {MIN_INSIDE})", len(merge_log), np.nan, np.nan),
        ("zones after 3rd", len(analyzer.trace_corr_groups) - len(merge_log), len(trace_corr), n_hotspots(trace_corr)),
        ("left after 2nd", np.nan, len(leftover), n_hotspots(leftover)),
        (f"4th fitted into a zone (fit >= {MIN_FIT})", np.nan, len(fitted), n_hotspots(fitted)),
        ("4th not fitted -> isolated", np.nan, len(isolated), n_hotspots(isolated)),
    ]
    for kind, label in (("A", "A: 0 px with every zone"), ("B", "B: touching a zone")):
        log = analyzer.overlap_log[analyzer.overlap_log["set"] == kind]
        units = [u for members in log["units"] for u in members]
        new = log[log["result"] != "dropped"]
        new_units = [u for members in new["units"] for u in members]
        rows.append((f"5th isolated {label}", np.nan, len(units), n_hotspots(units)))
        rows.append((f"5th {kind} overlap groups -> new zones", len(new), len(new_units), n_hotspots(new_units)))
    dropped = analyzer.dropped_units
    in_zones = [u for labels in analyzer.zones["joint_labels"] for u in labels]
    rows += [("5th dropped (single units)", np.nan, len(dropped), n_hotspots(dropped)),
             ("final zones", len(analyzer.zones), len(in_zones), n_hotspots(in_zones))]
    return pd.DataFrame(rows, columns=["stage", "n_groups", "n_units", "n_hotspots"])


def fit_merge_title(analyzer: SpontaneousZoneAnalyzer, rec: str) -> str:
    """Two-line title for the fit/merge maps."""
    return (f"{rec} | circle + best-r + no 1% min + merge {MIN_INSIDE} + fit {MIN_FIT}\n"
            f"{len(analyzer.zone_masks)} zones ({int(analyzer.zones['source'].str.startswith('overlap').sum())} "
            f"from isolated overlap groups) | {len(analyzer.dropped_units)} single units dropped")


def export_fit_merge_tiff(analyzer: SpontaneousZoneAnalyzer, rec: str, out_path: Path) -> int:
    """Pipeline export_zone_maps layout + an all-contours page 2; hotspots of dropped units are outlined, no zone.

    Page 1 = all zones (fill), page 2 = all zone contours, then one page per detection frame. Returns page count.
    """
    stack, center, sigma, thr = analyzer.stack_f16, analyzer.bg_center, analyzer.bg_sigma, analyzer.threshold
    det = analyzer.detections
    det_frames = det["frame"].to_numpy()
    frames = np.unique(det_frames)
    max_proj = stack.max(axis=0)
    det_raw_max = np.array([stack[f - 1][fp[:, 0], fp[:, 1]].max()
                            for f, fp in zip(det_frames, analyzer.footprints, strict=True)], dtype=np.float32)
    det_max_z = img_zscore_convert(det_raw_max, center, sigma)
    vmin, vmax = 1.0, float(np.median(det_max_z))
    thr_text = f"thr = {center:.3f} + {analyzer.sigma_ratio} × {sigma:.3f} = {thr:.3f}"
    max_z = img_zscore_convert(max_proj.astype(np.float32), center, sigma)

    colors = zone_colors(sorted(analyzer.zone_masks))
    label_to_zone = {label: row.zone_id for row in analyzer.zones.itertuples() for label in row.joint_labels}
    title = fit_merge_title(analyzer, rec)
    first = _figure_to_rgb(plot_zone_overview(max_z, analyzer.zone_masks, analyzer.zone_centroids, colors,
                                              vmin, vmax, f"{title}\n{thr_text}", analyzer.um_per_px))

    fig, ax = _z_page(max_z, vmin, vmax, f"{title} -- contours\n{thr_text}", analyzer.um_per_px)
    for zone_id in sorted(analyzer.zone_masks):
        ax.contour(analyzer.zone_masks[zone_id].astype(float), levels=[0.5], colors=[colors[zone_id]], linewidths=2)
        _label_zone(ax, zone_id, analyzer.zone_centroids[zone_id])
    second = _figure_to_rgb(fig)

    def frame_inputs() -> Iterator[tuple]:
        for frame in frames:
            z_frame = img_zscore_convert(stack[frame - 1].astype(np.float32), center, sigma)
            rows = np.flatnonzero(det_frames == frame)
            hotspot_mask = np.zeros(z_frame.shape, dtype=bool)
            for i in rows:
                hotspot_mask[analyzer.footprints[i][:, 0], analyzer.footprints[i][:, 1]] = True
            labels = det["joint_label"].iloc[rows]
            frame_zone_ids = sorted({label_to_zone[label] for label in labels if label in label_to_zone})
            n_iso = sum(label not in label_to_zone for label in labels)
            text = f"zones {', '.join(map(str, frame_zone_ids)) or '-'}" + (f" + {n_iso} dropped" if n_iso else "")
            page_title = (f"frame {frame} ({frame / analyzer.fps:.2f} s) | {thr_text} | max z = "
                          f"{det_max_z[rows].max():.2f}\n" + textwrap.fill(text, 100))
            yield z_frame, frame_zone_ids, hotspot_mask, page_title

    def pages() -> Iterator[np.ndarray]:
        yield first
        yield second
        for page_fig in frame_zone_figures(frame_inputs(), max_proj.shape, analyzer.zone_masks,
                                           analyzer.zone_centroids, colors, vmin, vmax, analyzer.um_per_px):
            yield _figure_to_rgb(page_fig)

    n_pages = 2 + frames.size
    tifffile.imwrite(out_path, pages(), shape=(n_pages, *first.shape), dtype=np.uint8, photometric="rgb",
                     compression="zlib")
    return n_pages


def save_fit_merge_xlsx(analyzer: SpontaneousZoneAnalyzer, fit_log: pd.DataFrame, merge_log: pd.DataFrame,
                        path: Path) -> None:
    """counts, zone_stats, zones (members / fitted / merged), step5_overlap, step3_merge, step4_fit."""
    zones = analyzer.zones.rename(columns={"joint_labels": "track_ids"})
    zones = zones[["zone_id", "source", "track_ids", "fitted_units", "merged_from"]]
    with pd.ExcelWriter(path) as writer:
        fit_merge_counts(analyzer, fit_log, merge_log).to_excel(writer, sheet_name="counts", index=False)
        analyzer.zone_stats.to_excel(writer, sheet_name="zone_stats", index=False)
        zones.to_excel(writer, sheet_name="zones", index=False)
        analyzer.overlap_log.to_excel(writer, sheet_name="step5_overlap", index=False)
        merge_log.to_excel(writer, sheet_name="step3_merge", index=False)
        fit_log.to_excel(writer, sheet_name="step4_fit", index=False)


def link_by_circle(detections: pd.DataFrame, footprints: list) -> tuple[pd.DataFrame, int]:
    """Give hotspots linked across consecutive frames (circle rule) one joint_label; also return the link count."""
    cy, cx = detections["centroid_y"].to_numpy(), detections["centroid_x"].to_numpy()
    radius = [np.hypot(fp[:, 0] - cy[i], fp[:, 1] - cx[i]).max() for i, fp in enumerate(footprints)]
    labels = detections["joint_label"].tolist()
    parent = {label: label for label in labels}

    by_frame = {frame: grp.index for frame, grp in detections.groupby("frame")}
    n_links = 0
    for frame, current in by_frame.items():
        for i in current:
            for j in by_frame.get(frame + 1, []):
                if np.hypot(cy[j] - cy[i], cx[j] - cx[i]) <= radius[i]:
                    _union(parent, labels[i], labels[j])
                    n_links += 1

    roots = [_find(parent, label) for label in labels]
    new_id = {root: k for k, root in enumerate(pd.unique(np.array(roots)))}
    linked = detections.copy()
    linked["joint_label"] = [new_id[root] for root in roots]
    return linked, n_links


def group_tracks_best_r(detections: pd.DataFrame, footprints: list, stack_f16: np.ndarray,
                        cuda_available: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Same as pipeline group_tracks, but trace-corr uses best r between two units' own hotspot traces."""
    det_r = np.nan_to_num(np.corrcoef(footprint_traces(footprints, stack_f16, cuda_available)))  # hotspot x hotspot
    members = detections.groupby("joint_label").groups  # unit label -> row indices
    labels = list(members)
    n = len(labels)
    best_r = np.ones((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            best_r[a, b] = best_r[b, a] = det_r[np.ix_(members[labels[a]], members[labels[b]])].max()

    if n == 1:
        raw_ids = np.array([0])
    else:
        distance = 1 - best_r
        np.fill_diagonal(distance, 0)
        raw_ids = fcluster(linkage(squareform(distance, checks=False), method="complete"),
                           t=1 - MIN_GROUP_CORR, criterion="distance")
    result1 = pd.DataFrame({"joint_label": labels, "group": _renumber(raw_ids)})

    frames_by_label = detections.groupby("joint_label")["frame"].apply(lambda s: sorted(s.tolist()))

    def frames_of(unit_labels: list) -> list[int]:
        return sorted(f for label in unit_labels for f in frames_by_label[label])

    sizes1 = result1.groupby("group")["joint_label"].transform("size")
    trace_corr = result1[sizes1 > 1].groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    trace_corr["group"] = range(len(trace_corr))
    trace_corr["frames"] = trace_corr["labels"].apply(frames_of)

    # proximity (115 px) on the leftovers -- unchanged from the pipeline
    centroids = track_centroids(detections)
    leftover = centroids[centroids["joint_label"].isin(result1.loc[sizes1 == 1, "joint_label"])].reset_index(drop=True)
    result2 = second_grouping(leftover, MAX_CENTROID_DEVIATION)
    sizes2 = result2.groupby("group")["joint_label"].transform("size")
    proximity = result2[sizes2 > 1].groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    proximity["group"] = range(len(proximity))
    proximity["frames"] = proximity["labels"].apply(frames_of)

    isolated = pd.DataFrame({"joint_label": sorted(result2.loc[sizes2 == 1, "joint_label"].tolist())})
    isolated["frames"] = isolated["joint_label"].apply(lambda label: frames_by_label[label])
    return trace_corr, proximity, isolated


def group_with_circle(analyzer: SpontaneousZoneAnalyzer) -> None:
    """Pipeline step 2 with the circle rule in 2b; fills the same analyzer attributes as group()."""
    frame_px = analyzer.height * analyzer.width
    min_area = 0 if NO_MIN_FRAC else MIN_HOTSPOT_FRAC * frame_px
    detections, analyzer.footprints, _ = spatiotemporally_connect_hotspots(
        analyzer.mask, min_area, MAX_HOTSPOT_FRAC * frame_px, CONNECT_RADIUS)
    analyzer.detections, n_links = link_by_circle(detections, analyzer.footprints)
    print(f"{len(detections)} hotspots, {n_links} circle links -> {analyzer.detections['joint_label'].nunique()} units")
    grouper = group_tracks_best_r if BEST_R else group_tracks
    analyzer.trace_corr_groups, analyzer.proximity_groups, analyzer.isolated_tracks = grouper(
        analyzer.detections, analyzer.footprints, analyzer.stack_f16, analyzer.cuda_available)
    print(f"{len(analyzer.trace_corr_groups)} trace-corr, {len(analyzer.proximity_groups)} proximity, "
          f"{len(analyzer.isolated_tracks)} isolated")


def grouping_counts(analyzer: SpontaneousZoneAnalyzer) -> pd.DataFrame:
    """Units and hotspots at each grouping stage."""
    hotspots_per_unit = analyzer.detections.groupby("joint_label").size()

    def n_hotspots(units: list) -> int:
        return int(hotspots_per_unit[units].sum())

    trace_corr = [u for labels in analyzer.trace_corr_groups["labels"] for u in labels]
    proximity = [u for labels in analyzer.proximity_groups["labels"] for u in labels]
    isolated = analyzer.isolated_tracks["joint_label"].tolist()
    stages = [
        ("detected", hotspots_per_unit.index.tolist(), np.nan),
        ("1st consecutive frames -> units", hotspots_per_unit.index.tolist(), np.nan),
        ("2nd in trace-corr groups", trace_corr, len(analyzer.trace_corr_groups)),
        ("left after 2nd (-> proximity)", proximity + isolated, np.nan),
        ("3rd in proximity groups", proximity, len(analyzer.proximity_groups)),
        ("left after 3rd (isolated)", isolated, len(isolated)),
    ]
    rows = [{"stage": name, "n_groups": n_groups, "n_units": len(units), "n_hotspots": n_hotspots(units)}
            for name, units, n_groups in stages]
    rows[0]["n_units"] = np.nan  # before linking every hotspot stands alone
    return pd.DataFrame(rows)


def save_xlsx(analyzer: SpontaneousZoneAnalyzer, path: Path) -> None:
    """Same sheets as the pipeline's {stem}_ZONES.xlsx + a counts sheet first."""
    with pd.ExcelWriter(path) as writer:
        grouping_counts(analyzer).to_excel(writer, sheet_name="counts", index=False)
        analyzer.zone_stats.to_excel(writer, sheet_name="zone_stats", index=False)
        _readable_groups(analyzer.trace_corr_groups).to_excel(writer, sheet_name="trace_corr_groups", index=False)
        _readable_groups(analyzer.proximity_groups).to_excel(writer, sheet_name="proximity_groups", index=False)
        analyzer.isolated_tracks.rename(columns={"joint_label": "track_id", "frames": "active_frames"}).to_excel(
            writer, sheet_name="isolated_tracks", index=False)


def save_leftover_png(analyzer: SpontaneousZoneAnalyzer, rec: str, path: Path) -> None:
    """Units left after trace-corr (proximity + isolated), drawn like the all-zones map, labelled by unit id."""
    leftover = ([u for labels in analyzer.proximity_groups["labels"] for u in labels]
                + analyzer.isolated_tracks["joint_label"].tolist())
    det = analyzer.detections
    unit_masks, unit_centroids = {}, {}
    for unit in leftover:
        rows = det.index[det["joint_label"] == unit]
        mask = np.zeros((analyzer.height, analyzer.width), dtype=bool)
        for i in rows:
            mask[analyzer.footprints[i][:, 0], analyzer.footprints[i][:, 1]] = True
        unit_masks[unit] = mask
        unit_centroids[unit] = (det.loc[rows, "centroid_y"].mean(), det.loc[rows, "centroid_x"].mean())

    stack, center, sigma = analyzer.stack_f16, analyzer.bg_center, analyzer.bg_sigma
    title = f"{rec} | {len(leftover)} units left after trace-corr (labels = unit id / track_id)"
    fig = plot_zone_overview(img_zscore_convert(stack.max(axis=0).astype(np.float32), center, sigma),
                             unit_masks, unit_centroids, zone_colors(sorted(unit_masks)), 1.0,
                             leftover_vmax(analyzer), title, analyzer.um_per_px)
    fig.savefig(path, dpi=MAP_DPI)


def leftover_vmax(analyzer: SpontaneousZoneAnalyzer) -> float:
    """Same gray range top as the all-zones map: median of the hotspots' max z."""
    stack, center, sigma = analyzer.stack_f16, analyzer.bg_center, analyzer.bg_sigma
    det_frames = analyzer.detections["frame"].to_numpy()
    det_raw_max = np.array([stack[f - 1][fp[:, 0], fp[:, 1]].max()
                            for f, fp in zip(det_frames, analyzer.footprints, strict=True)], dtype=np.float32)
    return float(np.median(img_zscore_convert(det_raw_max, center, sigma)))


def save_all_zones_png(analyzer: SpontaneousZoneAnalyzer, rec: str, path: Path) -> None:
    """Same drawing as page 1 of {stem}_ZONE_MAPS.tif (no striatum outline)."""
    stack, center, sigma = analyzer.stack_f16, analyzer.bg_center, analyzer.bg_sigma
    det_frames = analyzer.detections["frame"].to_numpy()
    det_raw_max = np.array([stack[f - 1][fp[:, 0], fp[:, 1]].max()
                            for f, fp in zip(det_frames, analyzer.footprints, strict=True)], dtype=np.float32)
    vmax = float(np.median(img_zscore_convert(det_raw_max, center, sigma)))

    if FIT_MERGE:
        title = fit_merge_title(analyzer, rec)
    else:
        n_tc, n_px = len(analyzer.trace_corr_groups), len(analyzer.proximity_groups)
        title = (f"{rec} | CIRCLE linking{' + BEST-R trace-corr' if BEST_R else ''}"
                 f"{' + NO 1% MIN' if NO_MIN_FRAC else ''} | {len(analyzer.zone_masks)} zones -- "
                 f"{n_tc} trace-corr, {n_px} proximity, {len(analyzer.isolated_tracks)} isolated")
    fig = plot_zone_overview(img_zscore_convert(stack.max(axis=0).astype(np.float32), center, sigma),
                             analyzer.zone_masks, analyzer.zone_centroids, zone_colors(sorted(analyzer.zone_masks)),
                             1.0, vmax, title, analyzer.um_per_px)
    fig.savefig(path, dpi=MAP_DPI)


def main() -> None:
    """Detect, group with the circle rule, map, save one all-zones PNG per recording."""
    cuda_available, _ = check_cuda()
    for rec in RECORDINGS:
        print(f"=== {rec} ===")
        stack = tifffile.imread(PROC_DIR / f"{rec}_BIEXP_ALS.tif")
        analyzer = SpontaneousZoneAnalyzer(stack, obj="10X", cuda_available=cuda_available)
        del stack
        analyzer.detect()
        group_with_circle(analyzer)
        tag = f"{rec}_all_zones_circle{'_bestr' if BEST_R else ''}{'_nomin' if NO_MIN_FRAC else ''}"
        leftover_path = OUT_DIR / f"{tag}_leftover.png"
        save_leftover_png(analyzer, rec, leftover_path)
        print(f"saved {leftover_path.resolve()}")

        if FIT_MERGE:
            fit_log, merge_log = fit_and_merge(analyzer)
            tag += "_mergefit"
            xlsx_path = OUT_DIR / f"{tag}.xlsx"
            save_fit_merge_xlsx(analyzer, fit_log, merge_log, xlsx_path)
            print(fit_merge_counts(analyzer, fit_log, merge_log).to_string(index=False))
        else:
            analyzer.map()
            xlsx_path = OUT_DIR / f"{tag}.xlsx"
            save_xlsx(analyzer, xlsx_path)
            print(grouping_counts(analyzer).to_string(index=False))
        print(f"saved {xlsx_path.resolve()}")
        path = OUT_DIR / f"{tag}.png"
        save_all_zones_png(analyzer, rec, path)
        print(f"saved {path.resolve()}")
        if SAVE_TIFF and FIT_MERGE:
            tif_path = OUT_DIR / f"{tag}.tif"
            n_pages = export_fit_merge_tiff(analyzer, rec, tif_path)
            print(f"saved {n_pages}-page zone-map TIFF {tif_path.resolve()}")
        elif SAVE_TIFF:
            tif_path = OUT_DIR / f"{tag}.tif"
            rule = ("CIRCLE + BEST-R" if BEST_R else "CIRCLE") + (" + NO 1% MIN" if NO_MIN_FRAC else "")
            n_pages = export_zone_maps(analyzer, f"{rec} | {rule}", tif_path)
            print(f"saved {n_pages}-page zone-map TIFF {tif_path.resolve()}")


if __name__ == "__main__":
    main()
