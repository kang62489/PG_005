"""
Spontaneous ACh flash -> zone analysis for one deltaF/F0 stack.

  Step 1. Detect : threshold every frame -> cleaned flash mask
  Step 2. Group  : per-frame flashes -> units (circle rule) -> best-r trace-corr groups + leftover units
  Step 3. Map    : merge zones -> spatial fit of leftovers = recur_zones -> masks, per-recur_zone stats;
                   overlapping unfitted units -> non-recurring (NR) zones (reference only, not in stats)

Zone = any mapped region; recur_zone = recurring zone (the only ones in stats); NR zone = non-recurring zone.

Example:
    >>> analyzer = SpontaneousZoneAnalyzer(stack, obj="10X")
    >>> analyzer.run()
    >>> analyzer.save(out_dir, stem)
"""

## Modules
# Standard library imports
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import tifffile
from rich.console import Console
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform
from skimage.measure import find_contours, regionprops

# Local imports
from classes.region_analyzer import PIXEL_SCALE
from functions.fit_hist import fit_background
from functions.zone_kernels import footprint_traces, zone_mask

console = Console()


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Log one timed sub-step: '<label>  12.3s'."""
    t_start = time.time()
    yield
    console.log(f"[cyan]{label:<40}[/cyan] [bold]{time.time() - t_start:6.1f}s[/bold]")


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 1: detect --------------------------------------------------------
# (histogram bins / percentile range / fit live in functions/fit_hist.py -- shared with img_proc)
CROSSOVER_RATIO = 2.0         # threshold = background peak + this many fitted sigmas
TH_SMALL_OBJ = 5000           # px: drop per-frame blobs smaller than this (noise speckle)

# --- Step 2: group ---------------------------------------------------------
MAX_FLASH_FRAC = 0.8          # frame fraction: drop larger merged flashes (over-exposed first frames)
CONNECT_RADIUS = 75           # px: merge same-frame fragments whose boundaries are this close
MIN_GROUP_CORR = 0.95         # trace-corr grouping: every pair of units in a group has best r >= this

# --- Step 3: map -----------------------------------------------------------
TH_MERGE_ZONES = 0.95         # 3a: recur_zone A merges into B if shared px / A px >= this
TH_FIT_ZONES = 0.90           # 3b: leftover unit joins a recur_zone if its best flash's shared px / flash px >= this
FRAME_RATE_HZ = 20            # imaging rate, converts frames -> seconds for recur_zone event stats
MIN_EVENTS_FOR_FREQ = 2       # summary freq / period stats use only recur_zones with at least this many events


class SpontaneousZoneAnalyzer:
    """Detect -> group -> map spontaneous flashes into recur_zones (+ NR zones) for one stack.

    zones / zone_masks / zone_centroids / zone_stats hold the recur_zones only.

    Results after run():
        bg_center, bg_sigma, threshold, mask                          (step 1)
        detections, footprints, trace_corr_groups, leftover_units     (step 2)
        merge_log, fit_log, unfitted_units, overlap_log, dropped_units,
        zones, zone_masks, zone_centroids, zone_stats,
        non_recur_zones, non_recur_masks, non_recur_centroids         (step 3)
    """

    def __init__(self, stack: np.ndarray, obj: str = "10X", fps: float = FRAME_RATE_HZ,
                 sigma_ratio: float = CROSSOVER_RATIO, cuda_available: bool = False) -> None:
        self.stack_f16 = np.asarray(stack, dtype=np.float16)  # no copy when already float16
        self.n_frames, self.height, self.width = stack.shape
        self.obj = obj
        self.um_per_px = 1.0 / PIXEL_SCALE[obj]
        self.fps = fps
        self.sigma_ratio = sigma_ratio
        self.cuda_available = cuda_available

    def run(self) -> None:
        """All three steps in order."""
        self.detect()
        self.group()
        self.map()

    # -----------------------------------------------------------------------
    # Step 1. Detect
    # -----------------------------------------------------------------------

    def detect(self) -> None:
        """Background threshold -> cleaned per-frame flash mask."""
        with timed("detect 1a threshold (histogram + fit)"):
            self.bg_center, self.bg_sigma = fit_background(self.stack_f16, cuda_available=self.cuda_available)
            self.threshold = float(self.bg_center + self.sigma_ratio * self.bg_sigma)
        console.log(f"  threshold (peak + {self.sigma_ratio} sigma) = {self.threshold:.5f}")

        with timed("detect 1b mask (open/close/fill/size)"):
            self.mask = zone_mask(self.stack_f16, self.threshold, TH_SMALL_OBJ, self.cuda_available)

    # -----------------------------------------------------------------------
    # Step 2. Group
    # -----------------------------------------------------------------------

    def group(self) -> None:
        """Per-frame flashes -> units (circle rule) -> best-r trace-corr groups + leftover units."""
        # 2a. per-frame flashes (no lower size limit beyond the mask cleanup's TH_SMALL_OBJ)
        max_area = MAX_FLASH_FRAC * self.height * self.width
        with timed("group  2a connect flashes"):
            detections, self.footprints, giant = spatiotemporally_connect_flashes(
                self.mask, max_area, CONNECT_RADIUS)
        if giant:
            console.log(f"  [yellow]{len(giant)} giant flash(s) dropped (> {MAX_FLASH_FRAC:.0%} of frame) "
                        f"in frames {[frame for frame, _ in giant]}[/yellow]")
        if detections.empty:
            console.log("[yellow]group: no flashes above threshold -- 0 recur_zones[/yellow]")
            self.detections = detections
            self.trace_corr_groups = pd.DataFrame(columns=["group", "labels", "frames"])
            self.leftover_units = []
            return

        # 2b. consecutive frames -> units (circle rule)
        with timed("group  2b link units (circle rule)"):
            self.detections, n_links = link_by_circle(detections, self.footprints)
        console.log(f"  {len(detections)} flashes, {n_links} circle links -> "
                    f"{self.detections['joint_label'].nunique()} units")

        # 2c + 2d. flash traces -> best-r trace-corr grouping
        self.trace_corr_groups, self.leftover_units = group_tracks_best_r(
            self.detections, self.footprints, self.stack_f16, self.cuda_available
        )
        console.log(f"  {len(self.trace_corr_groups)} trace-corr groups (best r>={MIN_GROUP_CORR}), "
                    f"{len(self.leftover_units)} leftover units")

    # -----------------------------------------------------------------------
    # Step 3. Map
    # -----------------------------------------------------------------------

    def map(self) -> None:
        """Trace-corr groups -> merged -> fitted leftovers = recur_zones (the only ones in stats);
        unfitted units that overlap -> NR zones (reference only)."""
        shape = (self.height, self.width)
        zones = [{"labels": list(labels), "source": f"trace_corr #{i}", "fitted": [], "merged_from": [],
                  "mask": unit_mask(self.detections, self.footprints, labels, shape)}
                 for i, labels in enumerate(self.trace_corr_groups["labels"])]

        with timed("map    3a merge recur_zones"):
            self.merge_log = merge_zones(zones, TH_MERGE_ZONES)
        with timed("map    3b spatial fit of leftovers"):
            self.fit_log, self.unfitted_units = spatial_fit_zones(
                zones, self.leftover_units, self.detections, self.footprints, shape, TH_FIT_ZONES)
        with timed("map    3c NR zones"):
            self.overlap_log, self.dropped_units, non_recur = non_recur_zones(
                zones, self.unfitted_units, self.detections, self.footprints, shape)

        with timed("map    3d tables + recur_zone stats"):
            # non-recurring zones: NR1, NR2, ... -- kept apart from the recur_zones, no stats
            self.non_recur_zones = pd.DataFrame({
                "zone_id": [f"NR{k}" for k in range(1, len(non_recur) + 1)],
                "source": [zone["source"] for zone in non_recur],
                "joint_labels": [zone["labels"] for zone in non_recur],
            }, dtype=object)  # object even when empty, so .str works on source
            self.non_recur_masks = dict(zip(self.non_recur_zones["zone_id"], [zone["mask"] for zone in non_recur],
                                            strict=True))
            self.non_recur_centroids = zone_centroids(self.non_recur_zones, self.detections)

            self.zones = pd.DataFrame({
                "joint_labels": [zone["labels"] for zone in zones],
                "source": [zone["source"] for zone in zones],
                "fitted_units": [zone["fitted"] for zone in zones],
                "merged_from": [zone["merged_from"] for zone in zones],
            })
            self.zones["zone_id"] = range(1, len(zones) + 1)  # 1-based, 0 stays background
            self.zone_masks = dict(zip(self.zones["zone_id"], [zone["mask"] for zone in zones], strict=True))
            self.zone_centroids = zone_centroids(self.zones, self.detections)
            self.zone_stats = zone_stats_table(self.zones, self.zone_masks, self.detections, self.fps, self.um_per_px)
        console.log(f"  {len(self.merge_log)} merges, {len(self.fit_log) - len(self.unfitted_units)} fitted, "
                    f"{len(self.dropped_units)} single units dropped -> {len(self.zone_masks)} recur_zones "
                    f"(+ {len(self.non_recur_masks)} NR zones)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, out_dir: Path, stem: str, save_mask: bool = True, debug: bool = False) -> dict[str, Path]:
        """Write {stem}_ZONES.xlsx, footprints/{stem}_ZONES.npz (+ mask/{stem}_FLASH_MASK.tif, raw detections)."""
        (out_dir / "footprints").mkdir(parents=True, exist_ok=True)
        paths = {
            "xlsx": out_dir / f"{stem}_ZONES.xlsx",
            "npz":  out_dir / "footprints" / f"{stem}_ZONES.npz",
        }

        frames_by_label = self.detections.groupby("joint_label")["frame"].apply(list)

        def active_frames(units: list) -> list[int]:
            return sorted(f for unit in units for f in frames_by_label[unit])

        recur_zones = self.zones.rename(columns={"zone_id": "recur_zone_id", "joint_labels": "track_ids"})
        recur_zones = recur_zones[["recur_zone_id", "source", "track_ids", "fitted_units", "merged_from"]].copy()
        recur_zones["active_frames"] = recur_zones["track_ids"].apply(active_frames)  # merged + fitted included
        non_recur = self.non_recur_zones.rename(columns={"joint_labels": "track_ids"})
        non_recur["active_frames"] = non_recur["track_ids"].apply(active_frames)
        non_recur["area_px"] = non_recur["zone_id"].map(lambda z: int(self.non_recur_masks[z].sum()))
        with pd.ExcelWriter(paths["xlsx"]) as writer:
            fit_merge_counts(self).to_excel(writer, sheet_name="counts", index=False)
            self.zone_stats.to_excel(writer, sheet_name="recur_zone_stats", index=False)
            recur_zones.to_excel(writer, sheet_name="recur_zones", index=False)
            non_recur.to_excel(writer, sheet_name="non_recur_zones", index=False)
            self.merge_log.to_excel(writer, sheet_name="step3_merge", index=False)
            self.fit_log.to_excel(writer, sheet_name="step4_fit", index=False)
            self.overlap_log.to_excel(writer, sheet_name="step5_overlap", index=False)

        if save_mask:  # 1200 x 1024 x 1024 uint8 = 1.26 GB raw -> zlib
            (out_dir / "mask").mkdir(exist_ok=True)
            paths["mask"] = out_dir / "mask" / f"{stem}_FLASH_MASK.tif"
            tifffile.imwrite(paths["mask"], self.mask.astype(np.uint8) * 255, compression="zlib")

        arrays: dict[str, np.ndarray] = {"background_threshold": np.array(self.threshold)}
        for zone_id, mask in self.zone_masks.items():
            arrays[f"zone{zone_id}_footprint"] = np.argwhere(mask)
            for k, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
                arrays[f"zone{zone_id}_contour{k}"] = contour
        for nr_id, mask in self.non_recur_masks.items():  # NR1 -> non_recur1_*
            arrays[f"non_recur{nr_id[2:]}_footprint"] = np.argwhere(mask)
            for k, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
                arrays[f"non_recur{nr_id[2:]}_contour{k}"] = contour
        np.savez_compressed(paths["npz"], **arrays)

        if debug:
            paths["detections"] = out_dir / f"{stem}_DETECTIONS.csv"
            self.detections.rename(columns={"joint_label": "track_id"}).to_csv(paths["detections"], index=False)

        return paths


# STEP 1 -- DETECT lives in functions/: fit_hist.fit_background + zone_kernels.zone_mask

# ===========================================================================
#
#   STEP 2 -- GROUP: per-frame flashes -> units -> trace-corr groups
#
# ===========================================================================

# --- Union-find helpers ----------------------------------------------------

def _find(parent: dict, x: int) -> int:
    """Union-find root, with path halving."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: dict, a: int, b: int) -> None:
    """Join the sets of a and b."""
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_a] = root_b


# --- 2a. Label each frame, merge nearby fragments --------------------------

def merge_adjacent_flashes(labeled_frame: np.ndarray, connect_radius: float) -> np.ndarray:
    """Merge same-frame labels whose boundaries are within connect_radius."""
    regions = regionprops(labeled_frame)
    if len(regions) <= 1:
        return labeled_frame

    boundaries: dict[int, np.ndarray] = {}
    centroids: dict[int, np.ndarray] = {}
    radii: dict[int, float] = {}
    for region in regions:
        eroded = ndimage.binary_erosion(region.image)
        boundary_local = region.image & ~eroded
        ys, xs = np.nonzero(boundary_local)
        min_row, min_col = region.bbox[0], region.bbox[1]
        pts = np.column_stack([ys + min_row, xs + min_col]).astype(np.float64)

        centroid = np.array(region.centroid)
        boundaries[region.label] = pts
        centroids[region.label] = centroid
        radii[region.label] = float(np.linalg.norm(pts - centroid, axis=1).max()) if len(pts) else 0.0

    labels = [region.label for region in regions]
    parent = {label: label for label in labels}

    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            centroid_a, centroid_b = centroids[label_a], centroids[label_b]
            direction = centroid_b - centroid_a
            centroid_dist = float(np.linalg.norm(direction))

            # prefilter 1: too far apart even in the best case
            if centroid_dist > radii[label_a] + radii[label_b] + connect_radius:
                continue

            # prefilter 2: only the boundary halves facing each other can hold the closest points
            pts_a, pts_b = boundaries[label_a], boundaries[label_b]
            if centroid_dist > 0:
                facing_a = pts_a[(pts_a - centroid_a) @ direction >= 0]
                facing_b = pts_b[(pts_b - centroid_b) @ -direction >= 0]
            else:
                facing_a, facing_b = pts_a, pts_b
            if len(facing_a) == 0 or len(facing_b) == 0:
                continue

            min_dist = float(KDTree(facing_b).query(facing_a, k=1)[0].min())
            if min_dist <= connect_radius:
                _union(parent, label_a, label_b)

    remap = np.zeros(int(labeled_frame.max()) + 1, dtype=np.int32)
    for label in labels:
        remap[label] = _find(parent, label)
    return remap[labeled_frame]


def spatiotemporally_connect_flashes(mask: np.ndarray, max_area: float,
                                      connect_radius: int) -> tuple[pd.DataFrame, list, list[tuple[int, int]]]:
    """Per frame: label, merge fragments, keep area <= max_area.

    Returns (detections table, footprints, dropped giants [(frame 1-based, area)]).
    """
    flash_props = []
    footprints = []
    dropped = []
    label_offset = 0

    for frame_id in range(mask.shape[0]):
        frame_mask = mask[frame_id]
        labeled_frame, n_frame_labels = ndimage.label(frame_mask, structure=np.ones((3, 3)))
        if n_frame_labels > 1:
            labeled_frame = merge_adjacent_flashes(labeled_frame, connect_radius)

        for joint_label_at_frame_id in np.unique(labeled_frame[frame_mask]):
            footprint_mask = frame_mask & (labeled_frame == joint_label_at_frame_id)
            area = int(footprint_mask.sum())
            if area > max_area:
                dropped.append((frame_id + 1, area))
                continue

            coords = np.argwhere(footprint_mask)
            footprints.append(coords)
            flash_props.append({
                "frame": frame_id + 1,  # 1-based
                "joint_label": int(joint_label_at_frame_id) + label_offset,
                "centroid_y": float(coords[:, 0].mean()),
                "centroid_x": float(coords[:, 1].mean()),
                "area": area,
            })

        label_offset += n_frame_labels

    columns = ["frame", "joint_label", "centroid_y", "centroid_x", "area"]
    return pd.DataFrame(flash_props, columns=columns), footprints, dropped


# --- 2b. Consecutive frames -> units (circle rule) -------------------------

def link_by_circle(detections: pd.DataFrame, footprints: list) -> tuple[pd.DataFrame, int]:
    """Link flash B (frame n+1) to A (frame n) if B's centroid is inside A's circle -> one joint_label per unit.

    A's circle: centred on A's centroid, radius = distance to A's farthest footprint pixel. Also returns the link count.
    """
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


# --- 2c + 2d. Best-r trace-corr grouping (per-flash traces: functions/zone_kernels.footprint_traces) ---

def _renumber(raw_group_ids: np.ndarray) -> list[int]:
    """Renumber cluster ids 0, 1, 2, ... in first-seen order."""
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    return [group_id_map[raw] for raw in raw_group_ids]


def group_tracks_best_r(detections: pd.DataFrame, footprints: list, stack_f16: np.ndarray,
                        cuda_available: bool = False) -> tuple[pd.DataFrame, list]:
    """Unit-unit r = best r among their flashes' own traces; complete linkage cut at MIN_GROUP_CORR.

    Returns (trace-corr groups of >= 2 units, leftover units).
    """
    with timed("group  2c flash traces"):
        det_r = np.nan_to_num(np.corrcoef(footprint_traces(footprints, stack_f16, cuda_available)))  # flash x flash

    with timed("group  2d best-r trace-corr grouping"):
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
    result = pd.DataFrame({"joint_label": labels, "group": _renumber(raw_ids)})

    frames_by_label = detections.groupby("joint_label")["frame"].apply(lambda s: sorted(s.tolist()))
    sizes = result.groupby("group")["joint_label"].transform("size")
    trace_corr = result[sizes > 1].groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    trace_corr["group"] = range(len(trace_corr))
    trace_corr["frames"] = trace_corr["labels"].apply(
        lambda units: sorted(f for label in units for f in frames_by_label[label]))
    return trace_corr, sorted(result.loc[sizes == 1, "joint_label"].tolist())


# ===========================================================================
#
#   STEP 3 -- MAP: merge recur_zones -> spatial fit -> NR zones -> masks, centroids, stats
#
#   A zone is a dict: labels (units), source, fitted (units from 3b), merged_from (sources from 3a),
#   mask (H, W bool union of member footprints). 3a-3b edit the recur_zone list in place; 3c returns NR zones apart.
#
# ===========================================================================

def unit_mask(detections: pd.DataFrame, footprints: list, units: list, shape: tuple[int, int]) -> np.ndarray:
    """Union of all flash footprints of the given units."""
    mask = np.zeros(shape, dtype=bool)
    for i in np.flatnonzero(detections["joint_label"].isin(units).to_numpy()):
        mask[footprints[i][:, 0], footprints[i][:, 1]] = True
    return mask


# --- 3a. Merge recur_zones lying inside another ---------------------------

def merge_zones(zones: list[dict], th_merge: float) -> pd.DataFrame:
    """Merge recur_zone A into B while A is >= th_merge inside B (shared px / A px); best pair first, smaller into
    larger."""
    merge_rows = []
    while True:
        best_pair, best_inside = None, th_merge
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
    return pd.DataFrame(merge_rows, columns=["merged", "into", "inside", "merged_px", "into_px"])


# --- 3b. Spatial fit of leftover units --------------------------------------

def spatial_fit_zones(zones: list[dict], units: list, detections: pd.DataFrame, footprints: list,
                      shape: tuple[int, int], th_fit: float) -> tuple[pd.DataFrame, list]:
    """Leftover unit -> best recur_zone if its best flash is >= th_fit inside (shared px / flash px).

    Recur zone masks are fixed after 3a (order-independent). Returns (fit log, unfitted units).
    """
    columns = ["unit", "unit_px", "frames", "best_recur_zone", "best_fit", "best_frame", "flash_fits", "result"]
    merged_masks = [zone["mask"].copy() for zone in zones]
    det_labels = detections["joint_label"].to_numpy()
    det_frames = detections["frame"].to_numpy()
    fit_rows, unfitted = [], []

    for unit in sorted(units):
        rows = np.flatnonzero(det_labels == unit)
        mask = unit_mask(detections, footprints, [unit], shape)
        row = {"unit": unit, "unit_px": int(mask.sum()), "frames": det_frames[rows].tolist(), "best_recur_zone": None,
               "best_fit": np.nan, "best_frame": None, "flash_fits": [], "result": "leftover"}
        if merged_masks:
            # recur_zones x flashes: shared px / flash px; per recur_zone, the unit's best flash counts
            per_flash = np.array([[zone_mask[footprints[i][:, 0], footprints[i][:, 1]].mean() for i in rows]
                                    for zone_mask in merged_masks])
            fits = per_flash.max(axis=1)
            best = int(np.argmax(fits))
            row.update(best_recur_zone=zones[best]["source"], best_fit=float(fits[best]),
                       best_frame=int(det_frames[rows[per_flash[best].argmax()]]),
                       flash_fits=[round(float(v), 3) for v in per_flash[best]])
            if fits[best] >= th_fit:
                zones[best]["labels"].append(unit)
                zones[best]["fitted"].append(unit)
                zones[best]["mask"] |= mask
                row["result"] = f"-> {zones[best]['source']}"
        if row["result"] == "leftover":
            unfitted.append(unit)
        fit_rows.append(row)
    return pd.DataFrame(fit_rows, columns=columns), unfitted


# --- 3c. NR zones from overlapping unfitted units ---------------------------

def non_recur_zones(zones: list[dict], unfitted: list, detections: pd.DataFrame, footprints: list,
                    shape: tuple[int, int]) -> tuple[pd.DataFrame, list, list[dict]]:
    """Unfitted units -> type_1 (0 px with every recur_zone) / type_2 (the rest); within each type, units sharing
    >= 1 px are chained into a non_recur_zone_type_N zone (>= 2 units); single units are dropped.

    NR zones are not recur_zones: returned apart, the recur_zone list stays unchanged.
    Returns (overlap log, dropped units, non-recurring zones).
    """
    all_zones = np.zeros(shape, dtype=bool)
    for zone in zones:
        all_zones |= zone["mask"]
    unit_masks = {unit: unit_mask(detections, footprints, [unit], shape) for unit in unfitted}
    sets = {"type_1": [u for u, m in unit_masks.items() if not (m & all_zones).any()]}
    sets["type_2"] = [u for u in unfitted if u not in sets["type_1"]]

    dropped, overlap_rows, non_recur = [], [], []
    for kind, units in sets.items():
        parent = {u: u for u in units}
        for k, a in enumerate(units):
            for b in units[k + 1:]:
                if (unit_masks[a] & unit_masks[b]).any():
                    _union(parent, a, b)
        groups: dict[int, list] = {}
        for u in units:
            groups.setdefault(_find(parent, u), []).append(u)

        n_new = 0
        for members in groups.values():
            if len(members) < 2:
                dropped += members
                overlap_rows.append({"set": kind, "units": members, "result": "dropped"})
                continue
            source = f"non_recur_zone_{kind} #{n_new}"
            n_new += 1
            mask = np.zeros(shape, dtype=bool)
            for u in members:
                mask |= unit_masks[u]
            non_recur.append({"labels": members, "source": source, "mask": mask})
            overlap_rows.append({"set": kind, "units": members, "result": f"-> {source}"})
    return pd.DataFrame(overlap_rows, columns=["set", "units", "result"]), dropped, non_recur


# --- 3d. Centroids, recur_zone stats, counts -------------------------------

def zone_centroids(zones: pd.DataFrame, detections: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """zone id -> mean (y, x) of its member units' centroids."""
    track_xy = detections.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    centroids: dict[int, tuple[float, float]] = {}
    for row in zones.itertuples():
        members = track_xy.loc[track_xy.index.intersection(row.joint_labels)]
        if not members.empty:
            centroids[row.zone_id] = (float(members["centroid_y"].mean()), float(members["centroid_x"].mean()))
    return centroids


def zone_event_stats(frames: np.ndarray, fps: float) -> tuple[int, float, float]:
    """(n_events, mean_period_s, mean_freq_hz); an event is a run of consecutive frames.

    Period = mean start-to-start interval; NaN if only 1 event (< 1 event per recording, no interval).
    """
    frames = np.unique(frames)
    starts = frames[np.r_[True, np.diff(frames) > 1]] if len(frames) else frames
    if len(starts) < 2:
        return len(starts), np.nan, np.nan

    period_s = np.diff(starts).mean() / fps
    return len(starts), float(period_s), float(1 / period_s)


def zone_stats_table(zones: pd.DataFrame, zone_masks: dict[int, np.ndarray], detections: pd.DataFrame,
                     fps: float, um_per_px: float) -> pd.DataFrame:
    """One row per recur_zone: size (px, um^2) and event frequency/period."""
    columns = ["recur_zone_id", "source", "n_tracks", "area_px", "area_um2", "n_events", "mean_period_s",
               "mean_freq_hz"]
    if zones.empty:
        return pd.DataFrame(columns=columns)

    stats = zones[["zone_id", "source"]].rename(columns={"zone_id": "recur_zone_id"})
    stats["n_tracks"] = zones["joint_labels"].apply(len)
    stats["area_px"] = stats["recur_zone_id"].map(lambda z: int(zone_masks[z].sum()) if z in zone_masks else 0)
    stats["area_um2"] = stats["area_px"] * um_per_px ** 2

    frames_by_label = detections.groupby("joint_label")["frame"].apply(np.array)
    event_stats = zones["joint_labels"].apply(lambda labels: zone_event_stats(
        np.concatenate([frames_by_label.get(label, np.array([], dtype=int)) for label in labels]), fps))
    stats[["n_events", "mean_period_s", "mean_freq_hz"]] = pd.DataFrame(event_stats.tolist(), index=stats.index)

    return stats.sort_values("recur_zone_id").reset_index(drop=True)


def fit_merge_counts(analyzer: SpontaneousZoneAnalyzer) -> pd.DataFrame:
    """Groups, units and flashes at each grouping stage (the xlsx counts sheet)."""
    flashes_per_unit = analyzer.detections.groupby("joint_label").size()

    def n_flashes(units: list) -> int:
        return int(flashes_per_unit[units].sum())

    fit_log = analyzer.fit_log
    trace_corr = [u for labels in analyzer.trace_corr_groups["labels"] for u in labels]
    leftover = fit_log["unit"].tolist()
    fitted = fit_log.loc[fit_log["result"] != "leftover", "unit"].tolist()
    unfitted = fit_log.loc[fit_log["result"] == "leftover", "unit"].tolist()
    n_trace_corr = len(analyzer.trace_corr_groups)
    rows = [
        ("detected", np.nan, np.nan, len(analyzer.detections)),
        ("1st consecutive frames -> units", np.nan, len(flashes_per_unit), len(analyzer.detections)),
        ("2nd in trace-corr groups", n_trace_corr, len(trace_corr), n_flashes(trace_corr)),
        (f"3rd merges (inside >= {TH_MERGE_ZONES})", len(analyzer.merge_log), np.nan, np.nan),
        ("recur_zones after 3rd", n_trace_corr - len(analyzer.merge_log), len(trace_corr),
         n_flashes(trace_corr)),
        ("left after 2nd", np.nan, len(leftover), n_flashes(leftover)),
        (f"4th fitted into a recur_zone (fit >= {TH_FIT_ZONES})", np.nan, len(fitted), n_flashes(fitted)),
        ("4th not fitted -> leftover", np.nan, len(unfitted), n_flashes(unfitted)),
    ]
    for kind, label in (("type_1", "type 1: 0 px with every recur_zone"),
                        ("type_2", "type 2: touching a recur_zone")):
        log = analyzer.overlap_log[analyzer.overlap_log["set"] == kind]
        units = [u for members in log["units"] for u in members]
        new = log[log["result"] != "dropped"]
        new_units = [u for members in new["units"] for u in members]
        rows.append((f"5th leftover {label}", np.nan, len(units), n_flashes(units)))
        rows.append((f"5th non_recur_zone_{kind} (NR zones, not recur_zones)", len(new), len(new_units),
                     n_flashes(new_units)))
    in_zones = [u for labels in analyzer.zones["joint_labels"] for u in labels]
    rows += [("5th dropped (single units)", np.nan, len(analyzer.dropped_units), n_flashes(analyzer.dropped_units)),
             ("final recur_zones", len(analyzer.zones), len(in_zones), n_flashes(in_zones))]
    return pd.DataFrame(rows, columns=["stage", "n_groups", "n_units", "n_flashes"])
