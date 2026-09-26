"""
Spontaneous ACh hotspot -> zone analysis for one deltaF/F0 stack (ported from PG_010 sp_ach_zones.py).

  Step 1. Detect : threshold every frame -> cleaned hotspot mask
  Step 2. Group  : per-frame hotspots -> tracks -> trace-corr / proximity groups
  Step 3. Map    : groups -> zones -> zone masks, contours, per-zone size/event stats

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
from scipy.spatial.distance import pdist, squareform
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
TH_SMALL_OBJ = 4000           # px: drop per-frame blobs smaller than this (noise speckle)
CLOSE_RADIUS = 3              # px: morphological closing to fill small notches in a blob's shape

# --- Step 2: group ---------------------------------------------------------
MIN_HOTSPOT_FRAC = 0.01       # frame fraction: drop smaller merged hotspots (1 % of 1024 x 1024 = 10,486 px)
MAX_HOTSPOT_FRAC = 0.8        # frame fraction: drop larger merged hotspots (over-exposed first frames)
CONNECT_RADIUS = 75          # px: merge same-frame fragments whose boundaries are this close
MAX_CENTROID_DEVIATION = 115  # px: max centroid distance for frame-to-frame chaining and proximity grouping
MIN_GROUP_CORR = 0.95         # trace-corr grouping: every pair in a group has r >= this

# --- Step 3: map -----------------------------------------------------------
FRAME_RATE_HZ = 20            # imaging rate, converts frames -> seconds for zone event stats
HIGH_FREQ_FLAG_HZ = 1.0       # zones above this are flagged (e.g. manually induced hotspots), not removed
MIN_EVENTS_FOR_FREQ = 3       # summary freq / period stats use only zones with at least this many events


class SpontaneousZoneAnalyzer:
    """Detect -> group -> map spontaneous hotspots into zones for one stack.

    Results after run():
        bg_center, bg_sigma, threshold, mask               (step 1)
        detections, footprints, trace_corr_groups,
        proximity_groups, isolated_tracks                  (step 2)
        zones, zone_masks, zone_centroids, zone_stats      (step 3)
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
        """Background threshold -> cleaned per-frame hotspot mask."""
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
        """Per-frame hotspots -> chained tracks -> trace-corr groups, proximity groups, isolated tracks."""
        # 2a. per-frame hotspots
        frame_px = self.height * self.width
        min_area, max_area = MIN_HOTSPOT_FRAC * frame_px, MAX_HOTSPOT_FRAC * frame_px
        with timed("group  2a connect hotspots"):
            detections, self.footprints, dropped = spatiotemporally_connect_hotspots(
                self.mask, min_area, max_area, CONNECT_RADIUS)
        small = [frame for frame, area in dropped if area < min_area]
        giant = [frame for frame, area in dropped if area > max_area]
        console.log(f"  {len(small)} small hotspot(s) dropped (< {MIN_HOTSPOT_FRAC:.0%} of frame = {min_area:.0f} px)")
        if giant:
            console.log(f"  [yellow]{len(giant)} giant hotspot(s) dropped (> {MAX_HOTSPOT_FRAC:.0%} of frame) "
                        f"in frames {giant}[/yellow]")
        if detections.empty:
            console.log("[yellow]group: no hotspots above threshold -- 0 zones[/yellow]")
            self.detections = detections
            self.trace_corr_groups = pd.DataFrame(columns=["group", "labels", "frames"])
            self.proximity_groups = pd.DataFrame(columns=["group", "labels", "frames"])
            self.isolated_tracks = pd.DataFrame(columns=["joint_label", "frames"])
            return

        # 2b. chain across frames into tracks
        n_before = detections["joint_label"].nunique()
        with timed("group  2b chain tracks"):
            self.detections = assign_frame_adjacent_joint_labels(detections, MAX_CENTROID_DEVIATION)
        console.log(f"  {len(detections)} detections, {n_before} -> "
                    f"{self.detections['joint_label'].nunique()} tracks (<{MAX_CENTROID_DEVIATION}px chains)")

        # 2c + 2d. traces -> two-stage grouping
        self.trace_corr_groups, self.proximity_groups, self.isolated_tracks = group_tracks(
            self.detections, self.footprints, self.stack_f16, self.cuda_available
        )
        console.log(f"  {len(self.trace_corr_groups)} trace-corr (r>={MIN_GROUP_CORR}), "
                    f"{len(self.proximity_groups)} proximity, {len(self.isolated_tracks)} isolated")

    # -----------------------------------------------------------------------
    # Step 3. Map
    # -----------------------------------------------------------------------

    def map(self) -> None:
        """Groups -> zones -> zone masks, centroids, and per-zone stats."""
        with timed("map    3  zones + masks + stats"):
            self.zones = build_zones(self.trace_corr_groups, self.proximity_groups, self.isolated_tracks)
            self.zone_masks = build_zone_masks(self.zones, self.detections, self.footprints, self.height, self.width)
            self.zone_centroids = zone_centroids(self.zones, self.detections)
            self.zone_stats = zone_stats_table(self.zones, self.zone_masks, self.detections, self.fps, self.um_per_px)
        console.log(f"  {len(self.zone_masks)} zones")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(self, out_dir: Path, stem: str, save_mask: bool = True, debug: bool = False) -> dict[str, Path]:
        """Write {stem}_ZONES.xlsx, footprints/{stem}_ZONES.npz (+ mask/{stem}_ZONE_MASK.tif, raw detections)."""
        (out_dir / "footprints").mkdir(parents=True, exist_ok=True)
        paths = {
            "xlsx": out_dir / f"{stem}_ZONES.xlsx",
            "npz":  out_dir / "footprints" / f"{stem}_ZONES.npz",
        }

        with pd.ExcelWriter(paths["xlsx"]) as writer:
            self.zone_stats.to_excel(writer, sheet_name="zone_stats", index=False)
            _readable_groups(self.trace_corr_groups).to_excel(writer, sheet_name="trace_corr_groups", index=False)
            _readable_groups(self.proximity_groups).to_excel(writer, sheet_name="proximity_groups", index=False)
            self.isolated_tracks.rename(columns={"joint_label": "track_id", "frames": "active_frames"}).to_excel(
                writer, sheet_name="isolated_tracks", index=False)

        if save_mask:  # 1200 x 1024 x 1024 uint8 = 1.26 GB raw -> zlib
            (out_dir / "mask").mkdir(exist_ok=True)
            paths["mask"] = out_dir / "mask" / f"{stem}_ZONE_MASK.tif"
            tifffile.imwrite(paths["mask"], self.mask.astype(np.uint8) * 255, compression="zlib")

        arrays: dict[str, np.ndarray] = {"background_threshold": np.array(self.threshold)}
        for zone_id, mask in self.zone_masks.items():
            arrays[f"zone{zone_id}_footprint"] = np.argwhere(mask)
            for k, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
                arrays[f"zone{zone_id}_contour{k}"] = contour
        np.savez_compressed(paths["npz"], **arrays)

        if debug:
            paths["detections"] = out_dir / f"{stem}_DETECTIONS.csv"
            self.detections.rename(columns={"joint_label": "track_id"}).to_csv(paths["detections"], index=False)

        return paths


# STEP 1 -- DETECT lives in functions/: fit_hist.fit_background + zone_kernels.zone_mask

# ===========================================================================
#
#   STEP 2 -- GROUP: per-frame hotspots -> tracks -> groups
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
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_a] = root_b


# --- 2a. Label each frame, merge nearby fragments --------------------------

def merge_adjacent_hotspots(labeled_frame: np.ndarray, connect_radius: float) -> np.ndarray:
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


def spatiotemporally_connect_hotspots(mask: np.ndarray, min_area: float, max_area: float,
                                      connect_radius: int) -> tuple[pd.DataFrame, list, list[tuple[int, int]]]:
    """Per frame: label, merge fragments, keep min_area <= area <= max_area.

    Returns (detections table, footprints, dropped [(frame 1-based, area)]).
    """
    hotspots_props = []
    footprints = []
    dropped = []
    label_offset = 0

    for frame_id in range(mask.shape[0]):
        frame_mask = mask[frame_id]
        labeled_frame, n_frame_labels = ndimage.label(frame_mask, structure=np.ones((3, 3)))
        if n_frame_labels > 1:
            labeled_frame = merge_adjacent_hotspots(labeled_frame, connect_radius)

        for joint_label_at_frame_id in np.unique(labeled_frame[frame_mask]):
            footprint_mask = frame_mask & (labeled_frame == joint_label_at_frame_id)
            area = int(footprint_mask.sum())
            if not min_area <= area <= max_area:
                dropped.append((frame_id + 1, area))
                continue

            coords = np.argwhere(footprint_mask)
            footprints.append(coords)
            hotspots_props.append({
                "frame": frame_id + 1,  # 1-based
                "joint_label": int(joint_label_at_frame_id) + label_offset,
                "centroid_y": float(coords[:, 0].mean()),
                "centroid_x": float(coords[:, 1].mean()),
                "area": area,
            })

        label_offset += n_frame_labels

    columns = ["frame", "joint_label", "centroid_y", "centroid_x", "area"]
    return pd.DataFrame(hotspots_props, columns=columns), footprints, dropped


# --- 2b. Chain frame-adjacent hotspots into tracks -------------------------

def chain_frame_adjacent_centroids(hotspots_props: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Link joint_labels in consecutive frames whose centroids are < max_dist apart."""
    rows = hotspots_props.groupby("joint_label").agg(
        frame=("frame", "first"),
        centroid_y=("centroid_y", "mean"),
        centroid_x=("centroid_x", "mean"),
    ).reset_index()

    labels = rows["joint_label"].tolist()
    parent = {label: label for label in labels}

    by_frame = dict(list(rows.groupby("frame")))
    for frame, current in by_frame.items():
        next_frame = by_frame.get(frame + 1)
        if next_frame is None:
            continue
        for _, row_a in current.iterrows():
            for _, row_b in next_frame.iterrows():
                dist = np.hypot(row_a["centroid_y"] - row_b["centroid_y"], row_a["centroid_x"] - row_b["centroid_x"])
                if dist < max_dist:
                    _union(parent, row_a["joint_label"], row_b["joint_label"])

    roots = [_find(parent, label) for label in labels]
    group_id_map = {root: new for new, root in enumerate(pd.unique(np.array(roots)))}
    group_ids = [group_id_map[root] for root in roots]
    return pd.DataFrame({"joint_label": labels, "group": group_ids})


def assign_frame_adjacent_joint_labels(hotspots_props: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Give every detection in one chained track the same joint_label."""
    result = chain_frame_adjacent_centroids(hotspots_props, max_dist)
    label_to_group = dict(zip(result["joint_label"], result["group"], strict=True))

    hotspots_props = hotspots_props.copy()
    hotspots_props["joint_label"] = hotspots_props["joint_label"].map(label_to_group)
    return hotspots_props


# --- 2c. Per-track deltaF/F0 traces (per-detection traces: functions/zone_kernels.footprint_traces) ---

def track_mean_traces(hotspots_props: pd.DataFrame, det_traces: np.ndarray) -> pd.DataFrame:
    """One averaged trace per joint_label."""
    rows = [
        {"joint_label": joint_label, "trace": det_traces[group.index].mean(axis=0)}
        for joint_label, group in hotspots_props.groupby("joint_label")
    ]
    return pd.DataFrame(rows)


def track_centroids(hotspots_props: pd.DataFrame) -> pd.DataFrame:
    """One mean (y, x) centroid per joint_label."""
    grouped = hotspots_props.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
    mean_centroid = list(zip(grouped["centroid_y"], grouped["centroid_x"], strict=True))
    return pd.DataFrame({"joint_label": grouped.index, "mean_centroid": mean_centroid})


# --- 2d. Two-stage grouping ------------------------------------------------

def _renumber(raw_group_ids: np.ndarray) -> list[int]:
    """Renumber cluster ids 0, 1, 2, ... in first-seen order."""
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    return [group_id_map[raw] for raw in raw_group_ids]


def first_grouping(tracks: pd.DataFrame, min_corr: float) -> pd.DataFrame:
    """Complete-linkage clustering on trace correlation, cut at r = min_corr."""
    labels = tracks["joint_label"].tolist()
    if len(labels) == 1:
        return pd.DataFrame({"joint_label": labels, "group": [0]})

    traces = np.stack(tracks["trace"].to_numpy())
    distance = 1 - np.corrcoef(traces)
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2  # force exact symmetry

    linkage_matrix = linkage(squareform(distance, checks=False), method="complete")
    raw_group_ids = fcluster(linkage_matrix, t=1 - min_corr, criterion="distance")
    return pd.DataFrame({"joint_label": labels, "group": _renumber(raw_group_ids)})


def second_grouping(centroids: pd.DataFrame, max_dist: float) -> pd.DataFrame:
    """Complete-linkage clustering on centroid distance, cut at max_dist."""
    labels = centroids["joint_label"].tolist()
    if len(labels) == 0:
        return pd.DataFrame({"joint_label": [], "group": []})
    if len(labels) == 1:
        return pd.DataFrame({"joint_label": labels, "group": [0]})

    coords = np.stack(centroids["mean_centroid"].to_numpy())
    linkage_matrix = linkage(pdist(coords), method="complete")
    raw_group_ids = fcluster(linkage_matrix, t=max_dist, criterion="distance")
    return pd.DataFrame({"joint_label": labels, "group": _renumber(raw_group_ids)})


def group_tracks(hotspots_props: pd.DataFrame, footprints: list, stack_f16: np.ndarray,
                 cuda_available: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trace-corr grouping, then proximity grouping of leftovers -> (trace_corr, proximity, isolated)."""
    # traces
    with timed("group  2c footprint traces"):
        det_traces = footprint_traces(footprints, stack_f16, cuda_available)
        tracks = track_mean_traces(hotspots_props, det_traces)
    frames_by_label = hotspots_props.groupby("joint_label")["frame"].apply(lambda s: sorted(s.tolist()))

    def _frames_of(labels: list) -> list[int]:
        return sorted(f for label in labels for f in frames_by_label[label])

    # 1st grouping: trace correlation
    with timed("group  2d trace-corr grouping"):
        result1 = first_grouping(tracks, MIN_GROUP_CORR)
    sizes1 = result1.groupby("group")["joint_label"].transform("size")
    ungrouped_labels = result1.loc[sizes1 == 1, "joint_label"].tolist()

    trace_corr = result1[sizes1 > 1].groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    trace_corr["group"] = range(len(trace_corr))
    trace_corr["frames"] = trace_corr["labels"].apply(_frames_of)

    # 2nd grouping: centroid distance, on the leftovers only
    centroids = track_centroids(hotspots_props)
    sub_centroids = centroids[centroids["joint_label"].isin(ungrouped_labels)].reset_index(drop=True)
    with timed("group  2e proximity grouping"):
        result2 = second_grouping(sub_centroids, MAX_CENTROID_DEVIATION)
    sizes2 = result2.groupby("group")["joint_label"].transform("size")
    isolated_labels = result2.loc[sizes2 == 1, "joint_label"].tolist()

    proximity = result2[sizes2 > 1].groupby("group")["joint_label"].apply(list).reset_index(name="labels")
    proximity["group"] = range(len(proximity))
    proximity["frames"] = proximity["labels"].apply(_frames_of)

    # whatever is still alone
    isolated = pd.DataFrame({"joint_label": sorted(isolated_labels)})
    isolated["frames"] = isolated["joint_label"].apply(lambda label: frames_by_label[label])

    return trace_corr, proximity, isolated


def _readable_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Group table with readable column names for the xlsx export."""
    readable = groups.rename(columns={"group": "group_id", "labels": "track_ids", "frames": "active_frames"})
    readable.insert(1, "n_tracks", readable["track_ids"].apply(len))
    return readable


# ===========================================================================
#
#   STEP 3 -- MAP: groups -> zones -> masks, centroids, stats
#
# ===========================================================================

def build_zones(trace_corr: pd.DataFrame, proximity: pd.DataFrame, isolated: pd.DataFrame) -> pd.DataFrame:
    """One row per zone (ids 1..N across trace-corr, proximity, isolated)."""
    zones = pd.concat([
        pd.DataFrame({"joint_labels": trace_corr["labels"],
                      "source": [f"trace_corr #{i}" for i in range(len(trace_corr))]}),
        pd.DataFrame({"joint_labels": proximity["labels"],
                      "source": [f"proximity #{i}" for i in range(len(proximity))]}),
        pd.DataFrame({"joint_labels": isolated["joint_label"].apply(lambda label: [label]),
                      "source": [f"isolated #{i}" for i in range(len(isolated))]}),
    ], ignore_index=True)
    zones["zone_id"] = range(1, len(zones) + 1)  # 1-based, 0 stays background
    return zones


def build_zone_masks(zones: pd.DataFrame, detections: pd.DataFrame, footprints: list,
                     height: int, width: int) -> dict[int, np.ndarray]:
    """zone id -> (H, W) bool mask: union of all member footprints over all frames."""
    joint_label_to_zone = {label: row.zone_id for row in zones.itertuples() for label in row.joint_labels}
    zone_masks: dict[int, np.ndarray] = {}

    for i, row in enumerate(detections.itertuples()):
        zone = joint_label_to_zone.get(row.joint_label)
        if zone is None:
            continue
        coords = footprints[i]
        mask = zone_masks.setdefault(zone, np.zeros((height, width), dtype=bool))
        mask[coords[:, 0], coords[:, 1]] = True
    return zone_masks


def zone_centroids(zones: pd.DataFrame, detections: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """zone id -> mean (y, x) of its member tracks' centroids."""
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
    """One row per zone: size (px, um^2) and event frequency/period."""
    columns = ["zone_id", "source", "n_tracks", "area_px", "area_um2", "n_events", "mean_period_s", "mean_freq_hz",
               "high_freq_flag"]
    if zones.empty:
        return pd.DataFrame(columns=columns)

    stats = zones[["zone_id", "source"]].copy()
    stats["n_tracks"] = zones["joint_labels"].apply(len)
    stats["area_px"] = stats["zone_id"].map(lambda z: int(zone_masks[z].sum()) if z in zone_masks else 0)
    stats["area_um2"] = stats["area_px"] * um_per_px ** 2

    frames_by_label = detections.groupby("joint_label")["frame"].apply(np.array)
    event_stats = zones["joint_labels"].apply(lambda labels: zone_event_stats(
        np.concatenate([frames_by_label.get(label, np.array([], dtype=int)) for label in labels]), fps))
    stats[["n_events", "mean_period_s", "mean_freq_hz"]] = pd.DataFrame(event_stats.tolist(), index=stats.index)
    stats["high_freq_flag"] = stats["mean_freq_hz"] > HIGH_FREQ_FLAG_HZ

    return stats.sort_values("zone_id").reset_index(drop=True)
