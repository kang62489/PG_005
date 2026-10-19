"""
Spontaneous ACh flash -> zone analysis for one deltaF/F0 stack.

  CLASS  SREAnalyzer             : detect() -> group_centroid() / group_trace() -> map(method) -> save_*()
  BLOCK  SHARED HELPERS          : timed, cut_h_name

Flash mask: functions/flash_mask.py; footprint traces: functions/footprint_traces.py;
clustering (centroid tree cut at the acceleration elbow, trace_corr tree at the best-silhouette h):
classes/hierarchical_clustering.py.

recur_zone = recurring zone (non-recurring zones are not defined for the clustering yet).

Example:
    >>> analyzer = SREAnalyzer(stack, obj="10X")
    >>> analyzer.detect()
    >>> footprint_path = analyzer.save_flash_footprints(out_dir, stem)
    >>> analyzer.group_centroid()
    >>> analyzer.group_trace(footprint_path)
    >>> analyzer.map("trace_corr")
    >>> analyzer.save_zones(out_dir, stem)
"""

## Modules
# Standard library imports
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import tifffile
from rich.console import Console
from scipy import ndimage
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from skimage.measure import find_contours

# Local imports
from classes.hierarchical_clustering import HierarchicalClustering
from classes.region_analyzer import PIXEL_SCALE
from functions.fit_hist import CROSSOVER_RATIO, fit_flash_background
from functions.flash_mask import flash_mask
from functions.footprint_traces import footprint_traces

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Flash mask + flashes (detect) ----------------------------------------
# (histogram bins / percentile range / fit live in functions/fit_hist.py -- shared with img_proc)
TH_SMALL_OBJ = 5000           # px: default size removal; SRE_analysis.py passes each recording's own
                              # th_small_obj_px from the permutation test (removal_suggestion.csv)
MAX_FLASH_FRAC = 0.8          # frame fraction: drop larger flashes (over-exposed first frames)
CONNECT_RADIUS = 25           # px: default same-frame join distance; SRE_analysis.py passes each recording's own
                              # (sqrt(th_small_obj / pi) in 5 px steps)

# --- Clustering (group) ---------------------------------------------------
CENTROID_CUT_PX = np.arange(5, 505, 5)  # px: centroid tree cut heights swept for the best mean silhouette
TRACE_CUT_1MR = np.round(np.arange(0.01, 0.91, 0.01), 2)  # 1 - r: trace tree cut heights swept the same way
TREE_UNITS = {"centroid": "px", "trace_corr": "1 - r"}  # unit of each tree's distance / cut height h
CUT_RULES = {"centroid": "acceleration", "trace_corr": "silhouette"}  # cut of each tree (both keep the silhouette
                                                                     # sweep + best-silhouette cut)

# --- Zones + stats (map) --------------------------------------------------
FRAME_RATE_HZ = 20            # imaging rate, converts frames -> seconds for recur_zone event stats
MIN_EVENTS_FOR_FREQ = 2       # summary freq / period stats use only recur_zones with at least this many events


class SREAnalyzer:
    """Detect -> group -> map spontaneous flashes into recur_zones for one stack.

    Results:
        bg_center, bg_sigma, threshold, mask, flashes, footprints     (detect)
        flashes["cluster_{method}"], detections (= flashes),
        clusterings[method] = {"clusters", "cut", "cut_rule", "silhouette_cut", "silhouette_cluster_ids",
                               "tree", "sweep"}    (group_centroid / group_trace)
        zone_method, zones (1 per cluster), zone_masks, zone_centroids, zone_stats   (map)
    """

    def __init__(self, stack: np.ndarray, obj: str = "10X", fps: float = FRAME_RATE_HZ,
                 sigma_ratio: float = CROSSOVER_RATIO, cuda_available: bool = False,
                 th_small_obj: int = TH_SMALL_OBJ, connect_radius: float = CONNECT_RADIUS) -> None:
        self.stack_f16 = np.asarray(stack, dtype=np.float16)  # no copy when already float16
        self.n_frames, self.height, self.width = stack.shape
        self.obj = obj
        self.um_per_px = 1.0 / PIXEL_SCALE[obj]
        self.fps = fps
        self.sigma_ratio = sigma_ratio
        self.cuda_available = cuda_available
        self.th_small_obj = th_small_obj
        self.connect_radius = connect_radius  # px: same-frame blobs this close are one flash
        self.clusterings: dict[str, dict] = {}  # method ("centroid" / "trace_corr") -> its clusters, cuts, tree, sweep

    def run(self) -> None:
        """detect -> group_centroid -> map("centroid")."""
        self.detect()
        self.group_centroid()
        self.map("centroid")

    # -----------------------------------------------------------------------
    # detect: background threshold -> flash mask -> flashes (close blobs joined)   (functions/flash_mask.py)
    # -----------------------------------------------------------------------

    def detect(self) -> None:
        """Background threshold -> cleaned per-frame flash mask -> per-frame flashes (+ footprints), same-frame blobs
        <= connect_radius apart joined."""
        with timed("detect threshold (histogram + fit)"):
            self.bg_center, self.bg_sigma = fit_flash_background(self.stack_f16, cuda_available=self.cuda_available)
            self.threshold = float(self.bg_center + self.sigma_ratio * self.bg_sigma)
        console.log(f"  threshold (peak + {self.sigma_ratio} sigma) = {self.threshold:.5f}")

        with timed("detect mask (open/close/fill/size)"):
            self.mask = flash_mask(self.stack_f16, self.threshold, self.th_small_obj, self.cuda_available)

        # per-frame flashes = 8-connected blobs, blobs <= connect_radius apart joined
        max_area = MAX_FLASH_FRAC * self.height * self.width
        with timed("detect label flashes"):
            self.flashes, self.footprints, giant = self.label_flashes(self.mask, max_area, self.connect_radius)
        self.detections = self.flashes  # zones / stats / maps read the flashes as "detections"
        console.log(f"  {len(self.flashes)} flashes (same-frame blobs <= {self.connect_radius} px apart joined)")
        if giant:
            console.log(f"  [yellow]{len(giant)} giant flash(s) dropped (> {MAX_FLASH_FRAC:.0%} of frame) "
                        f"in frames {[frame for frame, _ in giant]}[/yellow]")

    @staticmethod
    def label_flashes(mask: np.ndarray, max_area: float,
                      connect_radius: float) -> tuple[pd.DataFrame, list, list[tuple[int, int]]]:
        """Per frame: 8-connected blobs, blobs <= connect_radius apart joined, keep area <= max_area;
        joint_label unique over the recording.

        Returns (flashes table: frame (1-based), joint_label, centroid_y, centroid_x, area; footprints;
        dropped giants [(frame 1-based, area)]).
        """
        flash_props, footprints, dropped = [], [], []
        label_offset = 0
        for frame_id in range(mask.shape[0]):
            labeled_frame, n_frame_labels = ndimage.label(mask[frame_id], structure=np.ones((3, 3)))
            if n_frame_labels > 1:
                labeled_frame = SREAnalyzer.join_close_blobs(labeled_frame, n_frame_labels, connect_radius)
            for label in np.unique(labeled_frame[labeled_frame > 0]):
                coords = np.argwhere(labeled_frame == label)
                if len(coords) > max_area:
                    dropped.append((frame_id + 1, len(coords)))
                    continue
                footprints.append(coords)
                flash_props.append({"frame": frame_id + 1, "joint_label": int(label) + label_offset,
                                    "centroid_y": float(coords[:, 0].mean()), "centroid_x": float(coords[:, 1].mean()),
                                    "area": len(coords)})
            label_offset += n_frame_labels
        columns = ["frame", "joint_label", "centroid_y", "centroid_x", "area"]
        return pd.DataFrame(flash_props, columns=columns), footprints, dropped

    @staticmethod
    def join_close_blobs(labeled_frame: np.ndarray, n_labels: int, connect_radius: float) -> np.ndarray:
        """Same-frame blobs whose boundaries are <= connect_radius apart get one label (union-find, so A-B + B-C
        join)."""
        labels = list(range(1, n_labels + 1))
        boundaries = {label: np.argwhere((labeled_frame == label) & ~ndimage.binary_erosion(labeled_frame == label))
                      for label in labels}
        parent = {label: label for label in labels}

        def root(label: int) -> int:
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return label

        for i, label_a in enumerate(labels):
            tree = KDTree(boundaries[label_a])
            for label_b in labels[i + 1:]:
                if tree.query(boundaries[label_b], k=1)[0].min() <= connect_radius:
                    parent[root(label_a)] = root(label_b)
        remap = np.zeros(n_labels + 1, dtype=np.int32)
        for label in labels:
            remap[label] = root(label)
        return remap[labeled_frame]

    def save_flash_footprints(self, out_dir: Path, stem: str) -> Path:
        """Write footprints/{stem}_FLASH_FOOTPRINTS.npz: flash{label}_footprint = (n_px, 2) row / col of every
        flash (input of group_trace)."""
        path = out_dir / "footprints" / f"{stem}_FLASH_FOOTPRINTS.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f"flash{label}_footprint": footprint
                                     for label, footprint in zip(self.flashes["joint_label"], self.footprints,
                                                                 strict=True)})
        return path

    # -----------------------------------------------------------------------
    # group_centroid / group_trace: flashes -> clusters      (classes/hierarchical_clustering.py)
    # -----------------------------------------------------------------------

    def group_centroid(self) -> None:
        """Flash centroids -> Euclidean distance (px) -> HierarchicalClustering (tree cut at the acceleration elbow,
        CUT_RULES) -> flashes["cluster_centroid"] + clusterings["centroid"]."""
        n = len(self.flashes)
        with timed("group  centroid distance + tree + silhouette sweep"):
            distance = (squareform(pdist(self.flashes[["centroid_y", "centroid_x"]].to_numpy())) if n > 1
                        else np.zeros((n, n)))
            hc = HierarchicalClustering(distance, CENTROID_CUT_PX, cut_rule=CUT_RULES["centroid"])
            hc.run()
        self.store_clustering("centroid", hc)

    def group_trace(self, footprint_path: Path) -> None:
        """Flash footprints (save_flash_footprints npz) -> trace = mean z-score over the footprint, every frame ->
        1 - r (Pearson, zero lag) -> HierarchicalClustering (cut per CUT_RULES: best-silhouette h) ->
        flashes["cluster_trace_corr"] + clusterings["trace_corr"]."""
        n = len(self.flashes)
        with timed("group  trace_corr traces + r + tree + silhouette sweep"):
            with np.load(footprint_path) as npz:
                footprints = [npz[f"flash{label}_footprint"] for label in self.flashes["joint_label"]]
            distance = np.zeros((n, n))
            if n > 1:
                distance = 1 - np.corrcoef(footprint_traces(footprints, self.stack_f16, self.cuda_available))
                np.fill_diagonal(distance, 0)
            hc = HierarchicalClustering(distance, TRACE_CUT_1MR, cut_rule=CUT_RULES["trace_corr"])
            hc.run()
        self.store_clustering("trace_corr", hc)

    def store_clustering(self, method: str, hc: HierarchicalClustering) -> None:
        """flashes["cluster_{method}"] + clusterings[method] = {clusters, cut, cut_rule, silhouette_cut,
        silhouette_cluster_ids, tree, sweep}; log the cut (and the best-silhouette cut if another rule cut)."""
        cut, silhouette_cut, unit = hc.cut, hc.silhouette_cut, TREE_UNITS[method]
        column = f"cluster_{method}"
        self.flashes[column] = hc.cluster_ids
        clusters = (self.flashes.groupby(column)
                    .agg(labels=("joint_label", list), frames=("frame", sorted))
                    .reset_index().rename(columns={column: "group"}))
        self.clusterings[method] = {"clusters": clusters, "cut": cut, "cut_rule": hc.cut_rule,
                                    "silhouette_cut": silhouette_cut,
                                    "silhouette_cluster_ids": hc.silhouette_cluster_ids,
                                    "tree": hc.tree, "sweep": hc.sweep}
        console.log(f"  {len(self.flashes)} flashes -> {method} tree cut at {cut_h_name(hc.cut_rule)} "
                    f"{cut['h']:g} {unit}: {cut['n_clusters']} clusters, silhouette {cut['silhouette']:.3f}")
        if hc.cut_rule != "silhouette":
            console.log(f"    (best silhouette: h {silhouette_cut['h']:g} {unit}, {silhouette_cut['n_clusters']} "
                        f"clusters, silhouette {silhouette_cut['silhouette']:.3f})")

    # -----------------------------------------------------------------------
    # map: clusters of one method = recur_zones, masks, centroids, stats
    # -----------------------------------------------------------------------

    def map(self, method: str) -> None:
        """Every cluster of clusterings[method] = one recur_zone -> masks, centroids, stats (replaces the zones of
        the previous map call)."""
        shape = (self.height, self.width)
        clusters = self.clusterings[method]["clusters"]
        self.zone_method = method
        with timed(f"map    {method} tables + recur_zone stats"):
            self.zones = pd.DataFrame({
                "joint_labels": [list(labels) for labels in clusters["labels"]],
                "source": [f"{method} cluster #{i}" for i in clusters["group"]],
            })
            self.zones["zone_id"] = range(1, len(self.zones) + 1)  # 1-based, 0 stays background
            self.zone_masks = {row.zone_id: self.unit_mask(row.joint_labels, shape)
                               for row in self.zones.itertuples()}
            self.zone_centroids = self.find_zone_centroids()
            self.zone_stats = self.zone_stats_table()
        console.log(f"  {len(self.zone_masks)} {method} recur_zones")

    def unit_mask(self, units: list, shape: tuple[int, int]) -> np.ndarray:
        """Union of all flash footprints of the given units."""
        mask = np.zeros(shape, dtype=bool)
        for i in np.flatnonzero(self.detections["joint_label"].isin(units).to_numpy()):
            mask[self.footprints[i][:, 0], self.footprints[i][:, 1]] = True
        return mask

    def find_zone_centroids(self) -> dict[int, tuple[float, float]]:
        """zone id -> mean (y, x) of its member units' centroids."""
        track_xy = self.detections.groupby("joint_label")[["centroid_y", "centroid_x"]].mean()
        centroids: dict[int, tuple[float, float]] = {}
        for row in self.zones.itertuples():
            members = track_xy.loc[track_xy.index.intersection(row.joint_labels)]
            if not members.empty:
                centroids[row.zone_id] = (float(members["centroid_y"].mean()), float(members["centroid_x"].mean()))
        return centroids

    def zone_stats_table(self) -> pd.DataFrame:
        """One row per recur_zone: size (px, um^2) and event frequency/period."""
        columns = ["recur_zone_id", "source", "n_tracks", "area_px", "area_um2", "n_events", "mean_period_s",
                   "mean_freq_hz"]
        if self.zones.empty:
            return pd.DataFrame(columns=columns)

        stats = self.zones[["zone_id", "source"]].rename(columns={"zone_id": "recur_zone_id"})
        stats["n_tracks"] = self.zones["joint_labels"].apply(len)
        stats["area_px"] = stats["recur_zone_id"].map(
            lambda z: int(self.zone_masks[z].sum()) if z in self.zone_masks else 0)
        stats["area_um2"] = stats["area_px"] * self.um_per_px ** 2

        frames_by_label = self.detections.groupby("joint_label")["frame"].apply(np.array)
        event_stats = self.zones["joint_labels"].apply(lambda labels: self.zone_event_stats(
            np.concatenate([frames_by_label.get(label, np.array([], dtype=int)) for label in labels]), self.fps))
        stats[["n_events", "mean_period_s", "mean_freq_hz"]] = pd.DataFrame(event_stats.tolist(), index=stats.index)

        return stats.sort_values("recur_zone_id").reset_index(drop=True)

    @staticmethod
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

    # -----------------------------------------------------------------------
    # save: flash mask tif + flashes csv, zone xlsx + npz (save_flash_footprints: after detect)
    # -----------------------------------------------------------------------

    def save_flashes(self, out_dir: Path, stem: str, save_mask: bool = True, debug: bool = False) -> dict[str, Path]:
        """Write mask/{stem}_FLASH_MASK.tif (save_mask) + debug: {stem}_FLASHES.csv (cluster_centroid,
        cluster_trace_corr columns)."""
        paths = {}
        if save_mask:  # 1200 x 1024 x 1024 uint8 = 1.26 GB raw -> zlib
            (out_dir / "mask").mkdir(parents=True, exist_ok=True)
            paths["mask"] = out_dir / "mask" / f"{stem}_FLASH_MASK.tif"
            tifffile.imwrite(paths["mask"], self.mask.astype(np.uint8) * 255, compression="zlib")
        if debug:
            paths["flashes"] = out_dir / f"{stem}_FLASHES.csv"
            self.flashes.rename(columns={"joint_label": "flash_label"}).to_csv(paths["flashes"], index=False)
        return paths

    def save_zones(self, out_dir: Path, stem: str) -> dict[str, Path]:
        """Write the zones of the last map(method): {stem}_{METHOD}_ZONES.xlsx (counts, recur_zone_stats,
        recur_zones) + footprints/{stem}_{METHOD}_ZONES.npz (zone footprints + contours)."""
        tag = self.zone_method.upper()
        (out_dir / "footprints").mkdir(parents=True, exist_ok=True)
        paths = {
            "xlsx": out_dir / f"{stem}_{tag}_ZONES.xlsx",
            "npz":  out_dir / "footprints" / f"{stem}_{tag}_ZONES.npz",
        }

        frames_by_label = self.detections.groupby("joint_label")["frame"].apply(list)

        def active_frames(units: list) -> list[int]:
            return sorted(f for unit in units for f in frames_by_label[unit])

        recur_zones = self.zones.rename(columns={"zone_id": "recur_zone_id", "joint_labels": "track_ids"})
        recur_zones = recur_zones[["recur_zone_id", "source", "track_ids"]].copy()
        recur_zones["active_frames"] = recur_zones["track_ids"].apply(active_frames)
        with pd.ExcelWriter(paths["xlsx"]) as writer:
            self.cluster_counts().to_excel(writer, sheet_name="counts", index=False)
            self.zone_stats.to_excel(writer, sheet_name="recur_zone_stats", index=False)
            recur_zones.to_excel(writer, sheet_name="recur_zones", index=False)

        arrays: dict[str, np.ndarray] = {"background_threshold": np.array(self.threshold)}
        for zone_id, mask in self.zone_masks.items():
            arrays[f"zone{zone_id}_footprint"] = np.argwhere(mask)
            for k, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
                arrays[f"zone{zone_id}_contour{k}"] = contour
        np.savez_compressed(paths["npz"], **arrays)
        return paths

    def cluster_counts(self) -> pd.DataFrame:
        """Flashes, the tree cut and the recur_zones of the last map(method) (the xlsx counts sheet): one row per item."""
        method = self.zone_method
        clustering = self.clusterings[method]
        cut = clustering["cut"]
        rows = [
            ("flashes (8-connected blobs)", len(self.flashes)),
            (f"{method} tree: {cut_h_name(clustering['cut_rule'])} ({TREE_UNITS[method]})", cut["h"]),
            (f"{method} tree: mean silhouette", cut["silhouette"]),
            ("clusters K", cut["n_clusters"]),
            ("recur_zones (= clusters)", len(self.zones)),
            ("recur_zones with 1 flash", int((self.zones["joint_labels"].apply(len) == 1).sum())),
        ]
        return pd.DataFrame(rows, columns=["item", "value"])


# ===========================================================================
#
#   SHARED HELPERS
#
# ===========================================================================

# fn: timed, log one timed sub-step: '<label>  12.3s'.
# fn: cut_h_name, name of a cut height in logs / tables / plots: "best h" or "acceleration h".

def cut_h_name(cut_rule: str) -> str:
    """'best h' for the silhouette rule, else '{cut_rule} h' (e.g. 'acceleration h')."""
    return "best h" if cut_rule == "silhouette" else f"{cut_rule} h"


@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """Log one timed sub-step: '<label>  12.3s'."""
    t_start = time.time()
    yield
    console.log(f"[cyan]{label:<40}[/cyan] [bold]{time.time() - t_start:6.1f}s[/bold]")
