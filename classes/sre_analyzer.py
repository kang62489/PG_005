"""
Spontaneous ACh flash -> zone analysis for one deltaF/F0 stack.

  CLASS  SREAnalyzer             : detect() -> group() -> map() -> save()
  BLOCK  SHARED HELPERS          : timed, f16_lut, unit_mask
  BLOCK  FLASH MASK              : threshold -> open -> close -> fill holes -> drop small blobs   (detect)
  BLOCK  FLASHES                 : per-frame flashes = 8-connected blobs, <= connect_radius joined (detect)
  BLOCK  TRACES + CLUSTERING     : centroid tree cut at the best-silhouette h = the clusters      (group)
  BLOCK  ZONE STATS              : centroids, per-recur_zone event stats, counts sheet            (map, save)

recur_zone = recurring zone (non-recurring zones are not defined for the clustering yet).

Example:
    >>> analyzer = SREAnalyzer(stack, obj="10X")
    >>> analyzer.run()
    >>> analyzer.save(out_dir, stem)
"""

## Modules
# Standard library imports
import math
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cache
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import tifffile
from numba import cuda, njit, prange
from rich.console import Console
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from skimage.measure import find_contours

# Local imports
from classes.region_analyzer import PIXEL_SCALE
from functions.fit_hist import fit_flash_background

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Flash mask + flashes (detect) ----------------------------------------
# (histogram bins / percentile range / fit live in functions/fit_hist.py -- shared with img_proc)
CROSSOVER_RATIO = 2.0         # threshold = background peak + this many fitted sigmas
TH_SMALL_OBJ = 5000           # px: default size removal; SRE_analysis.py passes each recording's own
                              # th_small_obj_px from the permutation test (removal_suggestion.csv)
N_THREADS_FRAME_POOL = 16     # threads for the per-frame scipy label steps (fill holes, sizes)
MAX_FLASH_FRAC = 0.8          # frame fraction: drop larger flashes (over-exposed first frames)
CONNECT_RADIUS = 25           # px: default same-frame join distance; SRE_analysis.py passes each recording's own
                              # (sqrt(th_small_obj / pi) in 5 px steps)

# --- Clustering (group) ---------------------------------------------------
LINKAGE_METHOD = "average"    # both trees: cluster distance = mean of all cross pairs
CENTROID_CUT_PX = np.arange(5, 505, 5)  # px: centroid tree cut heights swept for the best mean silhouette
TRACE_THREADS = 256           # GPU threads per (flash, frame) block of the flash traces

# --- Zones + stats (map) --------------------------------------------------
FRAME_RATE_HZ = 20            # imaging rate, converts frames -> seconds for recur_zone event stats
MIN_EVENTS_FOR_FREQ = 2       # summary freq / period stats use only recur_zones with at least this many events


class SREAnalyzer:
    """Detect -> group -> map spontaneous flashes into recur_zones for one stack.

    Results after run():
        bg_center, bg_sigma, threshold, mask, flashes, footprints     (detect)
        flashes["cluster"], detections (= flashes), clusters, cluster_cut, centroid_tree, silhouette_sweep   (group)
        zones (1 per cluster), zone_masks, zone_centroids, zone_stats                           (map)
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

    def run(self) -> None:
        """detect -> group -> map."""
        self.detect()
        self.group()
        self.map()

    # -----------------------------------------------------------------------
    # detect: background threshold -> flash mask -> flashes (close blobs joined)   (BLOCKS FLASH MASK, FLASHES)
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
            self.flashes, self.footprints, giant = label_flashes(self.mask, max_area, self.connect_radius)
        console.log(f"  {len(self.flashes)} flashes (same-frame blobs <= {self.connect_radius} px apart joined)")
        if giant:
            console.log(f"  [yellow]{len(giant)} giant flash(s) dropped (> {MAX_FLASH_FRAC:.0%} of frame) "
                        f"in frames {[frame for frame, _ in giant]}[/yellow]")

    # -----------------------------------------------------------------------
    # group: flashes -> clusters                             (BLOCK TRACES + CLUSTERING)
    # -----------------------------------------------------------------------

    def group(self) -> None:
        """Flashes from detect() -> centroid tree cut at its best-silhouette h -> clusters (every cluster,
        1 flash or more)."""
        self.cluster_cut = {"centroid_h_px": np.nan, "n_clusters": 0, "centroid_silhouette": np.nan}
        self.centroid_tree = None
        self.silhouette_sweep = pd.DataFrame(columns=["cut_height", "n_clusters", "mean_silhouette"])
        self.flashes["cluster"] = pd.Series(dtype=int)
        self.detections = self.flashes  # zones / stats / maps read the flashes as "detections"
        if self.flashes.empty:
            console.log("[yellow]group: no flashes above threshold -- 0 recur_zones[/yellow]")
            self.clusters = pd.DataFrame(columns=["group", "labels", "frames"])
            return

        cluster_ids, self.cluster_cut, self.centroid_tree, self.silhouette_sweep = cluster_flashes(self.flashes)
        self.flashes["cluster"] = cluster_ids
        self.clusters = (self.flashes.groupby("cluster")
                         .agg(labels=("joint_label", list), frames=("frame", sorted))
                         .reset_index().rename(columns={"cluster": "group"}))
        cut = self.cluster_cut
        console.log(f"  {len(self.flashes)} flashes -> centroid tree cut at best h {cut['centroid_h_px']} px: "
                    f"{cut['n_clusters']} clusters, silhouette {cut['centroid_silhouette']:.3f}")

    # -----------------------------------------------------------------------
    # map: clusters = recur_zones, stats                   (BLOCK ZONE STATS)
    # -----------------------------------------------------------------------

    def map(self) -> None:
        """Every cluster from group() = one recur_zone -> masks, centroids, stats."""
        shape = (self.height, self.width)
        with timed("map    tables + recur_zone stats"):
            self.zones = pd.DataFrame({
                "joint_labels": [list(labels) for labels in self.clusters["labels"]],
                "source": [f"cluster #{i}" for i in self.clusters["group"]],
            })
            self.zones["zone_id"] = range(1, len(self.zones) + 1)  # 1-based, 0 stays background
            self.zone_masks = {row.zone_id: unit_mask(self.detections, self.footprints, row.joint_labels, shape)
                               for row in self.zones.itertuples()}
            self.zone_centroids = zone_centroids(self.zones, self.detections)
            self.zone_stats = zone_stats_table(self.zones, self.zone_masks, self.detections, self.fps, self.um_per_px)
        console.log(f"  {len(self.zone_masks)} recur_zones")

    # -----------------------------------------------------------------------
    # save: xlsx, npz, mask tif
    # -----------------------------------------------------------------------

    def save(self, out_dir: Path, stem: str, save_mask: bool = True, debug: bool = False) -> dict[str, Path]:
        """Write {stem}_ZONES.xlsx, footprints/{stem}_ZONES.npz (zone + flash footprints)
        (+ mask/{stem}_FLASH_MASK.tif, debug: {stem}_FLASHES.csv)."""
        (out_dir / "footprints").mkdir(parents=True, exist_ok=True)
        paths = {
            "xlsx": out_dir / f"{stem}_ZONES.xlsx",
            "npz":  out_dir / "footprints" / f"{stem}_ZONES.npz",
        }

        frames_by_label = self.detections.groupby("joint_label")["frame"].apply(list)

        def active_frames(units: list) -> list[int]:
            return sorted(f for unit in units for f in frames_by_label[unit])

        recur_zones = self.zones.rename(columns={"zone_id": "recur_zone_id", "joint_labels": "track_ids"})
        recur_zones = recur_zones[["recur_zone_id", "source", "track_ids"]].copy()
        recur_zones["active_frames"] = recur_zones["track_ids"].apply(active_frames)
        with pd.ExcelWriter(paths["xlsx"]) as writer:
            cluster_counts(self).to_excel(writer, sheet_name="counts", index=False)
            self.zone_stats.to_excel(writer, sheet_name="recur_zone_stats", index=False)
            recur_zones.to_excel(writer, sheet_name="recur_zones", index=False)

        if save_mask:  # 1200 x 1024 x 1024 uint8 = 1.26 GB raw -> zlib
            (out_dir / "mask").mkdir(exist_ok=True)
            paths["mask"] = out_dir / "mask" / f"{stem}_FLASH_MASK.tif"
            tifffile.imwrite(paths["mask"], self.mask.astype(np.uint8) * 255, compression="zlib")

        arrays: dict[str, np.ndarray] = {"background_threshold": np.array(self.threshold)}
        for zone_id, mask in self.zone_masks.items():
            arrays[f"zone{zone_id}_footprint"] = np.argwhere(mask)
            for k, contour in enumerate(find_contours(mask.astype(float), level=0.5)):
                arrays[f"zone{zone_id}_contour{k}"] = contour
        for label, footprint in zip(self.flashes["joint_label"], self.footprints, strict=True):
            arrays[f"flash{label}_footprint"] = footprint  # (n_px, 2) row / col: input of the flash traces
        np.savez_compressed(paths["npz"], **arrays)

        if debug:
            paths["flashes"] = out_dir / f"{stem}_FLASHES.csv"
            self.flashes.rename(columns={"joint_label": "flash_label"}).to_csv(paths["flashes"], index=False)

        return paths


# ===========================================================================
#
#   SHARED HELPERS
#
# ===========================================================================

# fn: timed, log one timed sub-step: '<label>  12.3s'.
# fn: f16_lut, float32 value of every float16 bit pattern (numba kernels decode float16 through it).
# fn: unit_mask, union of all flash footprints of some units.

@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """Log one timed sub-step: '<label>  12.3s'."""
    t_start = time.time()
    yield
    console.log(f"[cyan]{label:<40}[/cyan] [bold]{time.time() - t_start:6.1f}s[/bold]")


@cache
def f16_lut() -> np.ndarray:
    """float32 value of every float16 bit pattern (index = uint16 code)."""
    return np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)


def unit_mask(detections: pd.DataFrame, footprints: list, units: list, shape: tuple[int, int]) -> np.ndarray:
    """Union of all flash footprints of the given units."""
    mask = np.zeros(shape, dtype=bool)
    for i in np.flatnonzero(detections["joint_label"].isin(units).to_numpy()):
        mask[footprints[i][:, 0], footprints[i][:, 1]] = True
    return mask


# ===========================================================================
#
#   FLASH MASK related functions
#
#   The threshold itself: functions/fit_hist.fit_flash_background (shared with img_proc).
#   The stack is float16; numba can't load float16, so kernels read the raw uint16 codes and
#   decode them through a 65536-entry lookup table (exact, identical to numpy's float16->float32).
#   Results match the scipy chain.
#
# ===========================================================================

# fn: flash_mask, threshold -> open -> close -> fill holes -> drop small blobs (detect()).
# fn: flash_sizes / mask_flash_sizes, same mask, only the flash sizes (permutation test).
# fn: _threshold_open_close, threshold -> open -> close on the GPU / numba CPU.

# --- CPU (numba, one frame per thread) -------------------------------------

# scipy semantics: border_value=0 -> erosion treats outside-the-frame as False,
# dilation ignores outside-the-frame pixels.

@njit(inline="always")
def _erode_px(src: np.ndarray, y: int, x: int, h: int, w: int) -> bool:
    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
        return False
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if not src[y + dy, x + dx]:
                return False
    return True


@njit(inline="always")
def _dilate_px(src: np.ndarray, y: int, x: int, h: int, w: int) -> bool:
    for dy in range(-1, 2):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        for dx in range(-1, 2):
            xx = x + dx
            if 0 <= xx < w and src[yy, xx]:
                return True
    return False


@njit(parallel=True)
def _cpu_threshold_open_close(codes: np.ndarray, lut: np.ndarray, thr: np.float32) -> np.ndarray:
    n_frames, h, w = codes.shape
    out = np.empty((n_frames, h, w), dtype=np.bool_)
    for t in prange(n_frames):
        a = np.empty((h, w), dtype=np.bool_)
        b = np.empty((h, w), dtype=np.bool_)
        for y in range(h):
            for x in range(w):
                a[y, x] = lut[codes[t, y, x]] > thr
        for y in range(h):  # opening = erode ...
            for x in range(w):
                b[y, x] = _erode_px(a, y, x, h, w)
        for y in range(h):  # ... then dilate
            for x in range(w):
                a[y, x] = _dilate_px(b, y, x, h, w)
        for y in range(h):  # closing = dilate ...
            for x in range(w):
                b[y, x] = _dilate_px(a, y, x, h, w)
        for y in range(h):  # ... then erode
            for x in range(w):
                out[t, y, x] = _erode_px(b, y, x, h, w)
    return out


# --- GPU (one thread per pixel) --------------------------------------------

@cuda.jit
def _gpu_threshold_kernel(codes: np.ndarray, lut: np.ndarray, thr: np.float32, out: np.ndarray) -> None:
    x, y, t = cuda.grid(3)
    if t < codes.shape[0] and y < codes.shape[1] and x < codes.shape[2]:
        out[t, y, x] = 1 if lut[codes[t, y, x]] > thr else 0


@cuda.jit
def _gpu_erode_kernel(src: np.ndarray, out: np.ndarray) -> None:
    x, y, t = cuda.grid(3)
    n_frames, h, w = src.shape
    if t >= n_frames or y >= h or x >= w:
        return
    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
        out[t, y, x] = 0
        return
    val = 1
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if src[t, y + dy, x + dx] == 0:
                val = 0
    out[t, y, x] = val


@cuda.jit
def _gpu_dilate_kernel(src: np.ndarray, out: np.ndarray) -> None:
    x, y, t = cuda.grid(3)
    n_frames, h, w = src.shape
    if t >= n_frames or y >= h or x >= w:
        return
    val = 0
    for dy in range(-1, 2):
        yy = y + dy
        if 0 <= yy < h:
            for dx in range(-1, 2):
                xx = x + dx
                if 0 <= xx < w and src[t, yy, xx] != 0:
                    val = 1
    out[t, y, x] = val


def _gpu_threshold_open_close(codes: np.ndarray, lut: np.ndarray, thr: np.float32) -> np.ndarray:
    n_frames, h, w = codes.shape
    threads = (32, 8, 1)
    blocks = (math.ceil(w / 32), math.ceil(h / 8), n_frames)

    d_codes = cuda.to_device(codes)
    d_lut = cuda.to_device(lut)
    d_a = cuda.device_array((n_frames, h, w), dtype=np.uint8)
    d_b = cuda.device_array((n_frames, h, w), dtype=np.uint8)

    _gpu_threshold_kernel[blocks, threads](d_codes, d_lut, thr, d_a)
    del d_codes
    _gpu_erode_kernel[blocks, threads](d_a, d_b)   # opening
    _gpu_dilate_kernel[blocks, threads](d_b, d_a)
    _gpu_dilate_kernel[blocks, threads](d_a, d_b)  # closing
    _gpu_erode_kernel[blocks, threads](d_b, d_a)
    cuda.synchronize()
    return d_a.copy_to_host().view(np.bool_)


# --- Fill holes + drop small blobs (scipy labels, per frame) ---------------

_CROSS = ndimage.generate_binary_structure(2, 1)  # 4-connectivity, scipy's default for both steps


def _fill_holes(frame: np.ndarray) -> np.ndarray:
    """fill holes = background 4-regions not touching the edge become foreground."""
    bg_labels, _ = ndimage.label(~frame, structure=_CROSS)
    edge = np.unique(np.concatenate([bg_labels[0], bg_labels[-1], bg_labels[:, 0], bg_labels[:, -1]]))
    is_edge_bg = np.zeros(bg_labels.max() + 1, dtype=bool)
    is_edge_bg[edge] = True
    return frame | ~is_edge_bg[bg_labels]  # label 0 = foreground, never an edge background label


def _fill_and_filter_frame(frame: np.ndarray, th_small_obj: int) -> np.ndarray:
    """fill holes -> keep blobs >= th_small_obj px."""
    filled = _fill_holes(frame)
    labels, n = ndimage.label(filled, structure=_CROSS)
    if n == 0:
        return filled
    sizes = np.bincount(labels.ravel())
    keep = sizes >= th_small_obj
    keep[0] = False
    return keep[labels]


# --- Dispatch --------------------------------------------------------------

def _threshold_open_close(stack_f16: np.ndarray, threshold: float, cuda_available: bool) -> np.ndarray:
    """Threshold -> 3x3 opening -> 3x3 closing (GPU / numba), before fill holes."""
    codes = stack_f16.view(np.uint16)
    thr = np.float32(np.float16(threshold))  # numpy compares a float16 array against a float16-rounded scalar
    if cuda_available:
        return _gpu_threshold_open_close(codes, f16_lut(), thr)
    return _cpu_threshold_open_close(codes, f16_lut(), thr)


def flash_mask(stack_f16: np.ndarray, threshold: float, th_small_obj: int, cuda_available: bool = False) -> np.ndarray:
    """Threshold -> open -> close -> fill holes -> drop small blobs, same result as the scipy chain."""
    mask = _threshold_open_close(stack_f16, threshold, cuda_available)
    with ThreadPoolExecutor(N_THREADS_FRAME_POOL) as pool:
        frames = list(pool.map(lambda frame: _fill_and_filter_frame(frame, th_small_obj), mask))
    return np.stack(frames)


def _frame_flash_sizes(frame: np.ndarray) -> np.ndarray:
    """Fill holes -> size (px) of every flash of one frame, labelled once."""
    filled = _fill_holes(frame)
    labels, n = ndimage.label(filled, structure=_CROSS)
    return np.bincount(labels[filled])[1:] if n else np.zeros(0, dtype=np.int64)  # foreground px only


def mask_flash_sizes(opened_closed: np.ndarray) -> np.ndarray:
    """Fill holes + size (px) of every 4-connected flash of a threshold -> open -> close mask, frame by frame."""
    with ThreadPoolExecutor(N_THREADS_FRAME_POOL) as pool:
        sizes = list(pool.map(_frame_flash_sizes, opened_closed))
    return np.concatenate(sizes)


def flash_sizes(stack_f16: np.ndarray, threshold: float, cuda_available: bool = False) -> np.ndarray:
    """Size (px) of every 4-connected flash of flash_mask(th_small_obj = 0), each frame labelled only once."""
    return mask_flash_sizes(_threshold_open_close(stack_f16, threshold, cuda_available))


# ===========================================================================
#
#   FLASHES related functions
#
# ===========================================================================

# fn: label_flashes, per frame: 8-connected blobs, blobs <= connect_radius apart joined = flashes, drop giants
#     (detect()).
# fn: join_close_blobs, same-frame blobs <= connect_radius apart -> one label.

def join_close_blobs(labeled_frame: np.ndarray, n_labels: int, connect_radius: float) -> np.ndarray:
    """Same-frame blobs whose boundaries are <= connect_radius apart get one label (union-find, so A-B + B-C join)."""
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
            labeled_frame = join_close_blobs(labeled_frame, n_frame_labels, connect_radius)
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


# ===========================================================================
#
#   TRACES + CLUSTERING related functions
#
# ===========================================================================

# fn: footprint_traces, mean of every flash footprint in every frame (CPU numba / GPU).
# fn: cluster_flashes, centroid tree cut at the best mean silhouette h = the clusters (group()).
# fn: mean_silhouette / best_silhouette_cut, mean silhouette of a cut; best cut of an h sweep.

# --- Flash traces (CPU numba / GPU) ----------------------------------------

@njit(parallel=True)
def _cpu_footprint_traces(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                          starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """CPU: one frame per thread."""
    n_frames = codes_flat.shape[0]
    n_det = starts.shape[0]
    means = np.empty((n_det, n_frames), dtype=np.float32)
    for t in prange(n_frames):
        for d in range(n_det):
            s = np.float32(0.0)
            for k in range(starts[d], starts[d] + counts[d]):
                s += lut[codes_flat[t, linear_idx[k]]]
            means[d, t] = s / np.float32(counts[d])
    return means


@cuda.jit
def _gpu_footprint_traces_kernel(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                                 starts: np.ndarray, counts: np.ndarray, means: np.ndarray) -> None:
    """GPU: one block per detection x frame, shared-memory reduction."""
    d = cuda.blockIdx.x
    t = cuda.blockIdx.y
    tid = cuda.threadIdx.x
    partial = cuda.shared.array(TRACE_THREADS, dtype=np.float32)

    s = np.float32(0.0)
    start = starts[d]
    for k in range(start + tid, start + counts[d], TRACE_THREADS):
        s += lut[codes_flat[t, linear_idx[k]]]
    partial[tid] = s
    cuda.syncthreads()

    stride = TRACE_THREADS // 2
    while stride > 0:
        if tid < stride:
            partial[tid] += partial[tid + stride]
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        means[d, t] = partial[0] / np.float32(counts[d])


def _gpu_footprint_traces(codes_flat: np.ndarray, lut: np.ndarray, linear_idx: np.ndarray,
                          starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n_frames, n_det = codes_flat.shape[0], starts.shape[0]
    d_means = cuda.device_array((n_det, n_frames), dtype=np.float32)
    _gpu_footprint_traces_kernel[(n_det, n_frames), TRACE_THREADS](
        cuda.to_device(codes_flat), cuda.to_device(lut), cuda.to_device(linear_idx),
        cuda.to_device(starts), cuda.to_device(counts), d_means,
    )
    cuda.synchronize()
    return d_means.copy_to_host()


def footprint_traces(footprints: list, stack_f16: np.ndarray, cuda_available: bool = False) -> np.ndarray:
    """(n_detections, n_frames) float16: mean over each detection's own footprint, every frame."""
    n_frames, _, width = stack_f16.shape
    counts = np.array([len(coords) for coords in footprints], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    all_coords = np.concatenate(footprints, axis=0)
    linear_idx = (all_coords[:, 0] * width + all_coords[:, 1]).astype(np.int64)

    codes_flat = stack_f16.view(np.uint16).reshape(n_frames, -1)
    lut = f16_lut()
    if cuda_available:
        means = _gpu_footprint_traces(codes_flat, lut, linear_idx, starts, counts)
    else:
        means = _cpu_footprint_traces(codes_flat, lut, linear_idx, starts, counts)
    return means.astype(np.float16)


# --- Mean silhouette ------------------------------------------------------

def mean_silhouette(distance: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Mean silhouette over all flashes; a flash alone in its cluster scores 0 (= sklearn silhouette_score)."""
    clusters = np.unique(cluster_ids)
    one_hot = (cluster_ids[:, None] == clusters[None, :]).astype(float)  # flash x cluster
    sizes = one_hot.sum(axis=0)
    sum_to_cluster = distance @ one_hot  # flash x cluster: summed distance to each cluster
    own = np.searchsorted(clusters, cluster_ids)
    own_size = sizes[own]
    a = sum_to_cluster[np.arange(len(cluster_ids)), own] / np.maximum(own_size - 1, 1)
    mean_to_cluster = sum_to_cluster / sizes
    mean_to_cluster[np.arange(len(cluster_ids)), own] = np.inf
    b = mean_to_cluster.min(axis=1)
    scores = np.where(own_size > 1, (b - a) / np.maximum(a, b), 0.0)
    return float(scores.mean())


def best_silhouette_cut(tree: np.ndarray, distance: np.ndarray,
                        cut_heights: np.ndarray) -> tuple[float, int, float, pd.DataFrame]:
    """(h, clusters, mean silhouette, sweep) of the best cut; the first h of the maximum (start of a plateau).

    sweep: cut_height, n_clusters, mean_silhouette per h. Only cuts with 1 < clusters < flashes have a silhouette;
    none -> (NaN, 1, NaN, sweep).
    """
    best_h, best_k, best_score = np.nan, 1, -np.inf
    rows = []
    for cut_height in cut_heights:
        cluster_ids = fcluster(tree, t=cut_height, criterion="distance")
        score = mean_silhouette(distance, cluster_ids) if 1 < cluster_ids.max() < len(cluster_ids) else np.nan
        rows.append({"cut_height": float(cut_height), "n_clusters": int(cluster_ids.max()), "mean_silhouette": score})
        if score > best_score:
            best_h, best_k, best_score = float(cut_height), int(cluster_ids.max()), score
    return best_h, best_k, (best_score if np.isfinite(best_score) else np.nan), pd.DataFrame(rows)


# --- Centroid tree cut at the best h -----------------------------------------

def _renumber(raw_group_ids: np.ndarray) -> list[int]:
    """Renumber cluster ids 0, 1, 2, ... in first-seen order."""
    group_id_map = {raw: new for new, raw in enumerate(pd.unique(raw_group_ids))}
    return [group_id_map[raw] for raw in raw_group_ids]


def cluster_flashes(flashes: pd.DataFrame) -> tuple[np.ndarray, dict, np.ndarray | None, pd.DataFrame]:
    """Centroid tree (Euclidean px, LINKAGE_METHOD linkage) cut at its best mean silhouette h -> the clusters.

    Returns (cluster id per flash, 0..K-1 in first-seen order; cut: centroid_h_px, n_clusters, centroid_silhouette;
    centroid tree (None if < 2 flashes); silhouette sweep (empty if < 3 flashes)).
    """
    n = len(flashes)
    sweep = pd.DataFrame(columns=["cut_height", "n_clusters", "mean_silhouette"])
    with timed("group  centroid tree + silhouette sweep"):
        centroid_distance = squareform(pdist(flashes[["centroid_y", "centroid_x"]].to_numpy()))
        centroid_tree = linkage(squareform(centroid_distance, checks=False), method=LINKAGE_METHOD) if n > 1 else None
        centroid_h, centroid_score = np.nan, np.nan
        if n > 2:
            centroid_h, _, centroid_score, sweep = best_silhouette_cut(centroid_tree, centroid_distance,
                                                                       CENTROID_CUT_PX)
        if np.isnan(centroid_score):
            raw_ids = np.ones(n, dtype=int)
            console.log(f"  [yellow]{n} flash(es): no cut with 1 < clusters < flashes -- all flashes in one cluster"
                        f"[/yellow]")
        else:
            raw_ids = fcluster(centroid_tree, t=centroid_h, criterion="distance")

    cut = {"centroid_h_px": centroid_h, "n_clusters": int(raw_ids.max()), "centroid_silhouette": centroid_score}
    return np.array(_renumber(raw_ids)), cut, centroid_tree, sweep


# ===========================================================================
#
#   ZONE STATS related functions
#
# ===========================================================================

# fn: zone_centroids, zone id -> mean (y, x) of its units (map()).
# fn: zone_stats_table / zone_event_stats, size + event frequency / period per recur_zone (map()).
# fn: cluster_counts, flashes, tree cuts and recur_zones, xlsx counts sheet (save()).

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


def cluster_counts(analyzer: SREAnalyzer) -> pd.DataFrame:
    """Flashes, the tree cuts and the recur_zones (the xlsx counts sheet): one row per item."""
    cut = analyzer.cluster_cut
    rows = [
        ("flashes (8-connected blobs)", len(analyzer.flashes)),
        ("centroid tree: best h (px)", cut["centroid_h_px"]),
        ("centroid tree: mean silhouette", cut["centroid_silhouette"]),
        ("clusters K", cut["n_clusters"]),
        ("recur_zones (= clusters)", len(analyzer.zones)),
        ("recur_zones with 1 flash", int((analyzer.zones["joint_labels"].apply(len) == 1).sum())),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])
