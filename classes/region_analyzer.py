"""
Region analysis for categorized images.

For the spike frame, finds the largest bright connected component, then
combines every dim-category pixel in the frame into one dim region.
"""

import numpy as np
from sklearn.cluster import DBSCAN

# Category constants
CATEGORY_BACKGROUND = 0
CATEGORY_DIM = 1
CATEGORY_BRIGHT = 2

# Pixel scaling constants (pixel/um)
PIXEL_SCALE = {
    "10X": 0.75,
    "40X": 3.0,
    "60X": 4.5,
}

EPS_UM = 10.0  # inter-varicosity gap in um (tunable)
MIN_DENSITY_FRAC = 0.1  # min_samples = this fraction of the eps-circle area
MIN_CLUSTER_FRACTION = 0.05  # keep DBSCAN clusters covering at least this fraction of non-background pixels

AREA_PCT_SIGMA_MULT = 5.0  # baseline_mean + this many std devs = "significant" B+D% elevation

SATURATION_AREA_PCT = 15.0  # critical frame B+D% at/above this skips DBSCAN entirely (too dense, blows up memory)


class RegionAnalyzer:
    """
    Find DBSCAN clusters of non-background pixels on the critical frame.

    Picks the spike frame or spike+1 as the critical frame (whichever clears
    the B+D% significance threshold), clusters its non-background pixels with
    DBSCAN, and drops undersized clusters. Each kept cluster gets a centroid,
    an enclosing-circle radius (R), and a z-score trace across the segment
    (inner/outer ring split for a single cluster, whole-cluster trace when
    there are multiple).

    If the critical frame's B+D% is at/above SATURATION_AREA_PCT, DBSCAN is
    skipped (self.saturated = True) -- a dense enough point cloud makes
    sklearn's neighbor-graph construction blow up memory. Every non-background
    pixel is instead treated as one big cluster (ring-split like any other
    single-cluster case), since at that density it's one region, not discrete
    release sites.

    Result dict per cluster (from get_results()):
        centroid : (row, col) in pixels
        R_px     : enclosing-circle radius in pixels
        R_um     : enclosing-circle radius in µm

    Example:
        >>> categorizer = SpatialCategorizer.morphological()
        >>> categorizer.fit(image_segment, spike_frame_idx=spike_frame_idx)
        >>> analyzer = RegionAnalyzer(categorizer.categorized_frames, med_stack, spike_frame_idx, obj="10X")
        >>> results = analyzer.get_results()
    """

    def __init__(self, cat_stack: np.ndarray, med_stack: np.ndarray, spike_frame_idx: int, obj: str = "10X") -> None:
        """
        Analyze the categorized stack immediately on construction.

        Args:
            cat_stack: 3D array (frames, height, width) of categorized frames
                (0=background, 1=dim, 2=bright).
            med_stack: 3D array (frames, height, width) of z-scored median frames,
                same shape as cat_stack.
            spike_frame_idx: index of the spike frame within the segment.
            obj: Objective magnification ("10X", "40X", "60X")
        """
        if obj not in PIXEL_SCALE:
            msg = f"Unknown objective: {obj}. Choose from {list(PIXEL_SCALE.keys())}"
            raise ValueError(msg)

        self.obj = obj
        self.pixel_per_um = PIXEL_SCALE[obj]
        self.um_per_pixel = 1.0 / self.pixel_per_um
        self.spike_frame_idx = spike_frame_idx
        self._frame_total_px = cat_stack.shape[1] * cat_stack.shape[2]

        self.area_pct = compute_area_pct(cat_stack)
        self.critical_frame_idx = pick_critical_frame(self.area_pct, spike_frame_idx)
        self.saturated = self.area_pct[self.critical_frame_idx] >= SATURATION_AREA_PCT

        if self.saturated:
            # Too many non-background pixels for DBSCAN's neighbor-graph construction
            # to stay bounded -- skip DBSCAN and treat every non-background pixel as
            # one big cluster instead (this dense, it's one region, not discrete sites).
            frame = cat_stack[self.critical_frame_idx]
            mask = frame > CATEGORY_BACKGROUND
            coords = np.argwhere(mask)
            self.label_frame = np.where(mask, 0, -2).astype(int)
            self.centroids = [(float(coords[:, 0].mean()), float(coords[:, 1].mean()))]
            self.n_raw_clusters = 1
        else:
            eps_px, min_samples = eps_and_min_samples(obj)
            self.label_frame, self.centroids, self.n_raw_clusters = _run_cluster_seeker(
                cat_stack[self.critical_frame_idx], eps_px, min_samples
            )

        self.clusters = []
        if len(self.centroids) == 1:
            centroid = self.centroids[0]
            inner_trace, outer_trace, R, inner_mask, outer_mask = compute_ring_traces(
                self.label_frame, centroid, med_stack, 0
            )
            self.clusters.append({
                "centroid":    centroid,
                "R_px":        R,
                "R_um":        self._px_to_um(R),
                "inner_trace": inner_trace,
                "outer_trace": outer_trace,
                "inner_mask":  inner_mask,
                "outer_mask":  outer_mask,
            })
        elif len(self.centroids) > 1:
            for cluster_k, centroid in enumerate(self.centroids):
                trace, R, mask = compute_cluster_trace(self.label_frame, centroid, med_stack, cluster_k)
                self.clusters.append({
                    "centroid": centroid,
                    "R_px":     R,
                    "R_um":     self._px_to_um(R),
                    "trace":    trace,
                    "mask":     mask,
                })

    # ── Unit conversion helpers ────────────────────────────────────────────────

    def _px_to_um(self, pixels: float) -> float:
        return pixels * self.um_per_pixel

    def _area_to_um2(self, area_px: float) -> float:
        return area_px * (self.um_per_pixel ** 2)

    # ── Result accessors ──────────────────────────────────────────────────────

    def get_results(self) -> dict:
        """Get per-cluster region results for the critical frame."""
        critical_frame_area_pct = float(self.area_pct[self.critical_frame_idx])
        critical_frame_area_px = critical_frame_area_pct / 100.0 * self._frame_total_px
        return {
            "critical_frame_idx":      self.critical_frame_idx,
            "critical_frame_offset":   self.critical_frame_idx - self.spike_frame_idx,
            "critical_frame_area_pct": critical_frame_area_pct,
            "critical_frame_area_um2": self._area_to_um2(critical_frame_area_px),
            "n_clusters":               len(self.clusters),
            "clusters": [
                {"centroid": c["centroid"], "R_px": c["R_px"], "R_um": c["R_um"]}
                for c in self.clusters
            ],
        }

    def get_summary(self) -> dict:
        """Summary for the critical frame."""
        return {
            "obj":         self.obj,
            "n_clusters":  len(self.clusters),
            "has_region":  len(self.clusters) > 0,
            "saturated":   self.saturated,
        }

    def get_temporal_traces(self) -> list[dict]:
        """Per-cluster z-score traces computed in __init__.

        1 cluster -> inner/outer ring split (spread within the one release site).
        >1 clusters -> one whole-cluster trace per cluster (no ring split), so
        cluster-to-cluster peak timing can be compared directly.

        Returns:
            List of dicts, one per cluster (same order as self.clusters):
            {"inner_trace":, "outer_trace":} for 1 cluster, or {"trace":} for >1.
        """
        if len(self.clusters) == 1:
            c = self.clusters[0]
            return [{"inner_trace": c["inner_trace"], "outer_trace": c["outer_trace"]}]
        return [{"trace": c["trace"]} for c in self.clusters]

    def get_peak_latency_ms(self, frame_duration_ms: float) -> float | None:
        """Peak-timing latency in milliseconds; meaning depends on cluster count.

        0 clusters -> None (no event).
        1 cluster -> outer ring peak minus inner ring peak (spread within the
            one release site).
        >1 clusters -> max cluster peak time minus min cluster peak time
            (largest asynchrony between separate release sites).

        Args:
            frame_duration_ms: milliseconds per frame.

        Returns:
            Latency in ms, or None if it can't be computed (no clusters, or
            fewer than the required number of located peaks).
        """
        if not self.clusters:
            return None

        if len(self.clusters) == 1:
            c = self.clusters[0]
            inner_peak_rel = _peak_offset_from_spike(c["inner_trace"], self.spike_frame_idx)
            outer_peak_rel = _peak_offset_from_spike(c["outer_trace"], self.spike_frame_idx)
            if inner_peak_rel is None or outer_peak_rel is None:
                return None
            return (outer_peak_rel - inner_peak_rel) * frame_duration_ms

        peak_rels = [
            p
            for p in (_peak_offset_from_spike(c["trace"], self.spike_frame_idx) for c in self.clusters)
            if p is not None
        ]
        if len(peak_rels) < 2:
            return None
        return (max(peak_rels) - min(peak_rels)) * frame_duration_ms

    def get_export_data(self) -> dict:
        """Data for export (contours and internal fields stripped by exporter)."""
        return {
            "objective":      self.obj,
            "um_per_pixel":   self.um_per_pixel,
            "region_summary": self.get_summary(),
            "region_data":    self.get_results(),
        }


# ── Module-level helpers (used internally by RegionAnalyzer) ───────────────────


def compute_area_pct(stack: np.ndarray) -> np.ndarray:
    """Area percentage (B+D%) of non-background pixels per frame.

    This is the B+D% detection-criterion signal: a real ACh event shows a
    clear elevation at the spike frame (or spike_frame+1) vs the baseline
    frame before it.

    Args:
        stack: 3D array (frames, height, width) of categorized frames.

    Returns:
        1D array (n_frames,) of B+D% per frame.
    """
    total_px = stack.shape[1] * stack.shape[2]
    return np.count_nonzero(stack > CATEGORY_BACKGROUND, axis=(1, 2)) / total_px * 100


def pick_critical_frame(area_pct: np.ndarray, spike_frame_idx: int) -> int:
    """Pick spike or spike+1 for clustering, biased toward the spike frame.

    Compares each candidate frame's B+D% against a baseline-derived
    significance threshold (mean + AREA_PCT_SIGMA_MULT * std, over every frame
    before the spike frame) instead of simply picking whichever is higher --
    avoids flipping to spike+1 on frame-to-frame noise. Only switches to
    spike+1 when the spike frame itself isn't significantly elevated but
    spike+1 is (delayed-signal case).

    Args:
        area_pct: 1D array (n_frames,), from compute_area_pct().
        spike_frame_idx: index of the spike frame within the segment.

    Returns:
        Index of the frame to run clustering on (spike_frame_idx or spike_frame_idx + 1).
    """
    baseline = area_pct[:spike_frame_idx]
    threshold = baseline.mean() + AREA_PCT_SIGMA_MULT * baseline.std()

    if area_pct[spike_frame_idx] >= threshold:
        return spike_frame_idx

    next_idx = spike_frame_idx + 1
    if next_idx < len(area_pct) and area_pct[next_idx] >= threshold:
        return next_idx

    return spike_frame_idx


def eps_and_min_samples(obj: str) -> tuple[int, int]:
    """Convert EPS_UM to pixels for this objective and derive min_samples.

    Args:
        obj: Objective magnification, must be a key of PIXEL_SCALE.

    Returns:
        (eps_px, min_samples) for DBSCAN.
    """
    px_per_um = PIXEL_SCALE[obj]
    eps_px = int(EPS_UM * px_per_um)
    min_samples = max(1, int(MIN_DENSITY_FRAC * np.pi * eps_px**2))
    return eps_px, min_samples


def _run_cluster_seeker(frame: np.ndarray, eps_px: int, min_samples: int) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """Cluster non-background pixels with DBSCAN, then drop undersized clusters.

    Args:
        frame: 2D array (0=background, 1=dim, 2=bright) for a single frame.
        eps_px: DBSCAN neighborhood radius in pixels, from eps_and_min_samples().
        min_samples: DBSCAN minimum samples per cluster, from eps_and_min_samples().

    Returns:
        label_frame: (H, W) array; -2 = background, -1 = noise/undersized cluster,
            0..N-1 = kept clusters, largest first.
        centroids: (row, col) per kept cluster, same order as label_frame's indices.
        n_raw_clusters: number of clusters DBSCAN found before the size filter.
    """
    coords = np.argwhere(frame > CATEGORY_BACKGROUND)
    label_frame = np.full(frame.shape, -2, dtype=int)
    if coords.shape[0] == 0:
        return label_frame, [], 0

    raw_labels = DBSCAN(eps=eps_px, min_samples=min_samples).fit_predict(coords)

    total_non_bg = coords.shape[0]
    unique, counts = np.unique(raw_labels[raw_labels >= 0], return_counts=True)
    n_raw_clusters = len(unique)

    kept = []
    for cluster_label, pixel_count in zip(unique.tolist(), counts.tolist(), strict=True):
        if pixel_count / total_non_bg >= MIN_CLUSTER_FRACTION:
            kept.append((cluster_label, pixel_count))
    kept.sort(key=lambda pair: pair[1], reverse=True)  # key= is the sort-by value here (unrelated to dict keys), pair[1] = pixel_count

    remapped = np.full_like(raw_labels, -1)
    for new_cluster_label, (old_cluster_label, _) in enumerate(kept):
        remapped[raw_labels == old_cluster_label] = new_cluster_label
    label_frame[coords[:, 0], coords[:, 1]] = remapped

    centroids = [
        (float(coords[remapped == new_cluster_label, 0].mean()), float(coords[remapped == new_cluster_label, 1].mean()))
        for new_cluster_label in range(len(kept))
    ]

    return label_frame, centroids, n_raw_clusters


def _resolve_R(dists: np.ndarray, centroid: tuple[float, float], frame_shape: tuple[int, int]) -> float:
    """Enclosing-circle radius, capped so the circle never extends past the frame.

    The farthest-pixel distance alone can draw a circle that overshoots the
    frame edge whenever the centroid isn't near the image center -- it only
    guarantees the circle contains every cluster pixel, not that it stays
    inside the frame. Capping at the centroid's distance to the nearest frame
    edge (the largest circle around the centroid that still fits inside the
    frame) keeps the drawn circle within bounds.

    Args:
        dists: (N,) centroid-to-pixel distances for the cluster's pixels.
        centroid: (row, col) of the cluster.
        frame_shape: (height, width) of the frame.

    Returns:
        R in pixels.
    """
    height, width = frame_shape
    row_c, col_c = centroid
    edge_dist = min(row_c, height - 1 - row_c, col_c, width - 1 - col_c)
    return float(min(dists.max(), edge_dist))


def compute_ring_traces(
    label_frame: np.ndarray,
    centroid: tuple[float, float],
    med_stack: np.ndarray,
    cluster_k: int,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Inner/outer ring z-score traces for one DBSCAN cluster.

    R is the enclosing-circle radius (max centroid-to-pixel distance among the
    cluster's pixels), capped at the centroid's distance to the nearest frame
    edge so the drawn circle never extends past the frame (see _resolve_R).
    Cluster pixels are split into two equal-area rings at R/sqrt(2): inner =
    0 <= r <= R/sqrt(2), outer = R/sqrt(2) < r <= R. Mean z-score from
    med_stack is computed per ring per frame.

    Assumes label_frame contains at least one pixel labeled cluster_k; callers
    must skip clusters that don't exist (e.g. when there are 0 kept clusters).

    Args:
        label_frame: (H, W) array from _run_cluster_seeker (-2=background,
            -1=noise, 0..N-1=kept clusters).
        centroid: (row, col) of this cluster, from _run_cluster_seeker.
        med_stack: 3D array (frames, height, width) of z-scored median frames.
        cluster_k: which kept cluster to analyze.

    Returns:
        inner_trace: 1D array (n_frames,), mean z-score in the inner ring per frame.
        outer_trace: 1D array (n_frames,), mean z-score in the outer ring per frame.
        R: enclosing-circle radius in pixels.
        inner_mask: (H, W) boolean mask of the inner ring.
        outer_mask: (H, W) boolean mask of the outer ring.
    """
    coords = np.argwhere(label_frame == cluster_k)
    row_c, col_c = centroid
    dists = np.sqrt((coords[:, 0] - row_c) ** 2 + (coords[:, 1] - col_c) ** 2)
    R = _resolve_R(dists, centroid, label_frame.shape)
    split = R / np.sqrt(2)

    height, width = label_frame.shape
    inner_mask = np.zeros((height, width), dtype=bool)
    outer_mask = np.zeros((height, width), dtype=bool)
    inner_mask[coords[dists <= split, 0], coords[dists <= split, 1]] = True
    outer_mask[coords[dists > split, 0], coords[dists > split, 1]] = True

    n_frames = med_stack.shape[0]
    inner_trace = med_stack[:, inner_mask].mean(axis=1) if inner_mask.any() else np.full(n_frames, np.nan)
    outer_trace = med_stack[:, outer_mask].mean(axis=1) if outer_mask.any() else np.full(n_frames, np.nan)

    return inner_trace, outer_trace, R, inner_mask, outer_mask


def compute_cluster_trace(
    label_frame: np.ndarray,
    centroid: tuple[float, float],
    med_stack: np.ndarray,
    cluster_k: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Whole-cluster z-score trace for one DBSCAN cluster (no inner/outer ring split).

    Used when there is more than one kept cluster: each cluster is
    represented by a single trace over its own pixels so cluster-to-cluster
    peak timing can be compared directly, rather than splitting each cluster
    into rings (which measures spread within one release site, not
    synchrony between separate ones).

    Args:
        label_frame: (H, W) array from _run_cluster_seeker (-2=background,
            -1=noise, 0..N-1=kept clusters).
        centroid: (row, col) of this cluster, from _run_cluster_seeker.
        med_stack: 3D array (frames, height, width) of z-scored median frames.
        cluster_k: which kept cluster to analyze.

    Returns:
        trace: 1D array (n_frames,), mean z-score within the cluster per frame.
        R: enclosing-circle radius in pixels (for display only), capped at the
            centroid's distance to the nearest frame edge (see _resolve_R).
        mask: (H, W) boolean mask of the cluster's own pixels.
    """
    mask = label_frame == cluster_k
    coords = np.argwhere(mask)
    row_c, col_c = centroid
    dists = np.sqrt((coords[:, 0] - row_c) ** 2 + (coords[:, 1] - col_c) ** 2)
    R = _resolve_R(dists, centroid, label_frame.shape)

    trace = med_stack[:, mask].mean(axis=1)
    return trace, R, mask


def _peak_offset_from_spike(trace: np.ndarray, spike_frame_idx: int) -> int | None:
    """Index (relative to the spike frame) of trace's peak, or None if trace is all-NaN."""
    if np.all(np.isnan(trace)):
        return None
    return int(np.nanargmax(trace)) - spike_frame_idx
