"""
Region analysis for categorized images.

Picks a critical frame (spike or spike+1) by density-gated hotspot detection,
clusters its bright pixels with morphological dilation + connected components,
and computes per-cluster spatial/temporal stats. See RegionAnalyzer's docstring
for the full picture.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import curve_fit
from skimage.measure import label as skimage_label

from classes.spatial_categorization import CATEGORY_BRIGHT

# Pixel scaling constants (pixel/um)
PIXEL_SCALE = {
    "10X": 0.75,
    "40X": 3.0,
    "60X": 4.5,
}

EPS_UM = 50.0  # inter-varicosity gap in um; sets dilation disk radius (tunable)
MIN_CLUSTER_FRACTION = 0.05  # keep clusters covering at least this fraction of bright pixels

WINDOW_PX_BY_OBJ = {"10X": 201, "40X": 255, "60X": 511}  # local-density window size, per objective

# Min local bright-pixel density (uniform_filter) to qualify as a hotspot, per objective --
# lower magnification packs more (smaller, noisier) bright pixels per window, so needs a
# looser bar than higher magnification's larger, cleaner bright regions.
DENSITY_THRESH_BY_OBJ = {"10X": 0.15, "40X": 0.1, "60X": 0.05}

MIN_DECAY_FIT_FRAMES = 3  # fewer post-peak frames than this and the exponential fit is skipped
MIN_DECAY_FIT_RANGE = 1e-6  # post-peak signal must vary by at least this much or the fit is skipped (degenerate/flat trace)
MIN_DECAY_FIT_R2 = 0.8  # lasting time is suppressed (None) when the decay fit's R^2 is below this


class RegionAnalyzer:
    """
    Find clusters of bright pixels on the critical frame.

    Picks the spike frame or spike+1 as the critical frame -- whichever is the
    earliest to show a density-gated hotspot (local bright-pixel density,
    from the categorizer's own bright mask, gated at DENSITY_THRESH_BY_OBJ[obj] within a
    WINDOW_PX_BY_OBJ-sized window, spike frame checked first). Its bright
    pixels are then clustered with morphological dilation + connected
    components, and undersized clusters are dropped. Each kept cluster gets a
    centroid, an enclosing-circle radius (R), and a z-score trace across the
    segment (inner/outer ring split for a single cluster, whole-cluster trace
    when there are multiple).

    Result dict per cluster (from get_results()):
        centroid : (row, col) in pixels, z-score-weighted toward the cluster's
                   brightest sub-region rather than its plain geometric mean
                   (see _weighted_centroid)
        R_lat_px : enclosing-circle radius in pixels, from the critical/latency
                   frame above (used for ring split + latency only)
        R_lat_um : enclosing-circle radius in µm

    critical_frame_area_um2 (from get_results()) is this frame's own
    kept-cluster area (undersized-cluster pixels excluded) --
    critical_frame_area_pct stays the raw B% (diagnostic only; it no longer
    drives any detection decision).

    spike_frame_clusters / spike_plus1_frame_clusters (from get_results())
    independently report every density-gated cluster's own pixel/µm² size on
    the spike frame and spike+1 frame, regardless of which one was picked as
    critical -- there is no single "max-area frame" winner.

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

        self.area_pct = compute_area_pct(cat_stack)  # diagnostic B% trace only, no longer decision-driving

        eps_px = compute_eps_px(obj)
        window_px = compute_window_px(obj)
        density_thresh = compute_density_thresh(obj)

        (
            self.critical_frame_idx,
            self.significant,
            self.label_frame,
            self.centroids,
            self.n_raw_clusters,
        ) = self._detect_critical_frame(cat_stack, med_stack, spike_frame_idx, eps_px, window_px, density_thresh)

        self.hotspot_area_um2 = self._compute_hotspot_area_trace(cat_stack, eps_px, window_px, density_thresh)
        peak_search_end = min(self.spike_frame_idx + 2, len(self.hotspot_area_um2))
        self.decay_peak_frame_idx = self.spike_frame_idx + int(
            np.argmax(self.hotspot_area_um2[self.spike_frame_idx:peak_search_end])
        )
        self.decay_fit_A, self.decay_tau_frames, self.decay_fit_r2 = fit_decay_tau(
            self.hotspot_area_um2, self.decay_peak_frame_idx
        )

        self.clusters = self._build_clusters(med_stack)

        (
            self.spike_frame_label_frame,
            self.spike_frame_clusters,
            self.spike_plus1_frame_label_frame,
            self.spike_plus1_frame_clusters,
        ) = self._report_frame_clusters(cat_stack, med_stack, spike_frame_idx, eps_px, window_px, density_thresh)

    def _detect_critical_frame(
        self,
        cat_stack: np.ndarray,
        med_stack: np.ndarray,
        spike_frame_idx: int,
        eps_px: int,
        window_px: int,
        density_thresh: float,
    ) -> tuple[int, bool, np.ndarray, list[tuple[float, float]], int]:
        """Pick spike or spike+1 as the critical frame -- earliest one to show a density-gated hotspot.

        Thin wrapper around detect_hotspot(): builds the (frame_idx, categorized_frame,
        z_scored_frame) candidate list from this segment's full cat_stack/med_stack and
        delegates the earliest-wins detection loop to the shared module-level function.

        Returns:
            (critical_frame_idx, significant, label_frame, centroids, n_raw_clusters).
        """
        candidate_idxs = [spike_frame_idx]
        if spike_frame_idx + 1 < cat_stack.shape[0]:
            candidate_idxs.append(spike_frame_idx + 1)
        candidates = [(idx, cat_stack[idx], med_stack[idx]) for idx in candidate_idxs]

        significant, critical_frame_idx, label_frame, centroids, n_raw = detect_hotspot(
            candidates, eps_px, window_px, density_thresh
        )
        return critical_frame_idx, significant, label_frame, centroids, n_raw

    def _compute_hotspot_area_trace(
        self, cat_stack: np.ndarray, eps_px: int, window_px: int, density_thresh: float
    ) -> np.ndarray:
        """Density-gated total kept-cluster area (um^2) per frame, for the decay-tau fit.

        Centroids aren't needed here (z_frame=None) -- only the total kept-pixel count per
        frame matters, so _weighted_centroid falls back to its cheap unweighted-mean path.
        """
        n_frames = cat_stack.shape[0]
        hotspot_area_um2 = np.zeros(n_frames, dtype=float)
        for idx in range(n_frames):
            bright_mask = cat_stack[idx] == CATEGORY_BRIGHT
            label_frame, _, _ = _run_density_gated_cluster_seeker(
                bright_mask, eps_px, window_px, density_thresh, z_frame=None
            )
            kept_px = int(np.count_nonzero(label_frame >= 0))
            hotspot_area_um2[idx] = self._area_to_um2(kept_px)
        return hotspot_area_um2

    def _report_frame_clusters(
        self,
        cat_stack: np.ndarray,
        med_stack: np.ndarray,
        spike_frame_idx: int,
        eps_px: int,
        window_px: int,
        density_thresh: float,
    ) -> tuple[np.ndarray, list[dict], np.ndarray | None, list[dict] | None]:
        """Per-cluster pixel/um^2 sizes for the spike frame and spike+1 frame, independently.

        Unlike _detect_critical_frame (earliest-wins), both candidate frames are clustered here
        regardless of which one was picked as critical, so callers can compare the two frames'
        own hotspots directly instead of only seeing whichever one "won".

        Returns:
            (spike_frame_label_frame, spike_frame_clusters,
             spike_plus1_frame_label_frame, spike_plus1_frame_clusters)
            spike_plus1 fields are None when spike_frame_idx + 1 is out of range.
        """
        spike_label_frame, spike_clusters = self._cluster_report_for_frame(
            cat_stack, med_stack, spike_frame_idx, eps_px, window_px, density_thresh
        )

        plus1_idx = spike_frame_idx + 1
        if plus1_idx < cat_stack.shape[0]:
            plus1_label_frame, plus1_clusters = self._cluster_report_for_frame(
                cat_stack, med_stack, plus1_idx, eps_px, window_px, density_thresh
            )
        else:
            plus1_label_frame, plus1_clusters = None, None

        return spike_label_frame, spike_clusters, plus1_label_frame, plus1_clusters

    def _cluster_report_for_frame(
        self,
        cat_stack: np.ndarray,
        med_stack: np.ndarray,
        frame_idx: int,
        eps_px: int,
        window_px: int,
        density_thresh: float,
    ) -> tuple[np.ndarray, list[dict]]:
        """Density-gated clusters for one frame, as {cluster_id, centroid, area_px, area_um2} dicts."""
        bright_mask = cat_stack[frame_idx] == CATEGORY_BRIGHT
        label_frame, centroids, _ = _run_density_gated_cluster_seeker(
            bright_mask, eps_px, window_px, density_thresh, z_frame=med_stack[frame_idx]
        )
        clusters = []
        for cluster_id, centroid in enumerate(centroids):
            area_px = int(np.count_nonzero(label_frame == cluster_id))
            clusters.append({
                "cluster_id": cluster_id,
                "centroid": centroid,
                "area_px": area_px,
                "area_um2": self._area_to_um2(area_px),
            })
        return label_frame, clusters

    def _build_clusters(self, med_stack: np.ndarray) -> list[dict]:
        """Per-cluster result dicts for the critical frame (self.label_frame/self.centroids).

        1 centroid -> inner/outer ring split (compute_ring_traces): spread
        within the one release site.
        >1 centroids -> one whole-cluster trace per cluster
        (compute_cluster_trace): lets cluster-to-cluster peak timing be
        compared directly instead of splitting each into rings.
        """
        clusters = []
        if len(self.centroids) == 1:
            centroid = self.centroids[0]
            inner_trace, outer_trace, R_lat, inner_mask, outer_mask = compute_ring_traces(
                self.label_frame, centroid, med_stack, 0
            )
            clusters.append({
                "centroid":    centroid,
                "R_lat_px":    R_lat,
                "R_lat_um":    self._px_to_um(R_lat),
                "inner_trace": inner_trace,
                "outer_trace": outer_trace,
                "inner_mask":  inner_mask,
                "outer_mask":  outer_mask,
            })
        elif len(self.centroids) > 1:
            for cluster_k, centroid in enumerate(self.centroids):
                trace, R_lat, mask = compute_cluster_trace(self.label_frame, centroid, med_stack, cluster_k)
                clusters.append({
                    "centroid": centroid,
                    "R_lat_px": R_lat,
                    "R_lat_um": self._px_to_um(R_lat),
                    "trace":    trace,
                    "mask":     mask,
                })
        return clusters

    # ── Unit conversion helpers ────────────────────────────────────────────────

    def _px_to_um(self, pixels: float) -> float:
        return pixels * self.um_per_pixel

    def _area_to_um2(self, area_px: float) -> float:
        return area_px * (self.um_per_pixel ** 2)

    # ── Result accessors ──────────────────────────────────────────────────────

    def get_results(self) -> dict:
        """Get per-cluster region results for the critical frame."""
        critical_frame_area_pct = float(self.area_pct[self.critical_frame_idx])
        critical_frame_kept_px = int(np.count_nonzero(self.label_frame >= 0))
        return {
            "critical_frame_idx":      self.critical_frame_idx,
            "critical_frame_offset":   self.critical_frame_idx - self.spike_frame_idx,
            "critical_frame_area_pct": critical_frame_area_pct,
            "critical_frame_area_um2": self._area_to_um2(critical_frame_kept_px),
            "spike_frame_clusters":       self.spike_frame_clusters,
            "spike_plus1_frame_clusters": self.spike_plus1_frame_clusters,
            "decay_peak_frame_idx":    self.decay_peak_frame_idx if self.significant else None,
            "decay_peak_offset":       (self.decay_peak_frame_idx - self.spike_frame_idx) if self.significant else None,
            "decay_fit_r2":            self.decay_fit_r2 if self.significant else None,
            "n_clusters":               len(self.clusters),
            "clusters": [
                {"centroid": c["centroid"], "R_lat_px": c["R_lat_px"], "R_lat_um": c["R_lat_um"]}
                for c in self.clusters
            ],
        }

    def get_summary(self) -> dict:
        """Summary for the critical frame."""
        return {
            "obj":         self.obj,
            "n_clusters":  len(self.clusters),
            "has_region":  len(self.clusters) > 0,
            "significant": self.significant,
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

    def get_lasting_time_ms(self, frame_duration_ms: float) -> float | None:
        """Decay time constant (tau) of the post-peak Bright% falloff, in milliseconds.

        tau comes from fit_decay_tau(), fit in frame units (independent of
        frame_duration_ms), so this method just converts it. None if the fit
        was skipped, failed to converge (see fit_decay_tau's docstring), or
        the fit's R^2 is below MIN_DECAY_FIT_R2 (unreliable tau despite
        curve_fit converging).

        Args:
            frame_duration_ms: milliseconds per frame.

        Returns:
            tau in ms, or None if no reliable decay fit is available.
        """
        if self.decay_tau_frames is None:
            return None
        if self.decay_fit_r2 is None or self.decay_fit_r2 < MIN_DECAY_FIT_R2:
            return None
        return self.decay_tau_frames * frame_duration_ms

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
    """Area percentage (B%) of bright pixels per frame.

    This is the B% detection-criterion signal: a real ACh event shows a
    clear elevation at the spike frame (or spike_frame+1) vs the baseline
    frame before it.

    Args:
        stack: 3D array (frames, height, width) of categorized frames.

    Returns:
        1D array (n_frames,) of B% per frame.
    """
    total_px = stack.shape[1] * stack.shape[2]
    return np.count_nonzero(stack == CATEGORY_BRIGHT, axis=(1, 2)) / total_px * 100


def compute_eps_px(obj: str) -> int:
    """Convert EPS_UM to pixels for this objective.

    Args:
        obj: Objective magnification, must be a key of PIXEL_SCALE.

    Returns:
        Dilation disk radius in pixels.
    """
    return int(EPS_UM * PIXEL_SCALE[obj])


def compute_window_px(obj: str) -> int:
    """Local-density window size (pixels) for this objective.

    Args:
        obj: Objective magnification, must be a key of WINDOW_PX_BY_OBJ.

    Returns:
        uniform_filter window size in pixels, from WINDOW_PX_BY_OBJ.
    """
    return WINDOW_PX_BY_OBJ[obj]


def compute_density_thresh(obj: str) -> float:
    """Local-density hotspot threshold for this objective.

    Args:
        obj: Objective magnification, must be a key of DENSITY_THRESH_BY_OBJ.

    Returns:
        Minimum local bright-pixel density to qualify as a hotspot, from DENSITY_THRESH_BY_OBJ.
    """
    return DENSITY_THRESH_BY_OBJ[obj]


def _run_density_gated_cluster_seeker(
    bright_mask: np.ndarray, eps_px: int, window_px: int, density_thresh: float, z_frame: np.ndarray | None
) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """Cluster only the bright pixels that sit in a locally dense neighborhood.

    A flat bright/background mask treats an isolated noise pixel the same as a real,
    spatially-compact release site. Gating by local density (fraction of bright pixels
    within a window_px-sized neighborhood) before clustering rejects the former while
    keeping the latter. Since the density gate can clip a real hotspot's own ragged
    low-density edge, accepted clusters are expanded back out to the full bright-mask
    blob(s) they touch, so no genuinely connected bright pixel is lost to a marginal
    per-pixel density value.

    Args:
        bright_mask: (H, W) boolean array, True where the categorizer marked a pixel
            bright (cat_stack[idx] == CATEGORY_BRIGHT).
        eps_px: dilation disk radius in pixels, from compute_eps_px() -- passed straight
            through to _run_cluster_seeker.
        window_px: local-density window size in pixels, from compute_window_px().
        density_thresh: minimum local bright-pixel density to qualify as a hotspot.
        z_frame: (H, W) z-scored frame to weight centroids by pixel intensity, or None
            for an unweighted mean (see _weighted_centroid).

    Returns:
        (label_frame, centroids, n_raw_clusters) -- label_frame is expanded to full
        bright-mask blobs, centroids are computed on the pre-expansion gated pixels
        (see _run_cluster_seeker), same shapes/semantics as _run_cluster_seeker's own return.
    """
    density = uniform_filter(bright_mask.astype(float), size=window_px, mode="constant", cval=0.0)
    hotspot_mask = bright_mask & (density >= density_thresh)

    gated_label_frame, centroids, n_raw = _run_cluster_seeker(hotspot_mask.astype(int), eps_px, z_frame)
    n_clusters = len(centroids)

    labeled_bright = skimage_label(bright_mask)
    label_frame = np.full(bright_mask.shape, -2, dtype=int)
    for cluster_id in range(n_clusters):
        cluster_mask = gated_label_frame == cluster_id
        touched_blob_ids = set(labeled_bright[cluster_mask].tolist()) - {0}
        whole_mask = np.isin(labeled_bright, list(touched_blob_ids))
        label_frame[whole_mask] = cluster_id

    return label_frame, centroids, n_raw


def detect_hotspot(
    candidates: list[tuple[int, np.ndarray, np.ndarray]], eps_px: int, window_px: int, density_thresh: float
) -> tuple[bool, int, np.ndarray, list[tuple[float, float]], int]:
    """Earliest-wins density-gated hotspot detection over a list of candidate frames.

    Tries each candidate in order (e.g. spike frame before spike+1) and returns the first
    one with an accepted density-gated cluster. Shared by RegionAnalyzer._detect_critical_frame
    (which has a full cat_stack/med_stack on hand) and any lean, RegionAnalyzer-free caller that
    only has a couple of already-categorized candidate frames (e.g. a per-segment reliability
    check that categorizes just spike/spike+1, not the whole segment).

    Args:
        candidates: (frame_idx, categorized_frame, z_scored_frame) tuples, in the order to try
            them. categorized_frame is 0=background/1=bright (see CATEGORY_BRIGHT).
        eps_px: dilation disk radius in pixels, from compute_eps_px().
        window_px: local-density window size in pixels, from compute_window_px().
        density_thresh: minimum local bright-pixel density to qualify as a hotspot,
            from compute_density_thresh() -- varies per objective.

    Returns:
        (detected, frame_idx, label_frame, centroids, n_raw_clusters) -- frame_idx/label_frame/
        centroids/n_raw_clusters come from the first detecting candidate, or the first candidate
        (with an empty label_frame, no centroids) when none of them detect.
    """
    for idx, cat_frame, z_frame in candidates:
        bright_mask = cat_frame == CATEGORY_BRIGHT
        label_frame, centroids, n_raw = _run_density_gated_cluster_seeker(
            bright_mask, eps_px, window_px, density_thresh, z_frame=z_frame
        )
        if centroids:
            return True, idx, label_frame, centroids, n_raw

    first_idx, first_cat_frame, _ = candidates[0]
    empty_label_frame = np.full(first_cat_frame.shape, -2, dtype=int)
    return False, first_idx, empty_label_frame, [], 0


def _decay_model(t: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """Single-exponential decay: amplitude * exp(-t/tau)."""
    return amplitude * np.exp(-t / tau)


def fit_decay_tau(signal: np.ndarray, peak_frame_idx: int) -> tuple[float | None, float | None, float | None]:
    """Fit a single-exponential decay to a per-frame signal from its post-peak-frame peak onward.

    t=0 is pinned to peak_frame_idx (not the spike frame) so the fit only
    sees the falling side of the curve, never the rising side. r_squared is
    returned so a poor fit is visible in the exported data rather than
    silently producing a misleading tau.

    Skipped (all None) when there are fewer than MIN_DECAY_FIT_FRAMES frames
    after the peak, or when the post-peak trace barely varies (a flat/near-zero
    tail has no decay to fit -- curve_fit would either fail or return a
    meaningless tau). Also all None if curve_fit doesn't converge.

    Args:
        signal: 1D array (n_frames,) -- currently RegionAnalyzer's density-gated
            hotspot_area_um2 trace, from _compute_hotspot_area_trace().
        peak_frame_idx: index of the signal's peak (from RegionAnalyzer.__init__).

    Returns:
        (amplitude, tau_frames, r_squared), each None together if the fit was
        skipped or failed. tau_frames is in frame units -- multiply by
        frame_duration_ms to get milliseconds (see get_lasting_time_ms).
    """
    y = signal[peak_frame_idx:]
    if len(y) < MIN_DECAY_FIT_FRAMES or float(np.ptp(y)) < MIN_DECAY_FIT_RANGE:
        return None, None, None

    t = np.arange(len(y), dtype=np.float64)
    amplitude_guess = max(float(y[0]), 1e-3)
    tau_guess = len(y) / 2.0

    try:
        popt, _ = curve_fit(
            _decay_model, t, y, p0=[amplitude_guess, tau_guess], bounds=([0.0, 1e-3], [np.inf, np.inf]), maxfev=2000
        )
    except RuntimeError:
        return None, None, None

    amplitude, tau = float(popt[0]), float(popt[1])
    if not np.isfinite(tau) or tau <= 0:
        return None, None, None

    residuals = y - _decay_model(t, amplitude, tau)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return amplitude, tau, r_squared


def _weighted_centroid(rows: np.ndarray, cols: np.ndarray, z_frame: np.ndarray | None) -> tuple[float, float]:
    """Z-score-weighted centroid of a pixel set, falling back to an unweighted mean.

    Weighting by z_frame's value at each pixel pulls the centroid toward the
    brightest sub-region of a cluster instead of treating every non-background
    pixel (dim or bright) as equally important. Falls back to a plain mean
    when z_frame is None (caller has no z-score data, e.g.
    _compute_hotspot_area_trace(), which only needs kept-pixel counts and
    never uses the returned centroid) or when every weight is non-positive
    (degenerate/empty overlap, shouldn't happen in practice since these
    pixels are already above the dim/bright threshold).

    Args:
        rows: row coordinates of the pixel set.
        cols: column coordinates of the pixel set (same length as rows).
        z_frame: (H, W) z-scored frame to weight by, or None to skip weighting.

    Returns:
        (centroid_row, centroid_col).
    """
    if z_frame is not None:
        weights = np.clip(z_frame[rows, cols].astype(np.float64), 0.0, None)
        if weights.sum() > 0:
            return float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights))
    return float(rows.mean()), float(cols.mean())


def _run_cluster_seeker(
    frame: np.ndarray, eps_px: int, z_frame: np.ndarray | None = None
) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """Cluster bright pixels with morphological dilation + connected components.

    Uses a distance transform to find all pixels within eps_px of any bright pixel
    (equivalent to dilating by a disk of radius eps_px but O(H×W) regardless of
    eps size — no large kernel convolution). Runs connected components on that
    expanded mask, intersects each component back with the original bright pixels,
    then drops components below MIN_CLUSTER_FRACTION.

    Args:
        frame: 2D array (0=background, 1=dim, 2=bright) for a single frame.
        eps_px: dilation disk radius in pixels, from compute_eps_px().
        z_frame: (H, W) z-scored frame to weight centroids by pixel intensity;
            None for an unweighted mean (see _weighted_centroid).

    Returns:
        label_frame: (H, W) array; -2=background, -1=noise/undersized cluster,
            0..N-1=kept clusters, largest first.
        centroids: (row, col) per kept cluster, same order as label_frame indices.
        n_raw_components: number of connected components before the size filter.
    """
    bright_mask = frame == CATEGORY_BRIGHT
    label_frame = np.full(frame.shape, -2, dtype=int)
    if not bright_mask.any():
        return label_frame, [], 0

    # distance_transform_edt measures distance to the nearest False pixel.
    # Flipping bright_mask (~) makes bright pixels False so every other pixel's
    # distance = "how far am I from the nearest bright pixel?". Thresholding at
    # eps_px gives every pixel that falls within eps_px of any bright pixel.
    within_eps_of_bright = distance_transform_edt(~bright_mask) <= eps_px

    # Connected components on within_eps_of_bright: two bright pixels whose
    # eps_px zones overlap end up in the same component, bridging dark gaps up
    # to eps_px wide without needing them to physically touch.
    component_map = skimage_label(within_eps_of_bright, connectivity=2)
    n_raw_components = int(component_map.max())

    total_bright_px = int(bright_mask.sum())
    label_frame[bright_mask] = -1  # noise until promoted to a kept cluster

    # For each component, intersect back with bright_mask to recover only the
    # original bright pixels — the dilation expansion is discarded here.
    # Components too small relative to total bright pixels are rejected.
    accepted_components = []
    for component_id in range(1, n_raw_components + 1):
        bright_px_in_component = (component_map == component_id) & bright_mask
        bright_px_count = int(bright_px_in_component.sum())
        if total_bright_px > 0 and bright_px_count / total_bright_px >= MIN_CLUSTER_FRACTION:
            accepted_components.append((bright_px_count, bright_px_in_component))
    # Sort largest first so cluster index 0 always refers to the biggest cluster.
    accepted_components.sort(key=lambda component: component[0], reverse=True)

    centroids = []
    for cluster_idx, (_, bright_px_in_component) in enumerate(accepted_components):
        label_frame[bright_px_in_component] = cluster_idx
        bright_px_coords = np.argwhere(bright_px_in_component)
        centroids.append(_weighted_centroid(bright_px_coords[:, 0], bright_px_coords[:, 1], z_frame))

    return label_frame, centroids, n_raw_components


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
    """Inner/outer ring z-score traces for one kept cluster.

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
    is_outer = (dists > split) & (dists <= R)
    inner_mask[coords[dists <= split, 0], coords[dists <= split, 1]] = True
    outer_mask[coords[is_outer, 0], coords[is_outer, 1]] = True

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
    """Whole-cluster z-score trace for one kept cluster (no inner/outer ring split).

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
