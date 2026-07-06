"""
Region analysis for categorized images.

Picks a critical frame (spike or spike+1), clusters its non-background
pixels with DBSCAN, and computes per-cluster spatial/temporal stats. See
RegionAnalyzer's docstring for the full picture.
"""

import numpy as np
from scipy.optimize import curve_fit
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

MIN_DECAY_FIT_FRAMES = 3  # fewer post-peak frames than this and the exponential fit is skipped
MIN_DECAY_FIT_RANGE = 1e-6  # post-peak Bright% must vary by at least this much or the fit is skipped (degenerate/flat trace)
MIN_DECAY_FIT_R2 = 0.8  # lasting time is suppressed (None) when the decay fit's R^2 is below this


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
        centroid : (row, col) in pixels, z-score-weighted toward the cluster's
                   brightest sub-region rather than its plain geometric mean
                   (see _weighted_centroid)
        R_lat_px : enclosing-circle radius in pixels, from the critical/latency
                   frame above (used for ring split + latency only)
        R_lat_um : enclosing-circle radius in µm

    critical_frame_area_um2 (from get_results()) is this frame's own
    DBSCAN-kept-cluster area (noise/undersized-cluster pixels excluded) --
    critical_frame_area_pct stays the raw B+D% (it also drives the
    significance/saturation threshold logic above).

    Top-level max_area_* fields (from get_results()) instead describe the
    max-area frame -- whichever of spike/spike+1 has the larger raw B+D%,
    independent of the critical/latency frame's significance-threshold pick.
    The reported area is likewise that frame's DBSCAN-kept-cluster area, not
    the raw non-background count. max_area_eq_radius_um is the circle-equivalent
    radius (sqrt(area/pi)) of that same area, for direct comparison against R_lat_um.

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

        self.area_pct = compute_area_pct(cat_stack)
        self.critical_frame_idx, self.significant = pick_critical_frame(self.area_pct, spike_frame_idx)

        peak_search_end = min(self.spike_frame_idx + 2, len(self.area_pct))
        self.decay_peak_frame_idx = self.spike_frame_idx + int(
            np.argmax(self.area_pct[self.spike_frame_idx:peak_search_end])
        )
        self.decay_fit_A, self.decay_tau_frames, self.decay_fit_r2 = fit_decay_tau(
            self.area_pct, self.decay_peak_frame_idx
        )

        eps_px, min_samples = eps_and_min_samples(obj)
        if self.significant:
            self.label_frame, self.centroids, self.n_raw_clusters, self.saturated = _detect_clusters(
                cat_stack[self.critical_frame_idx], self.area_pct[self.critical_frame_idx], eps_px, min_samples,
                z_frame=med_stack[self.critical_frame_idx],
            )
        else:
            self.label_frame = np.full(cat_stack.shape[1:], -2, dtype=int)
            self.centroids, self.n_raw_clusters, self.saturated = [], 0, False

        self.clusters = self._build_clusters(med_stack)
        (
            self.max_area_frame_idx,
            self.max_area_offset,
            self.max_area_um2,
            self.max_area_eq_radius_um,
            self.max_area_x_span_px,
            self.max_area_y_span_px,
            self.max_area_x_span_um,
            self.max_area_y_span_um,
            self.max_area_x_min_px,
            self.max_area_y_min_px,
        ) = self._compute_max_area(cat_stack, med_stack, spike_frame_idx, eps_px, min_samples)

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

    def _compute_max_area(
        self, cat_stack: np.ndarray, med_stack: np.ndarray, spike_frame_idx: int, eps_px: int, min_samples: int
    ) -> tuple[int, int, float, float, int | None, int | None, float | None, float | None, int | None, int | None]:
        """Max-area frame stats, independent of the critical-frame pick above.

        Picks whichever of frame0 (spike) / frame1 (spike+1) has the larger
        raw area_pct -- used only for the headline area stat, not latency.
        The reported area is that frame's DBSCAN-kept-cluster area (noise
        excluded), not the raw non-background count. X/Y span is measured from
        the combined DBSCAN-kept mask on that same frame.

        Returns:
            (max_area_frame_idx, max_area_offset, max_area_um2,
            max_area_eq_radius_um, x_span_px, y_span_px, x_span_um, y_span_um,
            x_min_px, y_min_px)
            x_min_px/y_min_px are the top-left corner of the span bbox (col_min,
            row_min), used by callers drawing a Rectangle overlay. None when no
            accepted clusters exist.
        """
        candidate_idxs = [spike_frame_idx]
        if spike_frame_idx + 1 < len(self.area_pct):
            candidate_idxs.append(spike_frame_idx + 1)
        max_area_frame_idx = max(candidate_idxs, key=lambda idx: self.area_pct[idx])
        max_area_offset = max_area_frame_idx - spike_frame_idx

        label_frame, _, _, _ = _detect_clusters(
            cat_stack[max_area_frame_idx], self.area_pct[max_area_frame_idx], eps_px, min_samples,
            z_frame=med_stack[max_area_frame_idx],
        )
        max_area_kept_px = int(np.count_nonzero(label_frame >= 0))
        max_area_um2 = self._area_to_um2(max_area_kept_px)
        max_area_eq_radius_um = float(np.sqrt(max_area_um2 / np.pi))
        mask = label_frame >= 0
        x_span_px, y_span_px = compute_xy_span(mask)
        x_span_um = self._px_to_um(x_span_px) if x_span_px is not None else None
        y_span_um = self._px_to_um(y_span_px) if y_span_px is not None else None
        coords = np.argwhere(mask)
        x_min_px = int(coords[:, 1].min()) if coords.size > 0 else None
        y_min_px = int(coords[:, 0].min()) if coords.size > 0 else None
        return max_area_frame_idx, max_area_offset, max_area_um2, max_area_eq_radius_um, x_span_px, y_span_px, x_span_um, y_span_um, x_min_px, y_min_px

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
            "max_area_frame_idx":      self.max_area_frame_idx if self.significant else None,
            "max_area_offset":         self.max_area_offset if self.significant else None,
            "max_area_um2":            self.max_area_um2 if self.significant else None,
            "max_area_eq_radius_um":   self.max_area_eq_radius_um if self.significant else None,
            "max_area_x_span_px":      self.max_area_x_span_px if self.significant else None,
            "max_area_y_span_px":      self.max_area_y_span_px if self.significant else None,
            "max_area_x_span_um":      self.max_area_x_span_um if self.significant else None,
            "max_area_y_span_um":      self.max_area_y_span_um if self.significant else None,
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
            "saturated":   self.saturated,
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


def pick_critical_frame(area_pct: np.ndarray, spike_frame_idx: int) -> tuple[int, bool]:
    """Pick spike or spike+1 for clustering, biased toward the spike frame.

    Compares each candidate frame's B+D% against a baseline-derived
    significance threshold (mean + AREA_PCT_SIGMA_MULT * std, over every frame
    before the spike frame) instead of simply picking whichever is higher --
    avoids flipping to spike+1 on frame-to-frame noise. Only switches to
    spike+1 when the spike frame itself isn't significantly elevated but
    spike+1 is (delayed-signal case).

    Falls back to spike_frame_idx with significant=False when neither
    candidate clears the threshold, so the caller can skip clustering
    entirely instead of running DBSCAN on a frame that's indistinguishable
    from baseline noise (a small stray cluster there would otherwise still
    pass MIN_CLUSTER_FRACTION's purely relative size check and register as a
    false-positive detection).

    Args:
        area_pct: 1D array (n_frames,), from compute_area_pct().
        spike_frame_idx: index of the spike frame within the segment.

    Returns:
        (index of the frame to run clustering on -- spike_frame_idx or
        spike_frame_idx + 1, whether that frame actually cleared the
        significance threshold).
    """
    baseline = area_pct[:spike_frame_idx]
    threshold = baseline.mean() + AREA_PCT_SIGMA_MULT * baseline.std()

    if area_pct[spike_frame_idx] >= threshold:
        return spike_frame_idx, True

    next_idx = spike_frame_idx + 1
    if next_idx < len(area_pct) and area_pct[next_idx] >= threshold:
        return next_idx, True

    return spike_frame_idx, False


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


def compute_xy_span(mask: np.ndarray) -> tuple[int | None, int | None]:
    """X/Y span of a combined accepted-region mask, in pixels."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None, None
    y_span_px = int(coords[:, 0].max() - coords[:, 0].min() + 1)
    x_span_px = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    return x_span_px, y_span_px


def _decay_model(t: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """Single-exponential decay: amplitude * exp(-t/tau)."""
    return amplitude * np.exp(-t / tau)


def fit_decay_tau(area_pct: np.ndarray, peak_frame_idx: int) -> tuple[float | None, float | None, float | None]:
    """Fit a single-exponential decay to B+D% from its post-critical-frame peak onward.

    t=0 is pinned to peak_frame_idx (not the spike frame) so the fit only
    sees the falling side of the curve, never the rising side. A single
    exponential won't fit a genuine bi-phasic decay (e.g. a lingering Dim
    halo after the Bright core has faded) particularly well -- r_squared is
    returned precisely so that's visible in the exported data rather than
    silently producing a misleading tau.

    Skipped (all None) when there are fewer than MIN_DECAY_FIT_FRAMES frames
    after the peak, or when the post-peak trace barely varies (a flat/near-zero
    tail has no decay to fit -- curve_fit would either fail or return a
    meaningless tau). Also all None if curve_fit doesn't converge.

    Args:
        area_pct: 1D array (n_frames,), from compute_area_pct().
        peak_frame_idx: index of the B+D% peak (from RegionAnalyzer.__init__).

    Returns:
        (amplitude, tau_frames, r_squared), each None together if the fit was
        skipped or failed. tau_frames is in frame units -- multiply by
        frame_duration_ms to get milliseconds (see get_lasting_time_ms).
    """
    y = area_pct[peak_frame_idx:]
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
    when z_frame is None (caller has no z-score data, e.g. plot_results.py's
    hotspot-area line, which never uses the returned centroid anyway) or when
    every weight is non-positive (degenerate/empty overlap, shouldn't happen
    in practice since these pixels are already above the dim/bright threshold).

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
    frame: np.ndarray, eps_px: int, min_samples: int, z_frame: np.ndarray | None = None
) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """Cluster non-background pixels with DBSCAN, then drop undersized clusters.

    Args:
        frame: 2D array (0=background, 1=dim, 2=bright) for a single frame.
        eps_px: DBSCAN neighborhood radius in pixels, from eps_and_min_samples().
        min_samples: DBSCAN minimum samples per cluster, from eps_and_min_samples().
        z_frame: (H, W) z-scored frame (same frame as `frame`) to weight
            centroids by pixel intensity; None for an unweighted mean (see
            _weighted_centroid).

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
        _weighted_centroid(coords[remapped == new_cluster_label, 0], coords[remapped == new_cluster_label, 1], z_frame)
        for new_cluster_label in range(len(kept))
    ]

    return label_frame, centroids, n_raw_clusters


def _detect_clusters(
    frame: np.ndarray, area_pct_value: float, eps_px: int, min_samples: int, z_frame: np.ndarray | None = None
) -> tuple[np.ndarray, list[tuple[float, float]], int, bool]:
    """Cluster non-background pixels, or fall back to 1 whole-frame cluster if saturated.

    Single source of truth for the saturation guard, shared by both the
    critical frame (RegionAnalyzer.__init__) and the independent max-area
    frame (RegionAnalyzer._compute_max_area) -- skips DBSCAN when
    area_pct_value is at/above SATURATION_AREA_PCT (a dense enough point
    cloud makes sklearn's neighbor-graph construction blow up memory) and
    treats every non-background pixel as one big cluster instead, since at
    that density it's one region, not discrete release sites.

    Args:
        frame: 2D array (0=background, 1=dim, 2=bright) for a single frame.
        area_pct_value: this frame's B+D% (from compute_area_pct()).
        eps_px: DBSCAN neighborhood radius in pixels, from eps_and_min_samples().
        min_samples: DBSCAN minimum samples per cluster, from eps_and_min_samples().
        z_frame: (H, W) z-scored frame (same frame as `frame`) to weight
            centroids by pixel intensity; None for an unweighted mean.

    Returns:
        label_frame: (H, W) array; -2=background, -1=noise/undersized cluster,
            0..N-1=kept clusters, largest first (or 0=the whole frame if saturated).
        centroids: (row, col) per kept cluster, same order as label_frame's indices.
        n_raw_clusters: number of clusters DBSCAN found before the size filter (1 if saturated).
        saturated: True if area_pct_value was at/above SATURATION_AREA_PCT.
    """
    if area_pct_value >= SATURATION_AREA_PCT:
        mask = frame > CATEGORY_BACKGROUND
        coords = np.argwhere(mask)
        label_frame = np.where(mask, 0, -2).astype(int)
        centroids = [_weighted_centroid(coords[:, 0], coords[:, 1], z_frame)]
        return label_frame, centroids, 1, True

    label_frame, centroids, n_raw_clusters = _run_cluster_seeker(frame, eps_px, min_samples, z_frame)
    return label_frame, centroids, n_raw_clusters, False


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
