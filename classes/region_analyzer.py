"""
Region analysis of one spike-aligned median segment (categorized + raw-intensity stacks).

  Step 1. Detect : critical frame = earliest of spike / spike+1 with a density-gated hotspot
  Step 2. Area   : per-frame hotspot area -> decay-tau fit (lasting time)
  Step 3. Report : per-cluster centroid + enclosing radius, spike and spike+1 cluster sizes
  Step 4. Flow   : TV-L1 flow + CAT keep mask, spike-1->spike ... spike+3->spike+4 (compute_flow())

Example:
    >>> analyzer = RegionAnalyzer(cat_stack, med_stack, spike_frame_idx, obj="10X")
    >>> results = analyzer.get_results()
    >>> analyzer.compute_flow(cat_stack, med_stack)   # only when the recording is significant
"""

## Modules
# Third-party imports
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import curve_fit
from skimage.measure import label as skimage_label

# Local imports
from classes.spatial_categorization import CATEGORY_BRIGHT
from functions.hotspot_flow import compute_flow_pairs

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

PIXEL_SCALE = {  # pixel/µm per objective
    "10X": 0.75,
    "40X": 3.0,
    "60X": 4.5,
}

# --- Step 1: detect --------------------------------------------------------
EPS_UM = 50.0                # µm: inter-varicosity gap -> dilation disk radius for clustering
MIN_CLUSTER_FRACTION = 0.05  # keep clusters holding at least this fraction of the bright pixels
WINDOW_PX_BY_OBJ = {"10X": 201, "40X": 201, "60X": 201}          # local-density window size
DENSITY_THRESH_BY_OBJ = {"10X": 0.1, "40X": 0.025, "60X": 0.016}  # min local bright density

# --- Step 2: decay fit -----------------------------------------------------
MIN_DECAY_FIT_FRAMES = 3     # fewer post-peak frames -> fit skipped
MIN_DECAY_FIT_RANGE = 1e-6   # flatter post-peak trace -> fit skipped
MIN_DECAY_FIT_R2 = 0.8       # lower R² -> lasting time reported as None


class RegionAnalyzer:
    """Critical-frame hotspot clusters, hotspot-area decay, and (on request) hotspot flow.

    Results after construction:
        critical_frame_idx, significant, label_frame, centroids      (step 1)
        hotspot_area_um2, decay_* fields                             (step 2)
        clusters, spike_frame_clusters, spike_plus1_frame_clusters   (step 3)
    After compute_flow():
        flow_pairs                                                   (step 4)
    """

    def __init__(self, cat_stack: np.ndarray, med_stack: np.ndarray, spike_frame_idx: int, obj: str = "10X") -> None:
        """
        Args:
            cat_stack: (frames, H, W) categorized frames (CATEGORY_BRIGHT = bright).
            med_stack: (frames, H, W) raw-intensity median frames, same shape.
            spike_frame_idx: index of the spike frame within the segment.
            obj: objective ("10X", "40X", "60X").
        """
        if obj not in PIXEL_SCALE:
            msg = f"Unknown objective: {obj}. Choose from {list(PIXEL_SCALE.keys())}"
            raise ValueError(msg)

        self.obj = obj
        self.pixel_per_um = PIXEL_SCALE[obj]
        self.um_per_pixel = 1.0 / self.pixel_per_um
        self.spike_frame_idx = spike_frame_idx
        self.flow_pairs: list[dict] = []

        self.area_pct = compute_area_pct(cat_stack)  # diagnostic B% only

        eps_px = compute_eps_px(obj)
        window_px = compute_window_px(obj)
        density_thresh = compute_density_thresh(obj)

        # ----- Step 1. Detect: critical frame -----
        (
            self.critical_frame_idx,
            self.significant,
            self.label_frame,
            self.centroids,
            self.n_raw_clusters,
        ) = self._detect_critical_frame(cat_stack, med_stack, spike_frame_idx, eps_px, window_px, density_thresh)

        # ----- Step 2. Area: hotspot area trace -> decay fit -----
        self.hotspot_area_um2 = self._compute_hotspot_area_trace(cat_stack, eps_px, window_px, density_thresh)
        peak_search_end = min(self.spike_frame_idx + 2, len(self.hotspot_area_um2))
        self.decay_peak_frame_idx = self.spike_frame_idx + int(
            np.argmax(self.hotspot_area_um2[self.spike_frame_idx:peak_search_end])
        )
        self.decay_fit_A, self.decay_tau_frames, self.decay_fit_r2 = fit_decay_tau(
            self.hotspot_area_um2, self.decay_peak_frame_idx
        )

        # ----- Step 3. Report: critical-frame clusters + spike / spike+1 cluster sizes -----
        self.clusters = self._build_clusters()
        (
            self.spike_frame_label_frame,
            self.spike_frame_clusters,
            self.spike_plus1_frame_label_frame,
            self.spike_plus1_frame_clusters,
        ) = self._report_frame_clusters(cat_stack, med_stack, spike_frame_idx, eps_px, window_px, density_thresh)

    # -----------------------------------------------------------------------
    # Step 1. Detect
    # -----------------------------------------------------------------------

    def _detect_critical_frame(
        self,
        cat_stack: np.ndarray,
        med_stack: np.ndarray,
        spike_frame_idx: int,
        eps_px: int,
        window_px: int,
        density_thresh: float,
    ) -> tuple[int, bool, np.ndarray, list[tuple[float, float]], int]:
        """detect_hotspot() over [spike, spike+1] -> (critical_frame_idx, significant, label_frame, centroids, n_raw)."""
        candidate_idxs = [spike_frame_idx]
        if spike_frame_idx + 1 < cat_stack.shape[0]:
            candidate_idxs.append(spike_frame_idx + 1)
        candidates = [(idx, cat_stack[idx], med_stack[idx]) for idx in candidate_idxs]

        significant, critical_frame_idx, label_frame, centroids, n_raw = detect_hotspot(
            candidates, eps_px, window_px, density_thresh
        )
        return critical_frame_idx, significant, label_frame, centroids, n_raw

    # -----------------------------------------------------------------------
    # Step 2. Area
    # -----------------------------------------------------------------------

    def _compute_hotspot_area_trace(
        self, cat_stack: np.ndarray, eps_px: int, window_px: int, density_thresh: float
    ) -> np.ndarray:
        """Density-gated kept-cluster area (µm²) per frame, for the decay-tau fit."""
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

    # -----------------------------------------------------------------------
    # Step 3. Report
    # -----------------------------------------------------------------------

    def _build_clusters(self) -> list[dict]:
        """Critical-frame clusters as {centroid, R_lat_px, R_lat_um} (R = enclosing radius, display only)."""
        clusters = []
        for cluster_k, centroid in enumerate(self.centroids):
            radius_px = cluster_radius(self.label_frame, centroid, cluster_k)
            clusters.append({
                "centroid": centroid,
                "R_lat_px": radius_px,
                "R_lat_um": self._px_to_um(radius_px),
            })
        return clusters

    def _report_frame_clusters(
        self,
        cat_stack: np.ndarray,
        med_stack: np.ndarray,
        spike_frame_idx: int,
        eps_px: int,
        window_px: int,
        density_thresh: float,
    ) -> tuple[np.ndarray, list[dict], np.ndarray | None, list[dict] | None]:
        """Clusters of the spike frame and spike+1 frame, each on its own (spike+1 fields None if out of range)."""
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

    # -----------------------------------------------------------------------
    # Step 4. Flow
    # -----------------------------------------------------------------------

    def compute_flow(self, cat_stack: np.ndarray, med_stack: np.ndarray) -> list[dict]:
        """TV-L1 flow pairs + CAT keep masks around the spike (see functions/hotspot_flow.py); stored as flow_pairs."""
        self.flow_pairs = compute_flow_pairs(med_stack, cat_stack, self.spike_frame_idx)
        return self.flow_pairs

    # -----------------------------------------------------------------------
    # Unit conversion + result accessors
    # -----------------------------------------------------------------------

    def _px_to_um(self, pixels: float) -> float:
        return pixels * self.um_per_pixel

    def _area_to_um2(self, area_px: float) -> float:
        return area_px * (self.um_per_pixel ** 2)

    def get_results(self) -> dict:
        """Critical-frame, spike / spike+1, and decay results (decay fields None when not significant)."""
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
            "clusters":                 self.clusters,
        }

    def get_summary(self) -> dict:
        """Summary for the critical frame."""
        return {
            "obj":         self.obj,
            "n_clusters":  len(self.clusters),
            "has_region":  len(self.clusters) > 0,
            "significant": self.significant,
        }

    def get_lasting_time_ms(self, frame_duration_ms: float) -> float | None:
        """Decay tau in ms, or None when the fit was skipped/failed or R² < MIN_DECAY_FIT_R2."""
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


# ===========================================================================
#
#   STEP 1 -- DETECT: density-gated clustering (shared with SpikeReliabilityChecker)
#
# ===========================================================================


def compute_area_pct(stack: np.ndarray) -> np.ndarray:
    """B%: percentage of bright pixels per frame (diagnostic only)."""
    total_px = stack.shape[1] * stack.shape[2]
    return np.count_nonzero(stack == CATEGORY_BRIGHT, axis=(1, 2)) / total_px * 100


def compute_eps_px(obj: str) -> int:
    """EPS_UM in pixels for this objective (dilation disk radius)."""
    return int(EPS_UM * PIXEL_SCALE[obj])


def compute_window_px(obj: str) -> int:
    """Local-density window size (px) for this objective."""
    return WINDOW_PX_BY_OBJ[obj]


def compute_density_thresh(obj: str) -> float:
    """Min local bright-pixel density for a hotspot, for this objective."""
    return DENSITY_THRESH_BY_OBJ[obj]


def detect_hotspot(
    candidates: list[tuple[int, np.ndarray, np.ndarray]], eps_px: int, window_px: int, density_thresh: float
) -> tuple[bool, int, np.ndarray, list[tuple[float, float]], int]:
    """Earliest candidate frame with an accepted density-gated cluster wins.

    Args:
        candidates: (frame_idx, categorized_frame, intensity_frame) tuples, in the order to try them.

    Returns:
        (detected, frame_idx, label_frame, centroids, n_raw_clusters) of the first detecting
        candidate, or of the first candidate (empty label_frame, no centroids) when none detect.
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


def _run_density_gated_cluster_seeker(
    bright_mask: np.ndarray, eps_px: int, window_px: int, density_thresh: float, z_frame: np.ndarray | None
) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """Cluster only bright pixels in a locally dense neighborhood, then grow clusters back to their full blobs.

    The density gate rejects isolated noise pixels; the regrow step restores a real hotspot's
    ragged low-density edge. Centroids come from the pre-regrow gated pixels.

    Returns:
        (label_frame, centroids, n_raw_clusters) -- same semantics as _run_cluster_seeker.
    """
    # --- 1a. density gate ---
    density = uniform_filter(bright_mask.astype(float), size=window_px, mode="constant", cval=0.0)
    hotspot_mask = bright_mask & (density >= density_thresh)

    # --- 1b. cluster the gated pixels ---
    gated_label_frame, centroids, n_raw = _run_cluster_seeker(hotspot_mask.astype(int), eps_px, z_frame)
    n_clusters = len(centroids)

    # --- 1c. regrow each cluster to the full bright blob(s) it touches ---
    labeled_bright = skimage_label(bright_mask)
    label_frame = np.full(bright_mask.shape, -2, dtype=int)
    for cluster_id in range(n_clusters):
        cluster_mask = gated_label_frame == cluster_id
        touched_blob_ids = set(labeled_bright[cluster_mask].tolist()) - {0}
        whole_mask = np.isin(labeled_bright, list(touched_blob_ids))
        label_frame[whole_mask] = cluster_id

    return label_frame, centroids, n_raw


def _run_cluster_seeker(
    frame: np.ndarray, eps_px: int, z_frame: np.ndarray | None = None
) -> tuple[np.ndarray, list[tuple[float, float]], int]:
    """Cluster bright pixels: dilate by eps_px (distance transform) -> connected components -> size filter.

    Returns:
        label_frame: (H, W); -2 = background, -1 = noise/undersized, 0..N-1 = kept clusters, largest first.
        centroids: (row, col) per kept cluster, same order.
        n_raw_components: component count before the size filter.
    """
    bright_mask = frame == CATEGORY_BRIGHT
    label_frame = np.full(frame.shape, -2, dtype=int)
    if not bright_mask.any():
        return label_frame, [], 0

    # Every pixel within eps_px of a bright pixel -> bright pixels whose eps zones overlap share a component.
    within_eps_of_bright = distance_transform_edt(~bright_mask) <= eps_px
    component_map = skimage_label(within_eps_of_bright, connectivity=2)
    n_raw_components = int(component_map.max())

    total_bright_px = int(bright_mask.sum())
    label_frame[bright_mask] = -1  # noise until promoted to a kept cluster

    # Keep only the original bright pixels of each component; drop components below MIN_CLUSTER_FRACTION.
    accepted_components = []
    for component_id in range(1, n_raw_components + 1):
        bright_px_in_component = (component_map == component_id) & bright_mask
        bright_px_count = int(bright_px_in_component.sum())
        if total_bright_px > 0 and bright_px_count / total_bright_px >= MIN_CLUSTER_FRACTION:
            accepted_components.append((bright_px_count, bright_px_in_component))
    accepted_components.sort(key=lambda component: component[0], reverse=True)

    centroids = []
    for cluster_idx, (_, bright_px_in_component) in enumerate(accepted_components):
        label_frame[bright_px_in_component] = cluster_idx
        bright_px_coords = np.argwhere(bright_px_in_component)
        centroids.append(_weighted_centroid(bright_px_coords[:, 0], bright_px_coords[:, 1], z_frame))

    return label_frame, centroids, n_raw_components


def _weighted_centroid(rows: np.ndarray, cols: np.ndarray, z_frame: np.ndarray | None) -> tuple[float, float]:
    """Intensity-weighted centroid (pulled toward the brightest sub-region); plain mean if no weights."""
    if z_frame is not None:
        weights = np.clip(z_frame[rows, cols].astype(np.float64), 0.0, None)
        if weights.sum() > 0:
            return float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights))
    return float(rows.mean()), float(cols.mean())


# ===========================================================================
#
#   STEP 2 -- AREA: single-exponential decay fit
#
# ===========================================================================


def _decay_model(t: np.ndarray, amplitude: float, tau: float) -> np.ndarray:
    """Single-exponential decay: amplitude * exp(-t/tau)."""
    return amplitude * np.exp(-t / tau)


def fit_decay_tau(signal: np.ndarray, peak_frame_idx: int) -> tuple[float | None, float | None, float | None]:
    """Fit amplitude * exp(-t/tau) from peak_frame_idx onward (t=0 at the peak, falling side only).

    Returns:
        (amplitude, tau_frames, r_squared), all None when there are < MIN_DECAY_FIT_FRAMES
        post-peak frames, the tail is flat (< MIN_DECAY_FIT_RANGE), or curve_fit fails.
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


# ===========================================================================
#
#   STEP 3 -- REPORT: enclosing radius (display only)
#
# ===========================================================================


def cluster_radius(label_frame: np.ndarray, centroid: tuple[float, float], cluster_k: int) -> float:
    """Enclosing-circle radius (px) of one cluster, capped at the centroid's distance to the frame edge."""
    coords = np.argwhere(label_frame == cluster_k)
    row_c, col_c = centroid
    dists = np.sqrt((coords[:, 0] - row_c) ** 2 + (coords[:, 1] - col_c) ** 2)

    height, width = label_frame.shape
    edge_dist = min(row_c, height - 1 - row_c, col_c, width - 1 - col_c)
    return float(min(dists.max(), edge_dist))
