"""
Per-segment reliability check -- does each raw spike segment show its own hotspot?

  Step 1. Check  : categorize spike / spike+1 of every raw segment -> density-gated detection
  Step 2. Export : per-segment detection montage (RELIABILITY.png)
                   + success vs failure Vm with AP thresholds (VM_SUCCESS_FAIL.png)

Example:
    >>> checker = SpikeReliabilityChecker(obj="40X")
    >>> seg_results, reliability_pct = checker.check(lst_segments, spike_frame_idx)
    >>> checker.export_montage(exporter, rec_stem, export_data, animal_idx, slice_val, at,
    ...                        detrend_mode, normalization)
    >>> checker.export_vm_groups(exporter, rec_stem, clip.get_vm_segments(), export_data,
    ...                          animal_idx, slice_val, at, detrend_mode, normalization)
"""

## Modules
# Third-party imports
import numpy as np

# Local imports
from classes.region_analyzer import (
    CATEGORY_BRIGHT,
    compute_density_thresh,
    compute_eps_px,
    compute_window_px,
    detect_hotspot,
)
from classes.results_exporter import ResultsExporter
from classes.spatial_categorization import BASELINE_SIGMA_MULT, SpatialCategorizer
from functions.plot_results import plot_segment_reliability_montage, plot_vm_success_vs_failure


class SpikeReliabilityChecker:
    """Density-gated hotspot check on every raw segment, plus its two reliability exports.

    Cheaper than a full RegionAnalyzer per segment: only the spike and spike+1 frames are
    categorized (SpatialCategorizer.categorize_frame), and RegionAnalyzer's earliest-wins
    detection loop (detect_hotspot) is reused as a plain function.

    Results after check():
        seg_results, reliability_pct
    """

    def __init__(self, obj: str) -> None:
        self.obj = obj
        self.eps_px = compute_eps_px(obj)
        self.window_px = compute_window_px(obj)
        self.density_thresh = compute_density_thresh(obj)
        self.seg_results: list[dict] = []
        self.reliability_pct: float = 0.0

    # -----------------------------------------------------------------------
    # Step 1. Check
    # -----------------------------------------------------------------------

    def check(self, lst_segments: list[np.ndarray], spike_frame_idx: int) -> tuple[list[dict], float]:
        """Per-segment density-gated hotspot detection.

        Args:
            lst_segments: per-spike raw (detrended, unnormalized) segments, from load_img_segs().
            spike_frame_idx: spike frame index within each segment (same for all segments).

        Returns:
            (seg_results, reliability_pct) -- one dict per segment with "detected", "frame_offset"
            (0 = spike, 1 = spike+1), "bright_mask", "label_frame", "centroids", "n_clusters";
            and the % of segments detected. Both are also stored on self.
        """
        seg_results: list[dict] = []
        for segment in lst_segments:
            # 1a. baseline threshold from this segment's own pre-spike frames
            threshold = SpatialCategorizer.compute_baseline_threshold(segment[:spike_frame_idx])
            categorizer = SpatialCategorizer.morphological(threshold_method="baseline_n_sigma")

            # 1b. categorize the two candidate frames (spike, spike+1)
            candidates = []
            for idx in (spike_frame_idx, spike_frame_idx + 1):
                if idx >= segment.shape[0]:
                    continue
                cat_frame = categorizer.categorize_frame(segment[idx], idx, threshold)
                candidates.append((idx, cat_frame, segment[idx]))

            # 1c. earliest frame passing the density gate wins
            detected, frame_idx, label_frame, centroids, _ = detect_hotspot(
                candidates, self.eps_px, self.window_px, self.density_thresh
            )
            winning_cat_frame = next(cat_frame for idx, cat_frame, _ in candidates if idx == frame_idx)
            seg_results.append({
                "detected": detected,
                "frame_offset": frame_idx - spike_frame_idx,
                "bright_mask": winning_cat_frame == CATEGORY_BRIGHT,
                "label_frame": label_frame,
                "centroids": centroids,
                "n_clusters": len(centroids),
            })

        n_total = len(seg_results)
        n_detected = sum(r["detected"] for r in seg_results)
        reliability_pct = 100.0 * n_detected / n_total if n_total else 0.0

        self.seg_results = seg_results
        self.reliability_pct = reliability_pct
        return seg_results, reliability_pct

    # -----------------------------------------------------------------------
    # Step 2. Export (both need only self.seg_results -- run right after check())
    # -----------------------------------------------------------------------

    def export_montage(
        self,
        exporter: ResultsExporter,
        rec_stem: str,
        export_data: dict,
        animal_idx: int,
        slice_val: str,
        at: str,
        detrend_mode: str,
        normalization: str,
    ) -> None:
        """reliability/{stem}_RELIABILITY.png -- one panel per segment.

        More than MONTAGE_PAGE_SIZE segments split into RELIABILITY_P1.png, _P2.png, ...
        """
        figures = plot_segment_reliability_montage(
            self.seg_results, rec_stem, self.window_px, self.density_thresh, BASELINE_SIGMA_MULT, self.obj
        )
        n_pages = len(figures)
        for page_idx, fig in enumerate(figures):
            file_type = "RELIABILITY" if n_pages == 1 else f"RELIABILITY_P{page_idx + 1}"
            stem = ResultsExporter.build_export_stem(
                export_data["exp_date"], export_data["img_serial"], animal_idx,
                slice_val, at, detrend_mode, normalization, file_type,
            )
            exporter.export_figure("reliability", fig, f"{stem}.png")

    def export_vm_groups(
        self,
        exporter: ResultsExporter,
        rec_stem: str,
        vm_segments: list[tuple[np.ndarray, np.ndarray]],
        export_data: dict,
        animal_idx: int,
        slice_val: str,
        at: str,
        detrend_mode: str,
        normalization: str,
    ) -> None:
        """reliability/{stem}_VM_SUCCESS_FAIL.png -- Vm of detected vs failed segments, AP thresholds marked.

        Args:
            vm_segments: per-segment (time_ms, Vm) pairs from AbfClip.get_vm_segments(),
                in the same order as self.seg_results.
        """
        vm_success = [vm for r, vm in zip(self.seg_results, vm_segments, strict=True) if r["detected"]]
        vm_failure = [vm for r, vm in zip(self.seg_results, vm_segments, strict=True) if not r["detected"]]

        stem = ResultsExporter.build_export_stem(
            export_data["exp_date"], export_data["img_serial"], animal_idx,
            slice_val, at, detrend_mode, normalization, "VM_SUCCESS_FAIL",
        )
        fig = plot_vm_success_vs_failure(vm_success, vm_failure, rec_stem)
        exporter.export_figure("reliability", fig, f"{stem}.png")
