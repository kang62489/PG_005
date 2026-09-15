"""Per-segment reliability check -- does each raw spike segment show a real hotspot?"""

import numpy as np

from classes.region_analyzer import (
    CATEGORY_BRIGHT,
    compute_density_thresh,
    compute_eps_px,
    compute_window_px,
    detect_hotspot,
)
from classes.results_exporter import ResultsExporter
from classes.spatial_categorization import BASELINE_SIGMA_MULT, SpatialCategorizer
from functions.plot_results import plot_segment_reliability_montage


class SpikeReliabilityChecker:
    """Checks each raw segment for its own density-gated hotspot, cheaper than a full
    RegionAnalyzer per segment, and exports the resulting per-segment montage.

    Every raw segment has the same frame count as the final median (guaranteed by AbfClip's
    frame-range construction), so a naive per-segment RegionAnalyzer would cost roughly as much
    as the median-only categorization -- times the segment count. check() only needs a yes/no per
    segment, so it categorizes just the spike and spike+1 frames (SpatialCategorizer.
    categorize_frame) instead of the whole segment, and reuses RegionAnalyzer's own earliest-wins
    detection loop (detect_hotspot) directly -- it imports these as plain functions rather than
    constructing full SpatialCategorizer/RegionAnalyzer instances, to keep that speed benefit.

    Example:
        >>> checker = SpikeReliabilityChecker(obj="40X")
        >>> seg_results, reliability_pct = checker.check(lst_segments, spike_frame_idx)
        >>> checker.export_montage(exporter, rec_stem, export_data, animal_idx, slice_val, at,
        ...                        detrend_mode, normalization)
    """

    def __init__(self, obj: str) -> None:
        self.obj = obj
        self.eps_px = compute_eps_px(obj)
        self.window_px = compute_window_px(obj)
        self.density_thresh = compute_density_thresh(obj)
        self.seg_results: list[dict] = []
        self.reliability_pct: float = 0.0

    def check(self, lst_segments: list[np.ndarray], spike_frame_idx: int) -> tuple[list[dict], float]:
        """Per-segment density-gated hotspot detection.

        Args:
            lst_segments: per-spike raw (detrended, unnormalized) segments, from load_img_segs().
            spike_frame_idx: index of the spike frame within each segment (same for every segment
                in a recording, by construction of AbfClip's frame ranges).

        Returns:
            (seg_results, reliability_pct) -- one dict per segment: {"detected", "frame_offset"
            (0 for spike, 1 for spike+1), "bright_mask", "label_frame", "centroids", "n_clusters"}
            (enough to plot with export_montage()), and the percentage of segments detected overall.
            Also stored on self for export_montage() to use.
        """
        seg_results: list[dict] = []
        for segment in lst_segments:
            threshold = SpatialCategorizer.compute_baseline_threshold(segment[:spike_frame_idx])
            categorizer = SpatialCategorizer.morphological(threshold_method="baseline_n_sigma")

            candidates = []
            for idx in (spike_frame_idx, spike_frame_idx + 1):
                if idx >= segment.shape[0]:
                    continue
                cat_frame = categorizer.categorize_frame(segment[idx], idx, threshold)
                candidates.append((idx, cat_frame, segment[idx]))

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
        """Build and export the per-segment reliability montage PNG(s).

        Only needs self.seg_results (from check()) plus filename-naming lookups -- no dependency
        on the median/categorize/region-analysis steps, so this can (and should) run right after
        check(), not after them. More than MONTAGE_PAGE_SIZE segments split into multiple PNGs
        (RELIABILITY_P1.png, RELIABILITY_P2.png, ...) instead of one increasingly-tall image.
        """
        figures = plot_segment_reliability_montage(
            self.seg_results, rec_stem, self.window_px, self.density_thresh, BASELINE_SIGMA_MULT
        )
        n_pages = len(figures)
        for page_idx, fig in enumerate(figures):
            file_type = "RELIABILITY" if n_pages == 1 else f"RELIABILITY_P{page_idx + 1}"
            stem = ResultsExporter.build_export_stem(
                export_data["exp_date"], export_data["img_serial"], animal_idx,
                slice_val, at, detrend_mode, normalization, file_type,
            )
            exporter.export_figure("reliability", fig, f"{stem}.png")
