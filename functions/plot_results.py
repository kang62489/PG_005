"""Static, headless matplotlib export figures for spike-aligned region analysis.

No PySide6 dependency — these build plain Figure objects for fig.savefig()/
ResultsExporter.export_figure(), not interactive GUI windows. See classes/mpl_canvas.py
for the PySide6-coupled canvas widget used by the live GUI.

Four export figures, mirroring the validated demo (archive/_demo_dbscan_tmp.py):
- plot_spatiotemporal_summary (-> spatial/): density-gated hotspot-area trace showing
  why the critical frame was picked, + cluster shading on the spike and spike+1 panels.
- plot_full_trace (-> latency/, *_LATENCY.png): the same fixed cluster-ring overlay repeated
  across a 9-panel window, + the full-segment intensity trace with that window annotated.
- plot_segment_reliability_montage (-> reliability/, *_RELIABILITY.png): one panel per raw
  segment showing its own density-gated detection result, for reviewing reliability by eye.
- plot_spike_detection_summary (-> spikes/, *_spike_analysis.png): full-length Vm trace with
  picked/skipped/collapsed spikes marked, replacing AbfClip's old per-recording CSV exports.
"""

## Modules
# Standard library imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes.spatial_categorization import SpatialCategorizer

# Third-party imports
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from classes.region_analyzer import (
    MIN_DECAY_FIT_R2,
    RegionAnalyzer,
    _decay_model,
    _peak_offset_from_spike,
)

# Cluster fill/outline colors, cycled by cluster index (red, green, blue, orange, purple)
CLUSTER_RGBA = [
    (0.91, 0.30, 0.24, 0.45),
    (0.18, 0.80, 0.44, 0.45),
    (0.20, 0.60, 0.86, 0.45),
    (0.95, 0.61, 0.07, 0.45),
    (0.61, 0.35, 0.71, 0.45),
]

# Ring colors for the 1-cluster case -- match _plot_trace_panel's inner/outer trace colors
# ("#e74c3c" / "#3498db") so the panel overlay and the trace line below read as the same signal.
INNER_RGB = (0.906, 0.298, 0.235)
OUTER_RGB = (0.204, 0.596, 0.859)

# ── Static export figures ───────────────────────────────────────────────────


def _add_scale_bar(
    pixel_size_um: float,
    ax: mpl.axes.Axes,
    img_width: int,
    img_height: int,
    font_size: int | None = None,
    bar_height: int | None = None,
) -> None:
    """Add a scale bar to the axes

    Args:
        pixel_size_um: Pixel size in microns
        ax: Matplotlib axes
        img_width: Image width in pixels
        img_height: Image height in pixels
        font_size: Font size for label (default: auto-scaled based on image size)
        bar_height: Height of scale bar in pixels (default: auto-scaled based on image size)
    """
    # Calculate a nice scale bar length (aim for ~20% of image width)
    image_width_um = img_width * pixel_size_um
    target_length_um = image_width_um * 0.2

    # Round to nice values: include small values for cropped images
    nice_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    scale_bar_um = min(nice_values, key=lambda x: abs(x - target_length_um))
    scale_bar_px = scale_bar_um / pixel_size_um

    # Scale padding based on image size
    padding = max(2, int(img_width * 0.03))

    # Use provided bar_height or auto-scale
    if bar_height is None:
        bar_height = max(2, int(img_height * 0.015))

    # Position: bottom-right corner with padding
    x_pos = img_width - scale_bar_px - padding
    y_pos = img_height - padding - bar_height

    # Draw scale bar rectangle
    rect = Rectangle((x_pos, y_pos), scale_bar_px, bar_height, linewidth=0, edgecolor=None, facecolor="lime", zorder=15)
    ax.add_patch(rect)

    # Use provided font_size or auto-scale
    if font_size is None:
        font_size = max(6, min(10, int(img_width * 0.06)))

    ax.text(
        x_pos + scale_bar_px / 2,
        y_pos - 1,
        f"{scale_bar_um} µm",
        color="lime",
        fontsize=font_size,
        weight="bold",
        ha="center",
        va="bottom",
        zorder=15,
    )


def plot_spatiotemporal_summary(
    categorizer: "SpatialCategorizer",
    region_analyzer: RegionAnalyzer,
    spike_frame_idx: int,
    title_info: dict,
    vm_segments: list[tuple[np.ndarray, np.ndarray]],
    frame_duration_ms: float,
) -> Figure:
    """Static export figure: density-gated hotspot-area signal trace (row 1) +
    spike/spike+1 frame panels (row 2) + overlapped electrophysiology traces per
    segment (row 3).

    Shows why the critical frame was picked (hotspot area vs the spike/spike+1
    candidates) and what was found on each of those two frames independently --
    cluster shading is drawn on both the spike and spike+1 panels, not repeated
    across every panel (see plot_full_trace for the fixed-overlay version).
    Row 3 shows the actual detected Vm spike waveform
    for every picked segment in this recording, overlaid, to check shape/timing
    consistency across trials -- independent of the image data above it.

    Args:
        categorizer: fitted SpatialCategorizer (source_frames + categorized_frames)
        region_analyzer: RegionAnalyzer built from the segment
        spike_frame_idx: index of the spike frame within the segment
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial"
        vm_segments: per-segment (time_ms_relative_to_spike_frame, Vm) pairs, from AbfClip.get_vm_segments()
        frame_duration_ms: milliseconds per frame (e.g. AbfClip.ts_imgs * 1000), for the row-2 frame gridlines

    Returns:
        Figure, ready for fig.savefig(...) or ResultsExporter.export_figure(...)
    """
    n_frames = len(categorizer.source_frames)
    hotspot_area_um2 = region_analyzer.hotspot_area_um2
    critical_frame_idx = region_analyzer.critical_frame_idx
    um_per_pixel = region_analyzer.um_per_pixel

    # --- Figure and grid setup ---
    fig = Figure(figsize=(26, 13), dpi=100)
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.5, 2.0], hspace=0.7)
    gs_panels = gs_outer[1].subgridspec(1, 9, wspace=0.08)

    # --- Row 0: density-gated hotspot area per frame trace ---
    # Plotted in the same units the decay-tau fit below was actually fit on
    # (hotspot_area_um2), not the raw B% trace -- see RegionAnalyzer._compute_hotspot_area_trace.
    ax_bd = fig.add_subplot(gs_outer[0])
    ax_bd.plot(np.arange(n_frames) - spike_frame_idx, hotspot_area_um2, color="#3498db", linewidth=1.6,
               marker="o", markersize=3.5)

    for frame_idx, label, color in [
        (spike_frame_idx - 1,
         f"spike-1: {hotspot_area_um2[spike_frame_idx - 1]:.0f} µm²"
         if spike_frame_idx > 0 else "spike-1 (OOB)", "#888888"),
        (spike_frame_idx,
         f"spike: {hotspot_area_um2[spike_frame_idx]:.0f} µm²", "#e74c3c"),
        (spike_frame_idx + 1,
         f"spike+1: {hotspot_area_um2[spike_frame_idx + 1]:.0f} µm²"
         if spike_frame_idx + 1 < n_frames else "spike+1 (OOB)", "#f39c12"),
    ]:
        if 0 <= frame_idx < n_frames:
            ax_bd.axvline(frame_idx - spike_frame_idx, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)

    ax_bd.plot(critical_frame_idx - spike_frame_idx, hotspot_area_um2[critical_frame_idx], "*", color="white", markersize=14,
               markeredgecolor="black", markeredgewidth=1, zorder=5,
               label=f"critical: frame {critical_frame_idx}  {hotspot_area_um2[critical_frame_idx]:.0f} µm²")

    _draw_decay_fit(ax_bd, region_analyzer, spike_frame_idx, n_frames, frame_duration_ms)

    ax_bd.set_xlabel("Frame offset from spike (0 = spike)", fontsize=12)
    ax_bd.set_ylabel("Hotspot area (µm²)", fontsize=12)
    ax_bd.set_title("Density-gated hotspot area per frame  |  star = critical frame (spike or spike+1)", fontsize=12)
    ax_bd.legend(fontsize=10, loc="upper right")
    ax_bd.tick_params(labelsize=10)

    # --- Row 1: spike-4 .. spike+4 frame panels ---
    def _label_frame_for(idx: int) -> np.ndarray | None:
        if idx == spike_frame_idx:
            return region_analyzer.spike_frame_label_frame
        if idx == spike_frame_idx + 1:
            return region_analyzer.spike_plus1_frame_label_frame
        return None

    hotspot_area_lines = {
        idx: _format_hotspot_area_line(um_per_pixel, _label_frame_for(idx))
        for idx in range(spike_frame_idx - 4, spike_frame_idx + 5)
        if 0 <= idx < n_frames
    }

    for col, offset in enumerate(range(-4, 5)):
        frame_idx = spike_frame_idx + offset
        ax_frame = fig.add_subplot(gs_panels[0, col])
        if 0 <= frame_idx < n_frames:
            is_critical_frame = frame_idx == critical_frame_idx
            tag = "  ★ critical" if is_critical_frame else ""
            _plot_frame_panel(
                ax_frame,
                categorizer,
                frame_idx,
                offset,
                um_per_pixel,
                tag,
                hotspot_area_line=hotspot_area_lines.get(frame_idx),
            )
            # Both the spike frame and spike+1 frame get their own independent cluster
            # shading now -- there's no single "max-area frame" winner to pick between.
            if offset == 0:
                _draw_cluster_shading(
                    ax_frame, region_analyzer.spike_frame_label_frame,
                    [c["centroid"] for c in region_analyzer.spike_frame_clusters],
                )
            elif offset == 1 and region_analyzer.spike_plus1_frame_label_frame is not None:
                _draw_cluster_shading(
                    ax_frame, region_analyzer.spike_plus1_frame_label_frame,
                    [c["centroid"] for c in region_analyzer.spike_plus1_frame_clusters],
                )
        else:
            frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
            ax_frame.set_title(f"{frame_label}\n(out of range)", fontsize=9)
            ax_frame.axis("off")

    # --- Row 2: overlapped electrophysiology (Vm) traces ---
    ax_vm = fig.add_subplot(gs_outer[2])
    for time_ms, vm in vm_segments:
        ax_vm.plot(time_ms, vm, color="#3498db", linewidth=0.8, alpha=0.5)

    for offset in range(-4, 6):
        color = "#e74c3c" if offset == 0 else "#888888"
        label = "spike frame" if offset == 0 else None
        ax_vm.axvline(offset * frame_duration_ms, color=color, linestyle="--", linewidth=1.0, alpha=0.7, label=label)

    ax_vm.set_xlim(-4 * frame_duration_ms, 5 * frame_duration_ms)
    ax_vm.set_xlabel("Time relative to spike frame (ms)", fontsize=12)
    ax_vm.set_ylabel("Vm (mV)", fontsize=12)
    ax_vm.set_title(f"Overlapped spike waveforms — {len(vm_segments)} segment(s)", fontsize=12)
    ax_vm.legend(fontsize=10, loc="upper right")
    ax_vm.tick_params(labelsize=10)

    # --- Figure title ---
    title = (
        f"Spatial Analysis: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig


def plot_full_trace(
    region_analyzer: RegionAnalyzer,
    categorizer: "SpatialCategorizer",
    median_segment: np.ndarray,
    spike_frame_idx: int,
    frame_duration_ms: float,
    title_info: dict,
) -> Figure:
    """Standalone export figure: fixed cluster-ring overlay (row 1, 9 panels) +
    full-segment per-cluster intensity traces with that window annotated (row 2).

    Every panel (spike-4..spike+4) shows the same fixed critical-frame cluster
    overlay, so you can see how the underlying pixel pattern moves/changes under
    it; the trace row spans the whole segment (never cropped), with a shaded band
    marking which x-range the panels above cover.

    Args:
        region_analyzer: RegionAnalyzer built from the segment
        categorizer: fitted SpatialCategorizer (source_frames + categorized_frames)
        median_segment: 3D raw-intensity segment (frames, height, width)
        spike_frame_idx: index of the spike frame within the segment
        frame_duration_ms: milliseconds per frame (e.g. AbfClip.ts_imgs * 1000)
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial"

    Returns:
        Figure, ready for fig.savefig(...) or ResultsExporter.export_figure(...)
    """
    base_title = (
        f"Full Temporal Trace: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
    )

    clusters = region_analyzer.clusters
    if not clusters:
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No cluster detected — ring analysis skipped",
                ha="center", va="center", fontsize=12, color="#888888", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(base_title, fontsize=15, fontweight="bold")
        return fig

    n_frames = median_segment.shape[0]
    um_per_pixel = region_analyzer.um_per_pixel
    highlight = _highlight_clusters(clusters, spike_frame_idx)

    fig = Figure(figsize=(22, 8.5), dpi=100)
    gs = fig.add_gridspec(2, 9, height_ratios=[2.2, 1.6], hspace=0.45, wspace=0.08)

    # --- Row 0: spike-4 .. spike+4 panels, fixed cluster overlay on every panel ---
    for col, offset in enumerate(range(-4, 5)):
        frame_idx = spike_frame_idx + offset
        ax = fig.add_subplot(gs[0, col])
        if 0 <= frame_idx < n_frames:
            tag = "  [critical frame]" if frame_idx == region_analyzer.critical_frame_idx else ""
            _plot_frame_panel(ax, categorizer, frame_idx, offset, um_per_pixel, tag,
                               stats_lines=_frame_z_lines(clusters, frame_idx))
            _overlay_clusters(ax, clusters, highlight)
        else:
            frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
            ax.set_title(f"{frame_label}\n(out of range)", fontsize=9)
            ax.axis("off")

    # --- Row 1: full-segment intensity traces, with the 9-panel window annotated ---
    ax_trace = fig.add_subplot(gs[1, :])
    _plot_trace_panel(ax_trace, region_analyzer, median_segment, spike_frame_idx, frame_duration_ms, highlight)

    window_lo = max(0, spike_frame_idx - 4) - spike_frame_idx
    window_hi = min(n_frames - 1, spike_frame_idx + 4) - spike_frame_idx
    ax_trace.axvspan(window_lo, window_hi, color="#f1c40f", alpha=0.12, label="panels shown above")
    for frame_offset, color, label in [(-1, "#888888", "spike-1"), (0, "#e74c3c", "spike"), (1, "#f39c12", "spike+1")]:
        if 0 <= spike_frame_idx + frame_offset < n_frames:
            ax_trace.axvline(frame_offset, color=color, linestyle=":", linewidth=1.0, alpha=0.6, label=label)
    ax_trace.legend(loc="upper right", fontsize=10, ncol=2)

    fig.suptitle(base_title + _format_r_lat_suffix(clusters), fontsize=15, fontweight="bold")
    return fig


MONTAGE_NCOLS = 8
MONTAGE_MAX_ROWS = 4
MONTAGE_PAGE_SIZE = MONTAGE_NCOLS * MONTAGE_MAX_ROWS  # segments per PNG, before starting a new one


def plot_segment_reliability_montage(
    seg_results: list[dict],
    rec_stem: str,
    window_px: int,
    density_thresh: float,
    sigma_mult: float,
) -> list[Figure]:
    """Grid of per-segment density-gated detection panels, for reviewing reliability by eye.

    One panel per raw (un-merged) segment: its winning frame's bright mask in gray, cluster
    shading on top, a green (detected) or red (not detected) border, and which frame won
    (spike vs spike+1). Reliability% (computed over all segments, not just the current page)
    is reported in every page's suptitle.

    Capped at MONTAGE_MAX_ROWS x MONTAGE_NCOLS (4x8 = 32) panels per figure -- a recording with
    more segments than that gets multiple figures ("pages") instead of one increasingly-tall PNG.

    Args:
        seg_results: per-segment dicts from SpikeReliabilityChecker.check(), each with
            "detected", "frame_offset", "bright_mask", "label_frame", "centroids", "n_clusters".
        rec_stem: recording name, for the title.
        window_px: density window size used (see compute_window_px()), for the title.
        density_thresh: density threshold used (see compute_density_thresh()), for the title.
        sigma_mult: bright-pixel threshold's sigma multiplier (see BASELINE_SIGMA_MULT), for the title.

    Returns:
        One Figure per page (single-element list when segments fit on one page), each ready
        for fig.savefig(...) or ResultsExporter.export_figure(...).
    """
    n = len(seg_results)
    n_detected = sum(r["detected"] for r in seg_results)
    reliability_line = f"reliability: {n_detected}/{n} = {n_detected / n:.1%}" if n else "no segments"

    if n == 0:
        fig = Figure(figsize=(MONTAGE_NCOLS * 2.0, 2.0), dpi=130)
        fig.suptitle(f"{rec_stem} — no segments", fontsize=11)
        return [fig]

    page_starts = list(range(0, n, MONTAGE_PAGE_SIZE))
    n_pages = len(page_starts)
    figures = []

    for page_idx, page_start in enumerate(page_starts):
        page_results = seg_results[page_start : page_start + MONTAGE_PAGE_SIZE]
        n_page = len(page_results)
        nrows = int(np.ceil(n_page / MONTAGE_NCOLS))
        fig = Figure(figsize=(MONTAGE_NCOLS * 2.0, nrows * 2.0), dpi=130)
        axes = np.atleast_1d(fig.subplots(nrows, MONTAGE_NCOLS)).flatten()

        for i, result in enumerate(page_results):
            seg_idx = page_start + i
            ax = axes[i]
            ax.imshow(result["bright_mask"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")

            label_frame = result["label_frame"]
            overlay = np.zeros((*label_frame.shape, 4))
            for cluster_idx in range(result["n_clusters"]):
                overlay[label_frame == cluster_idx] = CLUSTER_RGBA[cluster_idx % len(CLUSTER_RGBA)]
            ax.imshow(overlay, interpolation="nearest")

            color = "limegreen" if result["detected"] else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            frame_tag = f"sp+{result['frame_offset']}" if result["frame_offset"] else "sp"
            ax.set_title(
                f"seg{seg_idx:02d} [{frame_tag}] n_clusters={result['n_clusters']}", fontsize=7, color=color
            )

        for j in range(n_page, len(axes)):
            axes[j].axis("off")

        page_tag = f"  (page {page_idx + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(
            f"{rec_stem} — window_px={window_px}, density>={density_thresh}, threshold={sigma_mult:g}σ{page_tag}\n"
            f"{reliability_line}",
            fontsize=11,
        )
        fig.tight_layout()
        figures.append(fig)

    return figures


def _plot_frame_panel(
    ax: mpl.axes.Axes,
    categorizer: "SpatialCategorizer",
    frame_idx: int,
    offset: int,
    um_per_pixel: float,
    tag: str = "",
    hotspot_area_line: str | None = None,
    stats_lines: list[str] | None = None,
) -> None:
    """One frame's categorized image with a stats title.

    Defaults to a hotspot area line (used by plot_spatiotemporal_summary,
    where row 0 above is the density-gated hotspot-area trace). Callers whose
    companion trace row plots something else (e.g. plot_full_trace's intensity
    ring trace) should pass stats_lines to show a title relevant to that instead.

    No cluster overlay is drawn here -- callers layer that on top afterward
    (see _draw_cluster_shading / _overlay_clusters), since the two export
    figures use different overlay styles on different subsets of panels.
    """
    cat_frame = categorizer.categorized_frames[frame_idx]

    # --- Image ---
    cmap_cat = ListedColormap(["black", "white"])
    ax.imshow(cat_frame, cmap=cmap_cat, vmin=0, vmax=1, interpolation="nearest")
    # Pin view so cluster circles/overlays added later get clipped, not rescaled.
    ax.set_xlim(0, cat_frame.shape[1])
    ax.set_ylim(cat_frame.shape[0], 0)
    ax.set_autoscale_on(False)

    # --- Stats lines ---
    if stats_lines is None:
        stats_lines = []
        if hotspot_area_line is not None:
            stats_lines.append(hotspot_area_line)

    # --- Title and decorations ---
    frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
    ax.set_title(
        "\n".join([f"{frame_label}{tag}", *stats_lines]),
        fontsize=10,
        fontweight="bold" if offset == 0 else "normal",
        color="red" if offset == 0 else "black",
        pad=3,
    )
    ax.axis("off")
    _add_scale_bar(um_per_pixel, ax, cat_frame.shape[1], cat_frame.shape[0], font_size=7)



def _format_hotspot_area_line(um_per_pixel: float, label_frame: np.ndarray | None) -> str | None:
    """Density-gated hotspot area line for the spike/spike+1 panel titles.

    Only the spike and spike+1 panels have a density-gated label_frame to report on; other
    panels (spike-4..spike-2, spike+2..spike+4) get no stats line at all -- there's no
    B%-style fallback anymore, since that's exactly the raw-threshold metric this pipeline
    moved away from.
    """
    if label_frame is None:
        return None
    hotspot_px = np.count_nonzero(label_frame >= 0)
    hotspot_um2 = hotspot_px * um_per_pixel ** 2
    hotspot_pct = 100.0 * hotspot_px / label_frame.size
    return f"hotspots: {hotspot_um2:.0f} µm² ({hotspot_pct:.1f}%)"


def _frame_z_lines(clusters: list[dict], frame_idx: int) -> list[str]:
    """Per-frame intensity line(s) for a plot_full_trace panel title.

    Mirrors _plot_trace_panel's split so the panel title and the trace row
    below refer to the same numbers: 1 cluster -> inner/outer ring intensity at
    this frame; >1 clusters -> each cluster's whole-cluster intensity.
    """
    if len(clusters) == 1:
        cluster = clusters[0]
        return [
            f"inner z={cluster['inner_trace'][frame_idx]:.2f}",
            f"outer z={cluster['outer_trace'][frame_idx]:.2f}",
        ]
    return [f"c{i} z={cluster['trace'][frame_idx]:.2f}" for i, cluster in enumerate(clusters)]


def _format_r_lat_suffix(clusters: list[dict]) -> str:
    """R_lat suptitle suffix for plot_full_trace -- one value for 1 cluster, per-cluster list otherwise."""
    if len(clusters) == 1:
        return f"  |  R_lat = {clusters[0]['R_lat_um']:.1f} µm"
    parts = ", ".join(f"c{i}={cluster['R_lat_um']:.1f}" for i, cluster in enumerate(clusters))
    return f"  |  R_lat: {parts} µm"


def _draw_cluster_shading(ax: mpl.axes.Axes, label_frame: np.ndarray, centroids: list[tuple[float, float]]) -> None:
    """Translucent per-cluster fill (DBSCAN's raw label map, no ring circles) +
    centroid cross and index label.

    Used on the spike frame's and spike+1 frame's own panels in
    plot_spatiotemporal_summary (each with its own independent label_frame/centroids --
    there's no single "critical"/"max-area" frame winner for this overlay anymore).
    plot_full_trace uses _overlay_clusters (enclosing-circle approximation)
    instead, repeated identically across every panel.
    """
    height, width = label_frame.shape
    overlay = np.zeros((height, width, 4), dtype=float)
    for cluster_idx in range(len(centroids)):
        r, g, b, a = CLUSTER_RGBA[cluster_idx % len(CLUSTER_RGBA)]
        overlay[label_frame == cluster_idx] = (r, g, b, a)
    ax.imshow(overlay, interpolation="nearest")

    for cluster_idx, (row_c, col_c) in enumerate(centroids):
        ax.plot(col_c, row_c, "x", color="black", markersize=16, markeredgewidth=4, zorder=10)
        ax.plot(col_c, row_c, "x", color="white", markersize=14, markeredgewidth=2.5, zorder=11)
        ax.text(col_c + 5, row_c - 5, str(cluster_idx), color="white", fontsize=9, fontweight="bold")


def _draw_decay_fit(
    ax: mpl.axes.Axes,
    region_analyzer: "RegionAnalyzer",
    spike_frame_idx: int,
    n_frames: int,
    frame_duration_ms: float,
) -> None:
    """Dashed exponential decay curve over the post-peak B+D% trace, with tau/R² in the label.

    Draws nothing but a "fit failed" note if fit_decay_tau() couldn't fit the
    post-peak trace (too few post-peak frames, a flat/degenerate tail, or
    curve_fit not converging -- see RegionAnalyzer.__init__).
    """
    peak_frame_idx = region_analyzer.decay_peak_frame_idx
    tau = region_analyzer.decay_tau_frames
    amplitude = region_analyzer.decay_fit_A
    r_squared = region_analyzer.decay_fit_r2

    if tau is None or amplitude is None:
        ax.text(
            0.99, 0.5, "decay fit: failed / insufficient post-peak data",
            transform=ax.transAxes, ha="right", va="center", fontsize=9, color="#888888", style="italic",
        )
        return

    t = np.arange(0, n_frames - peak_frame_idx, dtype=np.float64)
    fitted = _decay_model(t, amplitude, tau)
    tau_ms = tau * frame_duration_ms
    r2_text = f"{r_squared:.2f}" if r_squared is not None else "n/a"
    rejected = r_squared is None or r_squared < MIN_DECAY_FIT_R2
    color = "#e67e22" if rejected else "#2ecc71"
    label = f"decay fit: τ={tau_ms:.0f} ms (R²={r2_text})"
    if rejected:
        label += " [rejected, lasting time=None]"
    ax.plot(
        t + (peak_frame_idx - spike_frame_idx), fitted, "--", color=color, linewidth=1.8, zorder=4,
        label=label,
    )


def _overlay_clusters(ax: mpl.axes.Axes, clusters: list[dict], highlight: set[int]) -> None:
    """Translucent cluster fill + ring/circle outlines.

    1 cluster -> inner (dashed) + outer (solid) ring pair at R/sqrt(2) and R,
    shaded in the same red/blue as _plot_trace_panel's inner/outer trace lines
    (matching compute_ring_traces' split), so the panel overlay and the trace
    row below read as the same signal. >1 clusters -> a single solid circle at
    R per cluster, colored by cluster index, no ring split (matching
    compute_cluster_trace).
    """
    if not clusters:
        return

    is_single = "inner_mask" in clusters[0]
    mask_shape = clusters[0]["inner_mask"].shape if is_single else clusters[0]["mask"].shape
    overlay = np.zeros((*mask_shape, 4), dtype=float)
    for i, cluster in enumerate(clusters):
        if is_single:
            overlay[cluster["inner_mask"]] = (*INNER_RGB, 0.45)
            overlay[cluster["outer_mask"]] = (*OUTER_RGB, 0.45)
        else:
            r, g, b, _ = CLUSTER_RGBA[i % len(CLUSTER_RGBA)]
            overlay[cluster["mask"]] = (r, g, b, 0.4)
    ax.imshow(overlay, interpolation="nearest")

    for i, cluster in enumerate(clusters):
        row_c, col_c = cluster["centroid"]
        radius_px = cluster["R_lat_px"]
        line_width = 1.8 if i in highlight else 1.0
        if is_single:
            rings = [(radius_px / np.sqrt(2), "--", INNER_RGB), (radius_px, "-", OUTER_RGB)]
            marker_color = OUTER_RGB
        else:
            edge_color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
            rings = [(radius_px, "-", edge_color)]
            marker_color = edge_color
        for ring_radius, linestyle, ring_color in rings:
            circle = Circle((col_c, row_c), ring_radius, fill=False, edgecolor=ring_color,
                             linewidth=line_width, linestyle=linestyle)
            ax.add_patch(circle)
        ax.plot(col_c, row_c, "x", color="black", markersize=12, markeredgewidth=2.5, zorder=10)
        ax.plot(col_c, row_c, "x", color=marker_color, markersize=10, markeredgewidth=1.5, zorder=11)


def _highlight_clusters(clusters: list[dict], spike_frame_idx: int) -> set[int]:
    """Cluster indices to visually emphasize: the lone cluster, or the earliest/latest-peaking pair."""
    if len(clusters) == 1:
        return {0}
    peak_rels = [(i, _peak_offset_from_spike(c["trace"], spike_frame_idx)) for i, c in enumerate(clusters)]
    valid = [(i, peak) for i, peak in peak_rels if peak is not None]
    if len(valid) < 2:
        return set()
    earliest_i, _ = min(valid, key=lambda pair: pair[1])
    latest_i, _ = max(valid, key=lambda pair: pair[1])
    return {earliest_i, latest_i}


def _plot_trace_panel(
    ax: mpl.axes.Axes,
    region_analyzer: RegionAnalyzer,
    median_segment: np.ndarray,
    spike_frame_idx: int,
    frame_duration_ms: float,
    highlight: set[int],
) -> None:
    """Per-cluster intensity traces across the full segment (never cropped -- the
    caller draws a shaded window annotation and calls legend() on top of this)."""
    clusters = region_analyzer.clusters
    latency_ms = region_analyzer.get_peak_latency_ms(frame_duration_ms)
    latency_label = f"{latency_ms:.1f} ms" if latency_ms is not None else "n/a"

    n_frames = median_segment.shape[0]
    x = np.arange(n_frames) - spike_frame_idx

    if len(clusters) == 1:
        cluster = clusters[0]
        split_um = cluster["R_lat_um"] / np.sqrt(2)
        ax.plot(x, cluster["inner_trace"], color="#e74c3c", linewidth=1.8, label=f"inner (0-{split_um:.1f} µm)")
        ax.plot(x, cluster["outer_trace"], color="#3498db", linewidth=1.8,
                label=f"outer ({split_um:.1f}-{cluster['R_lat_um']:.1f} µm)")
        title = f"Ring intensity traces — 1 cluster (red=inner  blue=outer)\nLatency: {latency_label}"
    else:
        for i, cluster in enumerate(clusters):
            color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
            line_width = 2.2 if i in highlight else 1.2
            alpha = 1.0 if i in highlight else 0.55
            ax.plot(x, cluster["trace"], color=color, linewidth=line_width, alpha=alpha,
                    label=f"cluster {i} (R_lat={cluster['R_lat_um']:.1f} µm)")
        title = f"Whole-cluster intensity traces — {len(clusters)} clusters, no ring split\nLatency: {latency_label}"

    ax.set_xlabel("Frame offset from spike (0 = spike)", fontsize=12)
    ax.set_ylabel("Mean intensity", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)


def plot_spike_detection_summary(
    rec_time: np.ndarray,
    vm: np.ndarray,
    picked: tuple[np.ndarray, np.ndarray],
    skipped: tuple[np.ndarray, np.ndarray],
    collapsed: tuple[np.ndarray, np.ndarray],
    title: str,
) -> Figure:
    """Full-length Vm trace with picked/skipped/collapsed spikes marked.

    Replaces AbfClip's old *_Vm.csv/*_peaks.csv/*_collapsed_peaks.csv/*_segments.csv export --
    one glance at this PNG shows the same information the CSVs held (which spikes were used vs
    dropped, and why) without needing to open 4 separate files.

    Args:
        rec_time: Full recording time axis (seconds).
        vm: Full recording membrane-potential trace, same length as rec_time.
        picked: (times, values) of spikes kept for analysis.
        skipped: (times, values) of spikes dropped for insufficient baseline margin.
        collapsed: (times, values) of extra spikes sharing a frame with an earlier spike.
        title: Figure title (e.g. "2025_06_11 0004").

    Returns:
        Figure with one axis: Vm line trace + 3 colored spike-category scatter overlays.
    """
    fig = Figure(figsize=(14, 5))
    ax = fig.add_subplot(111)

    # rec_time is absolute ABF sweep time (starts wherever the TTL trigger fired, not 0) --
    # shift every time array by the same offset so the plot's x-axis starts at 0, without
    # touching AbfClip's own rec_time (other code relies on its absolute values for indexing).
    t0 = rec_time[0]
    ax.plot(rec_time - t0, vm, color="black", linewidth=0.5, alpha=0.7, label="Vm", zorder=1)

    picked_times, picked_values = picked
    skipped_times, skipped_values = skipped
    collapsed_times, collapsed_values = collapsed
    picked_times = picked_times - t0 if picked_times.size else picked_times
    skipped_times = skipped_times - t0 if skipped_times.size else skipped_times
    collapsed_times = collapsed_times - t0 if collapsed_times.size else collapsed_times

    if picked_times.size:
        ax.scatter(picked_times, picked_values, color="#2ecc71", s=35, zorder=3,
                   label=f"picked (n={picked_times.size})")
    if skipped_times.size:
        ax.scatter(skipped_times, skipped_values, color="#e74c3c", s=35, marker="x", zorder=3,
                   label=f"skipped (n={skipped_times.size})")
    if collapsed_times.size:
        ax.scatter(collapsed_times, collapsed_values, color="#f39c12", s=25, marker="^", zorder=2,
                   label=f"collapsed (n={collapsed_times.size})")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Vm (mV)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# Exploratory direction-analysis exports. Keep GUI and existing analyses unchanged.
def _direction_map(ax, old, new, center) -> None:
    image = np.zeros((*old.shape, 3)) + 0.10
    image[old & new] = (0.42, 0.42, 0.42)
    image[new & ~old] = (1.0, 0.28, 0.25)
    image[old & ~new] = (0.12, 0.65, 1.0)
    ax.imshow(image)
    ax.plot(center[1], center[0], marker="+", color="#77ff77", ms=13, mew=2)
    for angle in np.arange(12) * np.pi / 6:
        ax.plot([center[1], center[1] + 1500 * np.cos(angle)],
                [center[0], center[0] - 1500 * np.sin(angle)], color="white", lw=0.5, alpha=0.18)
    ax.set_xlim(0, old.shape[1] - 1)
    ax.set_ylim(old.shape[0] - 1, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_hotspot_change_maps(masks, pairs, title) -> Figure:
    fig = Figure(figsize=(20, 10), layout="constrained")
    for i, item in enumerate(pairs):
        ax = fig.add_subplot(2, 4, i + 1)
        _direction_map(ax, masks[i], masks[i + 1], item["center"])
        ax.set_title(f"spike+{i} -> +{i + 1}\nGain {item['gain_total'] / 1000:.1f}k px | loss {item['loss_total'] / 1000:.1f}k px")
        polar = fig.add_subplot(2, 4, i + 5, projection="polar")
        theta = (np.arange(12) + 0.5) * np.pi / 6
        polar.bar(theta - 0.11, item["gain"] / 1000, width=0.21, color="#ef534b", label="Gained")
        polar.bar(theta + 0.11, item["loss"] / 1000, width=0.21, color="#2196db", label="Lost")
        polar.set_xticks(np.arange(4) * np.pi / 2, ["E", "N", "W", "S"])
        polar.set_title(f"Sector area (1000 pixels)\nGain R1={item['gain_r1']:.2f} | loss R1={item['loss_r1']:.2f}")
        polar.legend(loc="lower right", bbox_to_anchor=(1.25, -0.15), fontsize=9)
    fig.suptitle(title + "\nRed: gained | blue: lost | gray: retained | green cross: fixed initial-mask centroid"
                 "\nAll accepted mask pixels; 12 angular sectors. R1: one-sided concentration, not a radial-flow score.", fontsize=16)
    return fig


def plot_hotspot_direction_heatmaps(results, title) -> Figure:
    fig = Figure(figsize=(21, 14), layout="constrained")
    axes = fig.subplots(3, 4)
    keys = ("gain", "loss", "gain_normalized", "loss_normalized")
    names = ("Gained area (1000 px)", "Lost area (1000 px)",
             "Gain / old boundary length (px)", "Loss / old boundary length (px)")
    for col, (key, name) in enumerate(zip(keys, names, strict=True)):
        arrays = [np.array([pair[key] for pair in pairs]).T / (1000 if col < 2 else 1)
                  for pairs in results.values()]
        upper = max(float(np.nanmax(array)) for array in arrays)
        for row, ((sigma, _), array) in enumerate(zip(results.items(), arrays, strict=True)):
            ax = axes[row, col]
            im = ax.imshow(array, origin="lower", aspect="auto", vmin=0, vmax=upper or 1,
                           cmap="Reds" if col % 2 == 0 else "Blues")
            label = f"sigma={sigma} px" if isinstance(sigma, (int, float)) else str(sigma)
            ax.set_title(f"{label} | {name}")
            ax.set_xticks(range(4), ["0 -> 1", "1 -> 2", "2 -> 3", "3 -> 4"])
            ax.set_yticks(range(12), [f"{j * 30}-{(j + 1) * 30}" for j in range(12)])
            ax.set_ylabel("Angle (degrees): E=0, N=90, W=180, S=270")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title + "\nSame color scale down each column | blank: no measurable old boundary in sector"
                 "\nArea / boundary length is a descriptive equivalent thickness, not tracked displacement or velocity.", fontsize=16)
    return fig


def plot_hotspot_model_comparison(masks, model_pairs, center, title) -> Figure:
    fig = Figure(figsize=(28, 19), layout="constrained")
    axes = fig.subplots(4, 6)
    for row, models in enumerate(model_pairs):
        _direction_map(axes[row, 0], masks[row], masks[row + 1], center)
        axes[row, 0].set_title(f"Observed: spike+{row} -> +{row + 1}\nRed gain / blue loss")
        for col, model in enumerate(models, 1):
            ax = axes[row, col]
            predicted, observed = model["prediction"], masks[row + 1]
            rgb = np.zeros((*predicted.shape, 3)) + 0.1
            rgb[predicted & observed] = (0.5, 0.5, 0.5)
            rgb[predicted & ~observed] = (1.0, 0.55, 0.15)
            rgb[observed & ~predicted] = (0.7, 0.3, 0.95)
            ax.imshow(rgb)
            ax.plot(center[1], center[0], "+", color="#77ff77", ms=10)
            ax.set_xticks([])
            ax.set_yticks([])
            flag = " | bound reached" if model["at_bound"] else ""
            flag += " | fit incomplete" if not model["converged"] else ""
            parameters = "Empty prediction" if model["name"] == "Disappearance" else (
                f"Scale {model['scale']:.2f} | shift (x,y) = ({model['shift'][1]:.0f}, {-model['shift'][0]:.0f}) px"
            )
            ax.set_title(f"{model['name']} ({model['parameters']} parameters){flag}\n"
                         f"Holdout error reduction: {model['reduction']:.0%} | IoU: {model['iou']:.2f}\n"
                         + parameters, fontsize=10)
    fig.suptitle(title + "\nModel panels: gray agreement | orange predicted-only | purple observed-only"
                 "\nFit: alternating 128-px tiles; score: held-out tiles. Reduction relative to unchanged; negative = worse."
                 "\nExploratory spatial holdout, not independent replication. Outside field assumed empty; clipped hotspots make fits provisional.", fontsize=16)
    return fig


def plot_hotspot_robustness(groups, title) -> Figure:
    fig = Figure(figsize=(22, 14), layout="constrained")
    axes = fig.subplots(3, 3)
    row_names = ("R1: one-sided concentration", "R2: opposing-axis concentration", "Preferred direction (degrees)")
    for col, (name, variants) in enumerate(groups.items()):
        for row, suffix in enumerate(("r1", "r2", "angle")):
            matrix = []
            for pair in range(4):
                for kind in ("gain", "loss"):
                    values = [pairs[pair][f"{kind}_{suffix}"] for pairs in variants.values()]
                    if suffix == "angle":
                        values = [np.degrees(value) % 360 if pairs[pair][f"{kind}_r1"] >= 0.1 else np.nan
                                  for value, pairs in zip(values, variants.values(), strict=True)]
                    matrix.append(values)
            ax = axes[row, col]
            im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=360 if suffix == "angle" else 1,
                           cmap="twilight" if suffix == "angle" else "viridis")
            ax.set_xticks(range(len(variants)), list(variants), rotation=35, ha="right")
            ax.set_yticks(range(8), [f"{i} -> {i + 1} {kind}" for i in range(4) for kind in ("gain", "loss")])
            ax.set_title(name + "\n" + row_names[row])
            for iy, values in enumerate(matrix):
                for ix, value in enumerate(values):
                    if np.isfinite(value):
                        ax.text(ix, iy, f"{value:.0f}" if suffix == "angle" else f"{value:.2f}",
                                ha="center", va="center", color="white", fontsize=8,
                                bbox={"facecolor": "black", "alpha": 0.25, "edgecolor": "none", "pad": 1})
            fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title + "\nAll variants use the same fixed reference center except the explicit center-shift columns."
                 "\nAngles: E=0, N=90, W=180, S=270. Angle hidden when R1<0.1 (display rule, not a significance threshold)."
                 "\nLow R1 does not prove uniformity; high R2 can reveal two opposing lobes.", fontsize=16)
    return fig


def plot_hotspot_quality(masks, pairs, cropped, counts, model_sets, title) -> Figure:
    fig = Figure(figsize=(18, 12), layout="constrained")
    axes = fig.subplots(2, 2)
    axes[0, 0].plot(range(5), [mask.sum() / 1000 for mask in masks], "o-", label="Accepted area")
    axes[0, 0].set(xlabel="Frame after spike", ylabel="Area (1000 pixels)", title="Hotspot extent and accepted cluster count")
    for i, count in enumerate(counts):
        axes[0, 0].annotate(f"{count} cluster(s)", (i, masks[i].sum() / 1000), xytext=(0, 10), textcoords="offset points", ha="center")
    axes[0, 0].margins(y=0.25)
    axes[0, 1].bar(range(4), [item["border_fraction"] * 100 for item in pairs], color="#cf873a")
    axes[0, 1].set(xlabel="Earlier frame in pair", ylabel="Boundary pixels within 16 px of image edge (%)",
                   title="Near the field-of-view edge: possible truncation")
    for kind, color in (("gain", "#ef534b"), ("loss", "#2196db")):
        axes[1, 0].plot(range(4), [p[f"{kind}_r1"] for p in pairs], "o-", color=color, label=f"{kind}: full image")
        axes[1, 0].plot(range(4), [p[f"{kind}_r1"] for p in cropped], "s--", color=color, label=f"{kind}: omit outer 32 px")
    axes[1, 0].set(xlabel="Earlier frame in pair", ylabel="R1", ylim=(0, 1), title="Border exclusion sensitivity (does not restore missing data)")
    axes[1, 0].legend()
    for sigma, models in model_sets.items():
        axes[1, 1].plot(range(4), [pair[2]["reduction"] for pair in models], "o-", label=f"Centered scale, sigma={sigma}")
    axes[1, 1].plot(range(4), [pair[4]["reduction"] for pair in model_sets[3]], "s--", color="black", label="Disappearance, sigma=3")
    axes[1, 1].axhline(0, color="gray", ls="--")
    axes[1, 1].set(xlabel="Earlier frame in pair", ylabel="Held-out error reduction vs unchanged", title="Does the centered-scale fit survive smoothing changes?")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(title + "\nUnion of accepted clusters; count changes do not establish merging/splitting. No physical-flow claim.", fontsize=16)
    return fig


def plot_hotspot_event_consistency(events, median_pairs, total, title) -> tuple[Figure, Figure]:
    heat = Figure(figsize=(21, 13), layout="constrained")
    scatter = Figure(figsize=(21, 11), layout="constrained")
    heat_axes, scatter_axes = heat.subplots(2, 4), scatter.subplots(2, 4)
    for row, kind in enumerate(("gain", "loss")):
        for pair in range(4):
            ax = heat_axes[row, pair]
            matrix = np.array([event[pair][kind] / max(event[pair][f"{kind}_total"], 1) for event in events])
            empty = np.array([event[pair][f"{kind}_total"] == 0 for event in events])
            matrix[empty] = np.nan
            im = ax.imshow(matrix, aspect="auto", origin="upper", vmin=0, vmax=0.5,
                           cmap="Reds" if kind == "gain" else "Blues")
            ax.set_xticks([0, 3, 6, 9], ["0-30 E", "90-120 N", "180-210 W", "270-300 S"], rotation=35, ha="right")
            ax.set_ylabel("Detected event (time order)")
            ax.set_title(f"{kind}: spike+{pair} -> +{pair + 1}")
            heat.colorbar(im, ax=ax, shrink=0.7, label="Fraction of this event's changed area")
            ax = scatter_axes[row, pair]
            x = [event[pair][f"{kind}_r1"] for event in events]
            y = [event[pair][f"{kind}_r2"] for event in events]
            ax.scatter(x, y, s=22, alpha=0.5, color="#ef534b" if row == 0 else "#2196db", label="Individual events")
            ax.scatter(median_pairs[pair][f"{kind}_r1"], median_pairs[pair][f"{kind}_r2"],
                       marker="*", s=220, color="gold", edgecolor="black", label="Median-stack mask")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="R1: one-sided", ylabel="R2: opposing-axis",
                   title=f"{kind}: {pair} -> {pair + 1} | n={np.sum(np.isfinite(x))}")
            ax.grid(alpha=0.2)
    scatter_axes[0, 0].legend(fontsize=8)
    for fig in (heat, scatter):
        fig.suptitle(title + f"\n{len(events)}/{total} events detected at spike or spike+1; sigma=3 px; same median-derived fixed center"
                     "\nEmpty changes are missing, not zero concentration. Events are repeated observations within one recording, not biological replicates.", fontsize=16)
    return heat, scatter



def plot_hotspot_reference_centers(masks, centers, results, fits, title) -> Figure:
    fig = Figure(figsize=(19, 13), layout="constrained")
    axes = fig.subplots(2, 2)
    ax = axes[0, 0]
    ax.set_facecolor("#202020")
    for i, mask in enumerate(masks):
        ax.contour(mask.astype(float), levels=[0.5], colors=[str(0.3 + i * 0.15)], linewidths=0.6)
    colors = ("#e64c3c", "#238bdf", "#34ad66")
    for (name, center), color in zip(centers.items(), colors, strict=True):
        ax.plot(center[1], center[0], "+", ms=15, mew=3, color=color, label=name)
    ax.set(xlim=(0, masks[0].shape[1]), ylim=(masks[0].shape[0], 0), aspect="equal",
           title="Three reference centers, each fixed for all four pairs")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for (name, models), color in zip(fits.items(), colors, strict=True):
        axes[0, 1].plot(range(4), [m["reduction"] for m in models], "o-", color=color, label=name)
        axes[1, 0].plot(range(4), [m["iou"] for m in models], "o-", color=color, label=name)
        values = [pairs["gain_r1"] if i == 0 else pairs["loss_r1"] for i, pairs in enumerate(results[name])]
        axes[1, 1].plot(range(4), values, "o-", color=color, label=name)
    axes[0, 1].axhline(0, color="gray", ls="--")
    axes[0, 1].set(title="Centered-scale prediction: held-out error reduction", ylabel="Reduction vs unchanged")
    axes[1, 0].set(title="Centered-scale prediction: full-image overlap", ylabel="Intersection / union", ylim=(0, 1))
    axes[1, 1].set(title="Directional concentration of the dominant change", ylabel="R1 (gain for 0->1; loss thereafter)", ylim=(0, 1))
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xticks(range(4), ["0 -> 1", "1 -> 2", "2 -> 3", "3 -> 4"])
        ax.set_xlabel("Frame pair after spike")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(title + "\nSensitivity to the choice of a specific center | sigma=3 px"
                 "\nInitial, maximum-area and final-mask centroids are geometric reference candidates, not identified release sites."
                 "\nLater-derived centers are post hoc; these comparisons do not independently validate a source center.", fontsize=16)
    return fig
