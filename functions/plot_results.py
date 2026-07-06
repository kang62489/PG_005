"""Static, headless matplotlib export figures for spike-aligned region analysis.

No PySide6 dependency — these build plain Figure objects for fig.savefig()/
ResultsExporter.export_figure(), not interactive GUI windows. See classes/mpl_canvas.py
for the PySide6-coupled canvas widget used by the live GUI.

Two export figures, mirroring the validated demo (archive/_demo_dbscan_tmp.py):
- plot_spatiotemporal_summary (-> region_sta/): B+D% signal trace showing why the
  critical frame was picked, + cluster shading on just that frame's own panel.
- plot_full_trace (-> region_sta/, *_LATENCY.png): the same fixed cluster-ring overlay repeated
  across a 9-panel window, + the full-segment z-score trace with that window annotated.
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
    RegionAnalyzer,
    _decay_model,
    _detect_clusters,
    _peak_offset_from_spike,
    eps_and_min_samples,
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
    """Static export figure: B+D% signal trace (row 1) + critical-frame panels (row 2)
    + overlapped electrophysiology traces per segment (row 3).

    Shows why the critical frame was picked (B+D% vs the spike/spike+1 candidates)
    and what DBSCAN found there -- cluster shading is drawn only on the critical
    frame's own panel, not repeated across every panel (see plot_full_trace for
    the fixed-overlay version). Row 3 shows the actual detected Vm spike waveform
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
    obj = region_analyzer.obj
    area_pct = region_analyzer.area_pct
    critical_frame_idx = region_analyzer.critical_frame_idx
    um_per_pixel = region_analyzer.um_per_pixel

    # --- Figure and grid setup ---
    fig = Figure(figsize=(26, 13), dpi=100)
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.5, 2.0], hspace=0.7)
    gs_panels = gs_outer[1].subgridspec(1, 9, wspace=0.08)

    # --- Row 0: B+D% per frame trace ---
    ax_bd = fig.add_subplot(gs_outer[0])
    ax_bd.plot(np.arange(n_frames) - spike_frame_idx, area_pct, color="#3498db", linewidth=1.6,
               marker="o", markersize=3.5)

    # Pre-compute µm² for each frame: area_pct / 100 * total_pixels * um_per_pixel²
    total_px = categorizer.categorized_frames[0].size
    frame_um2 = area_pct * total_px / 100.0 * um_per_pixel ** 2

    for frame_idx, label, color in [
        (spike_frame_idx - 1,
         f"spike-1: {frame_um2[spike_frame_idx - 1]:.0f} µm² ({area_pct[spike_frame_idx - 1]:.1f}%)"
         if spike_frame_idx > 0 else "spike-1 (OOB)", "#888888"),
        (spike_frame_idx,
         f"spike: {frame_um2[spike_frame_idx]:.0f} µm² ({area_pct[spike_frame_idx]:.1f}%)", "#e74c3c"),
        (spike_frame_idx + 1,
         f"spike+1: {frame_um2[spike_frame_idx + 1]:.0f} µm² ({area_pct[spike_frame_idx + 1]:.1f}%)"
         if spike_frame_idx + 1 < n_frames else "spike+1 (OOB)", "#f39c12"),
    ]:
        if 0 <= frame_idx < n_frames:
            ax_bd.axvline(frame_idx - spike_frame_idx, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)

    ax_bd.plot(critical_frame_idx - spike_frame_idx, area_pct[critical_frame_idx], "*", color="white", markersize=14,
               markeredgecolor="black", markeredgewidth=1, zorder=5,
               label=f"critical: frame {critical_frame_idx}  {frame_um2[critical_frame_idx]:.0f} µm² ({area_pct[critical_frame_idx]:.1f}%)")

    _draw_decay_fit(ax_bd, region_analyzer, spike_frame_idx, n_frames, frame_duration_ms)

    ax_bd.set_xlabel("Frame offset from spike (0 = spike)", fontsize=12)
    ax_bd.set_ylabel("B+D area (%)", fontsize=12)
    ax_bd.set_title("Bright+Dim area coverage per frame  |  star = critical frame (spike or spike+1)", fontsize=12)
    ax_bd.legend(fontsize=10, loc="upper right")
    ax_bd.tick_params(labelsize=10)

    # --- Row 1: spike-4 .. spike+4 frame panels ---
    eps_px, min_samples = eps_and_min_samples(obj)
    hotspot_area_lines = {
        idx: _format_hotspot_area_line(categorizer.categorized_frames[idx], area_pct[idx], eps_px, min_samples, um_per_pixel)
        for idx in range(spike_frame_idx - 4, spike_frame_idx + 5)
        if 0 <= idx < n_frames
    }
    span_line = _format_span_line(region_analyzer)

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
                span_line=span_line if frame_idx == region_analyzer.max_area_frame_idx else None,
            )
            if is_critical_frame:
                _draw_cluster_shading(ax_frame, region_analyzer.label_frame, region_analyzer.centroids)
            if frame_idx == region_analyzer.max_area_frame_idx:
                _draw_span_bbox(ax_frame, region_analyzer)
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
    full-segment per-cluster z-score traces with that window annotated (row 2).

    Every panel (spike-4..spike+4) shows the same fixed critical-frame cluster
    overlay, so you can see how the underlying pixel pattern moves/changes under
    it; the trace row spans the whole segment (never cropped), with a shaded band
    marking which x-range the panels above cover.

    Args:
        region_analyzer: RegionAnalyzer built from the segment
        categorizer: fitted SpatialCategorizer (source_frames + categorized_frames)
        median_segment: 3D z-scored segment (frames, height, width)
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

    # --- Row 1: full-segment z-score traces, with the 9-panel window annotated ---
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


def _plot_frame_panel(
    ax: mpl.axes.Axes,
    categorizer: "SpatialCategorizer",
    frame_idx: int,
    offset: int,
    um_per_pixel: float,
    tag: str = "",
    hotspot_area_line: str | None = None,
    span_line: str | None = None,
    stats_lines: list[str] | None = None,
) -> None:
    """One frame's categorized image with a stats title.

    Defaults to a B/D/B+D area breakdown (used by plot_spatiotemporal_summary,
    where row 0 above is the B+D% trace). Callers whose companion trace row
    plots something else (e.g. plot_full_trace's z-score ring trace) should
    pass stats_lines to show a title relevant to that instead.

    No cluster overlay is drawn here -- callers layer that on top afterward
    (see _draw_cluster_shading / _overlay_clusters), since the two export
    figures use different overlay styles on different subsets of panels.
    """
    cat_frame = categorizer.categorized_frames[frame_idx]

    # --- Image ---
    cmap_cat = ListedColormap(["black", "gray", "white"])
    ax.imshow(cat_frame, cmap=cmap_cat, vmin=0, vmax=2, interpolation="nearest")
    # Pin view so cluster circles/overlays added later get clipped, not rescaled.
    ax.set_xlim(0, cat_frame.shape[1])
    ax.set_ylim(cat_frame.shape[0], 0)
    ax.set_autoscale_on(False)

    # --- Stats lines ---
    if stats_lines is None:
        stats_lines = []
        if hotspot_area_line is not None:
            stats_lines.append(hotspot_area_line)
        if span_line is not None:
            stats_lines.append(span_line)

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



def _format_hotspot_area_line(
    cat_frame: np.ndarray, area_pct_value: float, eps_px: int, min_samples: int, um_per_pixel: float
) -> str:
    """DBSCAN-kept area line for spike/spike+1 panel titles."""
    label_frame, _, _, _ = _detect_clusters(cat_frame, area_pct_value, eps_px, min_samples)
    hotspot_px = np.count_nonzero(label_frame >= 0)
    hotspot_um2 = hotspot_px * um_per_pixel ** 2
    hotspot_pct = 100.0 * hotspot_px / label_frame.size
    return f"hotspots: {hotspot_um2:.0f} µm² ({hotspot_pct:.1f}%)"


def _format_span_line(region_analyzer: RegionAnalyzer) -> str | None:
    """X/Y span line for the max-area frame panel title."""
    x_span = region_analyzer.max_area_x_span_um
    y_span = region_analyzer.max_area_y_span_um
    if x_span is None or y_span is None:
        return None
    return f"span x/y: {x_span:.1f} / {y_span:.1f} µm"


def _frame_z_lines(clusters: list[dict], frame_idx: int) -> list[str]:
    """Per-frame z-score line(s) for a plot_full_trace panel title.

    Mirrors _plot_trace_panel's split so the panel title and the trace row
    below refer to the same numbers: 1 cluster -> inner/outer ring z-score at
    this frame; >1 clusters -> each cluster's whole-cluster z-score.
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

    Used only on the critical frame's own panel in plot_spatiotemporal_summary.
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


def _draw_span_bbox(ax: mpl.axes.Axes, region_analyzer: "RegionAnalyzer") -> None:
    """Dashed bounding-box rectangle over the accepted-mask span on the max-area panel."""
    x_min = region_analyzer.max_area_x_min_px
    y_min = region_analyzer.max_area_y_min_px
    x_span = region_analyzer.max_area_x_span_px
    y_span = region_analyzer.max_area_y_span_px
    if x_min is None or y_min is None:
        return
    rect = Rectangle(
        (x_min, y_min), x_span, y_span,
        linewidth=2.0, edgecolor="#f1c40f", facecolor="none",
        linestyle="--", alpha=0.85, zorder=4,
    )
    ax.add_patch(rect)


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
    ax.plot(
        t + (peak_frame_idx - spike_frame_idx), fitted, "--", color="#2ecc71", linewidth=1.8, zorder=4,
        label=f"decay fit: τ={tau_ms:.0f} ms (R²={r2_text})",
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
    """Per-cluster z-score traces across the full segment (never cropped -- the
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
        title = f"Ring z-score traces — 1 cluster (red=inner  blue=outer)\nLatency: {latency_label}"
    else:
        for i, cluster in enumerate(clusters):
            color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
            line_width = 2.2 if i in highlight else 1.2
            alpha = 1.0 if i in highlight else 0.55
            ax.plot(x, cluster["trace"], color=color, linewidth=line_width, alpha=alpha,
                    label=f"cluster {i} (R_lat={cluster['R_lat_um']:.1f} µm)")
        title = f"Whole-cluster z-score traces — {len(clusters)} clusters, no ring split\nLatency: {latency_label}"

    ax.set_xlabel("Frame offset from spike (0 = spike)", fontsize=12)
    ax.set_ylabel("Mean z-score", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
