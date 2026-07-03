"""Static, headless matplotlib export figures for spike-aligned region analysis.

No PySide6 dependency — these build plain Figure objects for fig.savefig()/
ResultsExporter.export_figure(), not interactive GUI windows. See classes/mpl_canvas.py
for the PySide6-coupled canvas widget used by the live GUI.

Two export figures, mirroring the validated demo (archive/_demo_dbscan_tmp.py):
- plot_spatiotemporal_summary (-> region_sta/): B+D% signal trace showing why the
  critical frame was picked, + cluster shading on just that frame's own panel.
- plot_full_trace (-> full_traces/): the same fixed cluster-ring overlay repeated
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
    CATEGORY_BRIGHT,
    CATEGORY_DIM,
    SATURATION_AREA_PCT,
    RegionAnalyzer,
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
) -> Figure:
    """Static export figure: B+D% signal trace (row 1) + critical-frame panels (row 2).

    Shows why the critical frame was picked (B+D% vs the spike/spike+1 candidates)
    and what DBSCAN found there -- cluster shading is drawn only on the critical
    frame's own panel, not repeated across every panel (see plot_full_trace for
    the fixed-overlay version).

    Args:
        categorizer: fitted SpatialCategorizer (source_frames + categorized_frames)
        region_analyzer: RegionAnalyzer built from the segment
        spike_frame_idx: index of the spike frame within the segment
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial"

    Returns:
        Figure, ready for fig.savefig(...) or ResultsExporter.export_figure(...)
    """
    n_frames = len(categorizer.source_frames)
    obj = region_analyzer.obj
    area_pct = region_analyzer.area_pct
    critical_frame_idx = region_analyzer.critical_frame_idx
    um_per_pixel = region_analyzer.um_per_pixel

    fig = Figure(figsize=(20, 8), dpi=100)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 2.5], hspace=0.55, wspace=0.08)

    # --- Row 0: B+D% per frame, relative to spike ---
    ax_bd = fig.add_subplot(gs[0, :])
    ax_bd.plot(np.arange(n_frames) - spike_frame_idx, area_pct, color="#3498db", linewidth=1.6,
               marker="o", markersize=3.5)

    for frame_idx, label, color in [
        (spike_frame_idx - 1,
         f"spike-1: {area_pct[spike_frame_idx - 1]:.2f}%" if spike_frame_idx > 0 else "spike-1 (OOB)", "#888888"),
        (spike_frame_idx, f"spike: {area_pct[spike_frame_idx]:.2f}%", "#e74c3c"),
        (spike_frame_idx + 1,
         f"spike+1: {area_pct[spike_frame_idx + 1]:.2f}%" if spike_frame_idx + 1 < n_frames else "spike+1 (OOB)", "#f39c12"),
    ]:
        if 0 <= frame_idx < n_frames:
            ax_bd.axvline(frame_idx - spike_frame_idx, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=label)

    ax_bd.plot(critical_frame_idx - spike_frame_idx, area_pct[critical_frame_idx], "*", color="white", markersize=14,
               markeredgecolor="black", markeredgewidth=1, zorder=5, label=f"critical frame {critical_frame_idx}")

    ax_bd.set_xlabel("Frame offset from spike (0 = spike)", fontsize=9)
    ax_bd.set_ylabel("B+D  %", fontsize=9)
    ax_bd.set_title("B+D% per frame  |  star = critical frame (spike or spike+1)", fontsize=9)
    ax_bd.legend(fontsize=8, loc="upper right")
    ax_bd.tick_params(labelsize=8)

    # --- Row 1: spike-1 .. spike+4 panels; cluster shading only on the critical-frame panel ---
    for col, offset in enumerate(range(-1, 5)):
        frame_idx = spike_frame_idx + offset
        ax = fig.add_subplot(gs[1, col])
        if 0 <= frame_idx < n_frames:
            tag = "  ★" if frame_idx == critical_frame_idx else ""
            _plot_frame_panel(ax, categorizer, frame_idx, offset, um_per_pixel, tag)
            if frame_idx == critical_frame_idx:
                _draw_cluster_shading(ax, region_analyzer.label_frame, region_analyzer.centroids)
        else:
            frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
            ax.set_title(f"{frame_label}\n(out of range)", fontsize=9)
            ax.axis("off")

    # Row-1 title: DBSCAN settings
    _, tops, _, _ = gs.get_grid_positions(fig)
    if region_analyzer.saturated:
        settings_text = (
            f"RegionAnalyzer — OBJ={obj}  SATURATED (B+D%={area_pct[critical_frame_idx]:.1f}% "
            f">= {SATURATION_AREA_PCT:.0f}%) — whole frame treated as 1 cluster"
        )
    else:
        _, min_samples = eps_and_min_samples(obj)
        settings_text = (
            f"RegionAnalyzer — OBJ={obj}  min_samples={min_samples}  "
            f"found {region_analyzer.n_raw_clusters} → kept {len(region_analyzer.clusters)}"
        )
    fig.text(0.5, tops[1] + 0.015, settings_text, ha="center", va="bottom", fontsize=9, fontweight="bold")

    title = (
        f"Spatiotemporal Analysis: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
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
    title = (
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
        fig.suptitle(title, fontsize=12, fontweight="bold")
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
            _plot_frame_panel(ax, categorizer, frame_idx, offset, um_per_pixel, tag)
            _overlay_clusters(ax, clusters, highlight)
        else:
            frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
            ax.set_title(f"{frame_label}\n(out of range)", fontsize=8)
            ax.axis("off")

    # --- Row 1: full-segment z-score traces, with the 9-panel window annotated ---
    ax_trace = fig.add_subplot(gs[1, :])
    _plot_trace_panel(ax_trace, region_analyzer, median_segment, spike_frame_idx, frame_duration_ms, highlight)

    window_lo = max(0, spike_frame_idx - 4) - spike_frame_idx
    window_hi = min(n_frames - 1, spike_frame_idx + 4) - spike_frame_idx
    ax_trace.axvspan(window_lo, window_hi, color="#f1c40f", alpha=0.12, label="panels shown above")
    for frame_offset, color in [(-1, "#888888"), (0, "#e74c3c"), (1, "#f39c12")]:
        if 0 <= spike_frame_idx + frame_offset < n_frames:
            ax_trace.axvline(frame_offset, color=color, linestyle=":", linewidth=1.0, alpha=0.6)
    ax_trace.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig


def _plot_frame_panel(
    ax: mpl.axes.Axes,
    categorizer: "SpatialCategorizer",
    frame_idx: int,
    offset: int,
    um_per_pixel: float,
    tag: str = "",
) -> None:
    """One frame's categorized image with a B/D/B+D area title.

    No cluster overlay is drawn here -- callers layer that on top afterward
    (see _draw_cluster_shading / _overlay_clusters), since the two export
    figures use different overlay styles on different subsets of panels.
    """
    cat_frame = categorizer.categorized_frames[frame_idx]

    cmap_cat = ListedColormap(["black", "gray", "white"])
    ax.imshow(cat_frame, cmap=cmap_cat, vmin=0, vmax=2, interpolation="nearest")
    # Pin the view to the image bounds so cluster circles/overlays added later
    # (which can extend past the frame edge) get clipped instead of shrinking
    # the image by expanding the panel's autoscaled view to fit them.
    ax.set_xlim(0, cat_frame.shape[1])
    ax.set_ylim(cat_frame.shape[0], 0)
    ax.set_autoscale_on(False)

    bright_px = np.count_nonzero(cat_frame == CATEGORY_BRIGHT)
    dim_px = np.count_nonzero(cat_frame == CATEGORY_DIM)
    total_px = cat_frame.size
    bright_pct = 100.0 * bright_px / total_px
    dim_pct = 100.0 * dim_px / total_px
    bright_um2 = bright_px * um_per_pixel ** 2
    dim_um2 = dim_px * um_per_pixel ** 2

    frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
    ax.set_title(
        f"{frame_label}{tag}\n"
        f"B: {bright_um2:.0f} µm² ({bright_pct:.1f}%)\n"
        f"D: {dim_um2:.0f} µm² ({dim_pct:.1f}%)\n"
        f"B+D: {bright_um2 + dim_um2:.0f} µm² ({bright_pct + dim_pct:.1f}%)",
        fontsize=7,
        fontweight="bold" if offset == 0 else "normal",
        color="red" if offset == 0 else "black",
    )
    ax.axis("off")
    _add_scale_bar(um_per_pixel, ax, cat_frame.shape[1], cat_frame.shape[0], font_size=6)


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
        ax.plot(col_c, row_c, "+", color="black", markersize=20, markeredgewidth=4)
        ax.plot(col_c, row_c, "+", color="white", markersize=18, markeredgewidth=2.5)
        ax.text(col_c + 5, row_c - 5, str(cluster_idx), color="white", fontsize=9, fontweight="bold")


def _overlay_clusters(ax: mpl.axes.Axes, clusters: list[dict], highlight: set[int]) -> None:
    """Translucent cluster fill + ring/circle outlines, colored by cluster index.

    1 cluster -> inner (dashed) + outer (solid) ring pair at R/sqrt(2) and R,
    matching compute_ring_traces' split. >1 clusters -> a single solid circle
    at R per cluster, no ring split, matching compute_cluster_trace.
    """
    if not clusters:
        return

    is_single = "inner_mask" in clusters[0]
    mask_shape = clusters[0]["inner_mask"].shape if is_single else clusters[0]["mask"].shape
    overlay = np.zeros((*mask_shape, 4), dtype=float)
    for i, cluster in enumerate(clusters):
        r, g, b, _ = CLUSTER_RGBA[i % len(CLUSTER_RGBA)]
        if is_single:
            overlay[cluster["inner_mask"]] = (r, g, b, 0.55)
            overlay[cluster["outer_mask"]] = (r, g, b, 0.28)
        else:
            overlay[cluster["mask"]] = (r, g, b, 0.4)
    ax.imshow(overlay, interpolation="nearest")

    for i, cluster in enumerate(clusters):
        edge_color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
        row_c, col_c = cluster["centroid"]
        radius_px = cluster["R_px"]
        line_width = 1.8 if i in highlight else 1.0
        rings = [(radius_px / np.sqrt(2), "--"), (radius_px, "-")] if is_single else [(radius_px, "-")]
        for ring_radius, linestyle in rings:
            circle = Circle((col_c, row_c), ring_radius, fill=False, edgecolor=edge_color,
                             linewidth=line_width, linestyle=linestyle)
            ax.add_patch(circle)
        ax.plot(col_c, row_c, "+", color="black", markersize=12, markeredgewidth=2.5)
        ax.plot(col_c, row_c, "+", color=edge_color, markersize=10, markeredgewidth=1.5)


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
        split_um = cluster["R_um"] / np.sqrt(2)
        ax.plot(x, cluster["inner_trace"], color="#e74c3c", linewidth=1.8, label=f"inner (0-{split_um:.1f} µm)")
        ax.plot(x, cluster["outer_trace"], color="#3498db", linewidth=1.8,
                label=f"outer ({split_um:.1f}-{cluster['R_um']:.1f} µm)")
        title = f"Ring z-score traces — 1 cluster (red=inner  blue=outer)\nLatency: {latency_label}"
    else:
        for i, cluster in enumerate(clusters):
            color = CLUSTER_RGBA[i % len(CLUSTER_RGBA)][:3]
            line_width = 2.2 if i in highlight else 1.2
            alpha = 1.0 if i in highlight else 0.55
            ax.plot(x, cluster["trace"], color=color, linewidth=line_width, alpha=alpha,
                    label=f"cluster {i} (R={cluster['R_um']:.1f} µm)")
        title = f"Whole-cluster z-score traces — {len(clusters)} clusters, no ring split\nLatency: {latency_label}"

    ax.set_xlabel("Frame offset from spike (0 = spike)")
    ax.set_ylabel("Mean z-score")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
