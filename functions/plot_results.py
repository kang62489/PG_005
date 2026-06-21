"""Static, headless matplotlib export figures for spike-aligned region analysis.

No PySide6 dependency — these build plain Figure objects for fig.savefig()/
ResultsExporter.export_figure(), not interactive GUI windows. See classes/mpl_canvas.py
for the PySide6-coupled canvas widget used by the live GUI.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from classes.region_analyzer import CATEGORY_BRIGHT, CATEGORY_DIM, RegionAnalyzer, _nanargmax_relative

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
    median_segment: np.ndarray,
    spike_frame_idx: int,
    frame_duration_ms: float,
    title_info: dict,
    window: int = 4,
) -> Figure:
    """Static export figure: per-frame bright/dim summary (row 1) + temporal traces (row 2).

    Args:
        categorizer: fitted SpatialCategorizer (source_frames + categorized_frames)
        region_analyzer: RegionAnalyzer built from the spike frame
        median_segment: 3D z-scored segment matching categorizer.source_frames in shape
        spike_frame_idx: index of the spike frame within the segment
        frame_duration_ms: milliseconds per frame (e.g. AbfClip.ts_imgs * 1000)
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial"
        window: frames shown on each side of the spike frame in row 1 (default 4 -> 9 panels)

    Returns:
        Figure, ready for fig.savefig(...) or ResultsExporter.export_figure(...)
    """
    n_frames = len(categorizer.source_frames)
    offsets = list(range(-window, window + 1))
    n_cols = len(offsets)
    um_per_pixel = region_analyzer.um_per_pixel

    fig = Figure(figsize=(2.4 * n_cols, 8.5), dpi=100)
    gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 2.2], wspace=0.0)

    for col, offset in enumerate(offsets):
        ax = fig.add_subplot(gs[0, col])
        frame_idx = spike_frame_idx + offset
        if 0 <= frame_idx < n_frames:
            _plot_frame_panel(ax, categorizer, region_analyzer, frame_idx, offset, um_per_pixel)
        else:
            ax.axis("off")

    ax_trace = fig.add_subplot(gs[1, :])
    _plot_trace_panel(ax_trace, region_analyzer, median_segment, spike_frame_idx, frame_duration_ms, window)

    title = (
        f"Spatiotemporal Analysis: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _plot_frame_panel(
    ax: mpl.axes.Axes,
    categorizer: "SpatialCategorizer",
    region_analyzer: RegionAnalyzer,
    frame_idx: int,
    offset: int,
    um_per_pixel: float,
) -> None:
    """Row-1 panel: one frame's categorized image with the spike frame's fixed bright/dim regions."""
    cat_frame = categorizer.categorized_frames[frame_idx]
    bright = region_analyzer.bright_largest
    dim = region_analyzer.dim_largest

    cmap_cat = ListedColormap(["black", "gray", "white"])
    ax.imshow(cat_frame, cmap=cmap_cat, vmin=0, vmax=2, interpolation="nearest")

    is_spike_frame = offset == 0
    _overlay_region(
        ax, bright, contour_color="magenta", span_color="yellow", centroid_color="black", show_span=is_spike_frame
    )
    _overlay_region(ax, dim, contour_color="cyan", span_color="lime", centroid_color="white", show_centroid=False, show_span=False)

    bright_area_um2 = region_analyzer.area_in_combined_region(cat_frame, CATEGORY_BRIGHT)
    dim_area_um2 = region_analyzer.area_in_combined_region(cat_frame, CATEGORY_DIM)

    if bright is None:
        bright_line = "Bright [x (µm), y (µm), area (µm²)];\nnone detected" if is_spike_frame else "Bright area (µm²): none detected"
    elif is_spike_frame:
        bright_line = (
            f"Bright [x (µm), y (µm), area (µm²)];\n"
            f"({bright['x_span_um']:.1f}, {bright['y_span_um']:.1f}, {bright_area_um2:.1f})"
        )
    else:
        bright_line = f"Bright area (µm²): {bright_area_um2:.1f}"

    dim_line = f"Dim area (µm²): {dim_area_um2:.1f}" if dim is not None else "Dim area (µm²): none detected"
    frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
    ax.set_title(
        f"{frame_label}\n{bright_line}\n{dim_line}",
        fontsize=8,
        fontweight="bold" if offset == 0 else "normal",
        color="red" if offset == 0 else "black",
    )
    ax.axis("off")

    legend_elements = [
        Line2D([], [], color="magenta", linewidth=1.5, label="Bright contour (spike frame)"),
        Line2D([], [], color="cyan", linewidth=1.5, label="Dim contour (spike frame)"),
        Line2D(
            [], [], marker="+", color="black", linestyle="", markersize=8, markeredgewidth=2, label="Bright centroid"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=5)
    _add_scale_bar(um_per_pixel, ax, cat_frame.shape[1], cat_frame.shape[0], font_size=6)


def _overlay_region(
    ax: mpl.axes.Axes,
    region: dict | None,
    contour_color: str,
    span_color: str,
    centroid_color: str,
    show_centroid: bool = True,
    show_span: bool = True,
) -> None:
    """Draw a region's contour, and optionally its centroid and x/y-span crosshair, on ax."""
    if region is None:
        return

    contour = region["contour"]
    if contour is not None:
        contours = contour if isinstance(contour, list) else [contour]
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=contour_color, linewidth=1.5)

    y, x = region["centroid"]
    if show_centroid:
        ax.scatter(x, y, c=centroid_color, s=60, marker="+", linewidths=2, zorder=20)

    if show_span:
        x_span_px = region["x_span_px"]
        y_span_px = region["y_span_px"]
        x_west, x_east = x - x_span_px / 2, x + x_span_px / 2
        y_north, y_south = y - y_span_px / 2, y + y_span_px / 2
        ax.plot([x_west, x_east], [y, y], color=span_color, linewidth=1.5, zorder=15, alpha=0.8)
        ax.plot([x, x], [y_north, y_south], color=span_color, linewidth=1.5, zorder=15, alpha=0.8)


def _plot_trace_panel(
    ax: mpl.axes.Axes,
    region_analyzer: RegionAnalyzer,
    median_segment: np.ndarray,
    spike_frame_idx: int,
    frame_duration_ms: float,
    window: int,
) -> None:
    """Row-2 panel: bright/dim/total temporal traces, x-aligned to row 1's frame window."""
    traces = region_analyzer.get_temporal_traces(median_segment)
    n_frames = median_segment.shape[0]
    x = np.arange(n_frames) - spike_frame_idx

    ax.plot(x, traces["bright_trace"], color="magenta", label="Bright")
    ax.plot(x, traces["dim_trace"], color="cyan", label="Dim")
    ax.plot(x, traces["total_trace"], color="black", linestyle="--", label="Total")

    bright_peak_rel = _nanargmax_relative(traces["bright_trace"], spike_frame_idx)
    dim_peak_rel = _nanargmax_relative(traces["dim_trace"], spike_frame_idx)

    if bright_peak_rel is not None:
        ax.axvline(bright_peak_rel, color="magenta", linestyle=":", alpha=0.7)
    if dim_peak_rel is not None:
        ax.axvline(dim_peak_rel, color="cyan", linestyle=":", alpha=0.7)

    latency_ms = region_analyzer.get_peak_latency_ms(median_segment, spike_frame_idx, frame_duration_ms)
    if latency_ms is not None:
        latency_line = f"Peak Latency: {latency_ms:.1f} ms"
    else:
        latency_line = "Peak Latency: N/A (region not detected)"

    ax.set_xlim(-window, window)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Mean z-score")
    ax.set_title(f"Temporal change of bright and dim area\n{latency_line}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
