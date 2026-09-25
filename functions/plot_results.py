"""Static, headless matplotlib export figures for spike-aligned region analysis.

No PySide6 dependency — these build plain Figure objects for fig.savefig()/
ResultsExporter.export_figure(), not interactive GUI windows. See classes/mpl_canvas.py
for the PySide6-coupled canvas widget used by the live GUI.

Export figures:
- plot_spatiotemporal_summary (-> spatial/): density-gated hotspot-area trace showing
  why the critical frame was picked, + cluster shading on the spike and spike+1 panels.
- plot_flow_panels (-> flow/, *_FLOW.png): TV-L1 flow arrows + speed (µm/s), in the CAT mask
  and over the full field, spike-1->spike through spike+3->spike+4.
- plot_flow_streamlines (-> flow/, *_STREAMLINES.png): the same flow as streamlines, in the CAT
  mask and over the full field; CAT-row titles show the pair's source / sink / anisotropic type.
- plot_segment_reliability_montage (-> reliability/, *_RELIABILITY.png): one panel per raw
  segment showing its own density-gated detection result, for reviewing reliability by eye.
- plot_vm_success_vs_failure (-> reliability/, *_VM_SUCCESS_FAIL.png): peak-aligned Vm of
  detected vs failed segments, ±50 ms, with AP threshold dots.
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
from scipy.ndimage import maximum_filter

from classes.region_analyzer import (
    MIN_DECAY_FIT_R2,
    PIXEL_SCALE,
    RegionAnalyzer,
    _decay_model,
)
from functions.ap_threshold import baseline_vm, find_ap_threshold
from functions.flow_pattern import FLOW_PATTERN_BLOCK, block_mean

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
    vm_segments: list[tuple[np.ndarray, np.ndarray]],
    frame_duration_ms: float,
) -> Figure:
    """Static export figure: density-gated hotspot-area signal trace (row 1) +
    spike/spike+1 frame panels (row 2) + overlapped electrophysiology traces per
    segment (row 3).

    Shows why the critical frame was picked (hotspot area vs the spike/spike+1
    candidates) and what was found on each of those two frames independently --
    cluster shading is drawn on both the spike and spike+1 panels.
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


FLOW_QUIVER_STEP = 24     # px between drawn arrows
FLOW_AUTO_ARROW_FRAC = 0.9  # auto scale: each panel's p95 arrow is this fraction of FLOW_QUIVER_STEP


def _draw_flow_quivers(ax: mpl.axes.Axes, frame: np.ndarray, pair: dict, draw: np.ndarray,
                       grid_y: np.ndarray, grid_x: np.ndarray, um_per_pixel: float,
                       intensity_range: tuple[float, float]) -> None:
    """MED frame (shared intensity_range) + red auto-scaled arrows at the grid points in `draw` + scale bar."""
    ax.imshow(frame, cmap="gray", origin="upper", vmin=intensity_range[0], vmax=intensity_range[1])
    if draw.any():
        u_grid, v_grid = pair["u"][grid_y, grid_x][draw], pair["v"][grid_y, grid_x][draw]
        scale = max(float(np.percentile(np.hypot(u_grid, v_grid), 95)), 1e-3) / (FLOW_AUTO_ARROW_FRAC * FLOW_QUIVER_STEP)
        ax.quiver(
            grid_x[draw], grid_y[draw], u_grid, v_grid,
            color="red", angles="xy", scale_units="xy", scale=scale, width=0.003,
            headwidth=2.5, headlength=3, headaxislength=2.5,
        )
    _add_scale_bar(um_per_pixel, ax, frame.shape[1], frame.shape[0])


def plot_flow_panels(
    med_stack: np.ndarray,
    flow_pairs: list[dict],
    title_info: dict,
    frame_duration_ms: float,
) -> Figure:
    """One column per flow pair (RegionAnalyzer.compute_flow), four rows:

      Row 1: MED "from" frame + red arrows inside the CAT mask
      Row 2: flow speed (µm/s) inside the CAT mask
      Row 3: MED "from" frame + red arrows over the full field
      Row 4: flow speed (µm/s) over the full field

    CAT mask = the pair's keep_mask (union of both frames' CAT-bright pixels); row-1 arrows sit at grid
    points within FLOW_QUIVER_STEP of it (a dilation, so a coarse grid can't miss a thin hotspot).
    Arrows are auto-scaled per panel; MED panels (rows 1, 3) share one gray range (1st-99th percentile
    of med_stack); rows 2 and 4 share one color scale.

    Args:
        med_stack: (frames, H, W) median stack the flow was computed on.
        flow_pairs: dicts with "label", "idx_from", "u", "v", "keep_mask" (from compute_flow_pairs()).
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial".
        frame_duration_ms: imaging frame duration, converts px/frame -> µm/s.
    """
    n_panels = max(len(flow_pairs), 1)
    fig = Figure(figsize=(6 * n_panels, 24), dpi=110, layout="constrained")
    axes = fig.subplots(4, n_panels, squeeze=False)
    height, width = med_stack.shape[1], med_stack.shape[2]
    grid_y, grid_x = np.mgrid[0:height:FLOW_QUIVER_STEP, 0:width:FLOW_QUIVER_STEP]
    um_per_pixel = 1.0 / PIXEL_SCALE[title_info["obj"]]
    px_per_frame_to_um_per_s = um_per_pixel * 1000.0 / frame_duration_ms
    intensity_range = tuple(float(x) for x in np.percentile(med_stack, [1, 99]))

    speeds = [np.hypot(p["u"], p["v"]) * px_per_frame_to_um_per_s for p in flow_pairs]
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in speeds]), 99.5)) if speeds else 1.0
    im = None

    for i, (pair, speed) in enumerate(zip(flow_pairs, speeds, strict=True)):
        frame = med_stack[pair["idx_from"]]
        near_bright = maximum_filter(pair["keep_mask"], size=FLOW_QUIVER_STEP)[grid_y, grid_x]
        all_points = np.ones_like(near_bright)

        _draw_flow_quivers(axes[0, i], frame, pair, near_bright, grid_y, grid_x, um_per_pixel, intensity_range)
        axes[0, i].set_title(f"{pair['label']}  (CAT mask)", fontsize=10)
        im = axes[1, i].imshow(np.where(pair["keep_mask"], speed, np.nan), cmap="magma", vmin=0, vmax=vmax)
        axes[1, i].set_title("flow speed (µm/s), CAT mask", fontsize=10)
        _draw_flow_quivers(axes[2, i], frame, pair, all_points, grid_y, grid_x, um_per_pixel, intensity_range)
        axes[2, i].set_title(f"{pair['label']}  (full field)", fontsize=10)
        axes[3, i].imshow(speed, cmap="magma", vmin=0, vmax=vmax)
        axes[3, i].set_title("flow speed (µm/s), full field", fontsize=10)
        for row in (1, 3):
            _add_scale_bar(um_per_pixel, axes[row, i], width, height)
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])

    if im is not None:
        for row in (1, 3):
            fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, label="µm/s")
    fig.suptitle(
        f"Flow Analysis: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
        "  (unmasked TV-L1; rows 1-2 in CAT mask, rows 3-4 full field)",
        fontsize=13,
    )
    return fig


FLOW_STREAM_DENSITY = 2.5    # matplotlib streamplot density


def _draw_flow_streamlines(ax: mpl.axes.Axes, frame: np.ndarray, pair: dict, masked: bool,
                           um_per_pixel: float, intensity_range: tuple[float, float]) -> None:
    """MED frame (shared intensity_range) + red streamlines (block-averaged flow; CAT blocks only if masked) + scale bar."""
    height, width = frame.shape
    ax.imshow(frame, cmap="gray", origin="upper", vmin=intensity_range[0], vmax=intensity_range[1])
    u_small = block_mean(pair["u"])
    v_small = block_mean(pair["v"])
    if masked:
        keep_small = block_mean(pair["keep_mask"].astype(float)) > 0
    else:
        keep_small = np.ones(u_small.shape, dtype=bool)
    if keep_small.any():
        ys = np.arange(u_small.shape[0]) * FLOW_PATTERN_BLOCK + FLOW_PATTERN_BLOCK / 2
        xs = np.arange(u_small.shape[1]) * FLOW_PATTERN_BLOCK + FLOW_PATTERN_BLOCK / 2
        ax.streamplot(
            xs, ys, np.ma.masked_where(~keep_small, u_small), np.ma.masked_where(~keep_small, v_small),
            color="red", density=FLOW_STREAM_DENSITY, linewidth=0.8, arrowsize=1.2,
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    _add_scale_bar(um_per_pixel, ax, width, height)


def _draw_angle_crosshair(ax: mpl.axes.Axes, width: int, height: int) -> None:
    """Dashed crosshair at the frame centre with 0° (→) / 90° (↑) / 180° (←) / 270° (↓) arm labels, for reading drift angles."""
    cx, cy, arm = width / 2, height / 2, 0.3 * min(width, height)
    style = {"color": "cyan", "linewidth": 1.5, "linestyle": "--", "zorder": 5}
    ax.plot([cx - arm, cx + arm], [cy, cy], **style)
    ax.plot([cx, cx], [cy - arm, cy + arm], **style)
    bbox = {"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
    for text, (x, y) in {"0°": (cx + arm, cy), "90°": (cx, cy - arm), "180°": (cx - arm, cy), "270°": (cx, cy + arm)}.items():
        ax.text(x, y, text, ha="center", va="center", fontsize=9, color="teal", fontweight="bold", bbox=bbox, zorder=6)


def _pattern_title(pair: dict) -> str:
    """' · source' / ' · sink' / ' · anisotropic 53°'; '' if unfitted."""
    pattern = pair.get("pattern", {})
    label = pattern.get("label")
    if label is None:
        return ""
    if label == "anisotropic":
        return f" · anisotropic {pattern['drift_angle_deg']:.0f}°"
    return f" · {label}"


def plot_flow_streamlines(
    med_stack: np.ndarray,
    flow_pairs: list[dict],
    title_info: dict,
) -> Figure:
    """One column per flow pair (same flow as plot_flow_panels), two rows:

      Row 1: MED "from" frame + red streamlines inside the CAT mask, title = CAT-fit pattern
      Row 2: MED "from" frame + red streamlines over the full field

    Streamlines run on FLOW_PATTERN_BLOCK x FLOW_PATTERN_BLOCK block-averaged u, v; row 1 keeps the blocks
    touching the pair's keep_mask; its title carries the CAT-fit pattern (pair["pattern"],
    functions/flow_pattern.py), and anisotropic panels get a 0/90/180/270° crosshair for reading the
    drift angle. MED panels share one gray range (1st-99th percentile of med_stack).

    Args:
        med_stack: (frames, H, W) median stack the flow was computed on.
        flow_pairs: dicts with "label", "idx_from", "u", "v", "keep_mask" (from compute_flow_pairs()).
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial".
    """
    n_panels = max(len(flow_pairs), 1)
    fig = Figure(figsize=(6 * n_panels, 12), dpi=110, layout="constrained")
    axes = fig.subplots(2, n_panels, squeeze=False)
    um_per_pixel = 1.0 / PIXEL_SCALE[title_info["obj"]]
    intensity_range = tuple(float(x) for x in np.percentile(med_stack, [1, 99]))

    for i, pair in enumerate(flow_pairs):
        frame = med_stack[pair["idx_from"]]
        _draw_flow_streamlines(axes[0, i], frame, pair, True, um_per_pixel, intensity_range)
        if pair.get("pattern", {}).get("label") == "anisotropic":
            _draw_angle_crosshair(axes[0, i], frame.shape[1], frame.shape[0])
        axes[0, i].set_title(f"{pair['label']}  (CAT mask){_pattern_title(pair)}", fontsize=10)
        _draw_flow_streamlines(axes[1, i], frame, pair, False, um_per_pixel, intensity_range)
        axes[1, i].set_title(f"{pair['label']}  (full field)", fontsize=10)
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"Flow Streamlines: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
        "  (unmasked TV-L1; row 1 in CAT mask, row 2 full field)",
        fontsize=13,
    )
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
    obj: str,
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
        obj: objective ("10X" / "40X" / "60X"), for the per-panel scale bar.

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
            _add_scale_bar(1.0 / PIXEL_SCALE[obj], ax, label_frame.shape[1], label_frame.shape[0], font_size=5)

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


VM_WINDOW_MS = 50  # ± ms around each trace's own peak


def _mean_sd(values: np.ndarray, signed: bool = False) -> str:
    """'-2.4 ± 0.7' (or '+5.6 ± 0.7' when signed); SD is 0 for a single value."""
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():{'+' if signed else ''}.1f} ± {sd:.1f}"


def plot_vm_success_vs_failure(
    vm_success: list[tuple[np.ndarray, np.ndarray]],
    vm_failure: list[tuple[np.ndarray, np.ndarray]],
    rec_stem: str,
) -> Figure:
    """1x2 Vm overlay: detected (left) vs failed (right) segments, each trace peak-aligned at t=0.

    The AP threshold of every trace (find_ap_threshold) is marked with a dot; each panel's
    title gives its group's mean ± SD threshold.

    Args:
        vm_success, vm_failure: per-segment (time_ms, Vm) pairs, from AbfClip.get_vm_segments().
        rec_stem: recording name, for the suptitle.
    """
    fig = Figure(figsize=(12, 4), dpi=150)
    ax_success, ax_fail = fig.subplots(1, 2, sharey=True)

    groups = (
        (ax_success, vm_success, "tab:green", "Detected (success)"),
        (ax_fail, vm_failure, "tab:red", "Failure"),
    )
    for ax, vm_pairs, color, label in groups:
        thresholds, rel_thresholds = [], []
        for time_ms, vm in vm_pairs:
            t_rel = time_ms - time_ms[int(np.argmax(vm))]
            ax.plot(t_rel, vm, alpha=0.3, color=color, linewidth=0.8)
            hit = find_ap_threshold(t_rel, vm)
            if hit is None:
                continue
            thresholds.append(hit)
            base = baseline_vm(t_rel, vm)
            if base is not None:
                rel_thresholds.append(hit[1] - base)

        if thresholds:
            t_th, v_th = np.array(thresholds).T
            ax.scatter(t_th, v_th, s=10, color="black", zorder=3)
            th_line = f"AP threshold {_mean_sd(v_th)} mV (n={len(v_th)})"
            if rel_thresholds:
                th_line += f"\n{_mean_sd(np.array(rel_thresholds), signed=True)} mV above baseline"
        else:
            th_line = "AP threshold: n/a"

        ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(f"{label}, n={len(vm_pairs)}\n{th_line}", fontsize=10)
        ax.set_xlabel("Time from own spike peak (ms)")
        ax.set_xlim(-VM_WINDOW_MS, VM_WINDOW_MS)
        ax.grid(True)
    ax_success.set_ylabel("Vm (mV)")

    fig.suptitle(rec_stem)
    fig.tight_layout()
    return fig


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

    Defaults to a hotspot area line (plot_spatiotemporal_summary); pass stats_lines
    for a different title. No cluster overlay here -- callers add it (_draw_cluster_shading).
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


def _draw_cluster_shading(ax: mpl.axes.Axes, label_frame: np.ndarray, centroids: list[tuple[float, float]]) -> None:
    """Translucent per-cluster fill (DBSCAN's raw label map, no ring circles) +
    centroid cross and index label.

    Used on the spike frame's and spike+1 frame's own panels in
    plot_spatiotemporal_summary (each with its own independent label_frame/centroids).
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


def plot_full_stack_kymographs(profiles, um_per_pixel, spikes: np.ndarray, title) -> Figure:
    """Full-duration, whole-image band means and peak positions, without time crops."""
    names = ("Left to right", "Top to bottom", "Top-left to bottom-right", "Bottom-left to top-right")
    frames = profiles[0].shape[1]
    times = np.arange(frames) / 20
    vmin, vmax = np.percentile(np.concatenate([p.ravel() for p in profiles]), [1, 99])
    fig = Figure(figsize=(21, 15), layout="constrained")
    axes = fig.subplots(4, 1, sharex=True)
    for ax, profile, name in zip(axes, profiles, names, strict=True):
        image = ax.imshow(profile, origin="lower", aspect="auto", interpolation="nearest", cmap="inferno",
                          vmin=vmin, vmax=vmax, extent=(-0.025, times[-1] + 0.025, 0, len(profile) * 16 * um_per_pixel))
        peak = np.argmax(profile, axis=0)
        ax.plot(times, (peak + 0.5) * 16 * um_per_pixel, color="#59d9ff", lw=0.6, alpha=0.9,
                label="Brightest band (every frame)")
        ax.plot(spikes, np.full(len(spikes), 1.01), "|", transform=ax.get_xaxis_transform(),
                color="#d34b69", ms=5, clip_on=False, label="ABF spikes")
        ax.set_ylabel("Position (µm)")
        ax.set_title(name, loc="left", fontsize=11, pad=12)
        ax.set_xlim(0, frames / 20)
        ax.set_xticks(np.arange(0, frames / 20 + 0.1, 5))
        ax.tick_params(labelbottom=True)
    axes[-1].set_xlabel("Time from first processed frame (seconds)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=list(axes), shrink=0.8, label="Mean strip intensity (shared scale within figure)")
    fig.suptitle(f"{title} | full {frames / 20:.0f} seconds | 20 Hz | whole image, no ROI\n"
                 f"16-pixel bands ({16 * um_per_pixel:.1f} µm); no temporal smoothing; spikes shown as ticks above panels\n"
                 "Cyan = position of the maximum strip mean, not a tracked hotspot. Diagonal end bands contain fewer pixels.",
                 fontsize=13)
    return fig


def plot_wave_persistence(tests, title) -> Figure:
    """Plot observed directional runs against block-shuffled peak trajectories."""
    names = ("Left to right", "Top to bottom", "Top-left to bottom-right", "Bottom-left to top-right")
    fig = Figure(figsize=(13, 9), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, test, name in zip(axes.flat, tests, names, strict=True):
        blocks = np.asarray(test["block_frames"]) * 50
        ax.plot(blocks, test["null95_ms"], color="#777777", label="Shuffle 95th percentile")
        ax.plot(blocks, test["null_max_ms"], color="#e08a30", label="Shuffle maximum")
        ax.axhline(test["observed_ms"], color="#218ca8", lw=2, label="Observed")
        ax.set_title(f"{name}\nPeaks in first/last band: {test['edge_peak_fraction']:.0%}")
        ax.set_xlabel("Shuffled block duration (ms)")
        ax.set_ylabel("Mean directional run duration (ms)")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{title} | ALS peak-position persistence | full recording\n"
                 "1,000 permutations per block length; zero-velocity steps break runs; mean includes all nonzero runs.\n"
                 "Exploratory Matityahu-inspired null comparison, not their tuned event detector or a wave diagnosis.", fontsize=12)
    return fig


def plot_med_kymographs(med: np.ndarray, um_per_pixel, frame_ms, title) -> Figure:
    """Band-average unmodified MED intensities across the entire image.

    Four axes are exploratory image coordinates, not anatomical directions.
    No temporal interpolation/smoothing or statistical wave detection is used.
    """
    yy, xx = np.indices(med.shape[1:])
    axes_spec = ((1, 0, "Left to right"), (0, 1, "Top to bottom"),
                 (1, 1, "Top-left to bottom-right"), (1, -1, "Bottom-left to top-right"))
    band_px = 16
    profiles = []
    for dx, dy, label in axes_spec:
        coordinate = (dx * xx + dy * yy) / np.hypot(dx, dy)
        origin = coordinate.min()
        bins = np.floor((coordinate - origin) / band_px).astype(int)
        count = int(bins.max()) + 1
        samples = np.bincount(bins.ravel(), minlength=count)
        profile = np.full((count, len(med)), np.nan)
        for t, frame in enumerate(med):
            sums = np.bincount(bins.ravel(), weights=frame.ravel(), minlength=count)
            np.divide(sums, samples, out=profile[:, t], where=samples > 0)
        profiles.append((profile, label))
    vmin, vmax = np.nanpercentile(np.concatenate([p.ravel() for p, _ in profiles]), [1, 99])
    times = (np.arange(len(med)) - len(med) // 2) * frame_ms
    fig = Figure(figsize=(14, 11), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax, (profile, label) in zip(axes.flat, profiles, strict=True):
        image = ax.imshow(
            profile, origin="lower", aspect="auto", interpolation="nearest", cmap="inferno",
            vmin=vmin, vmax=vmax,
            extent=(times[0] - frame_ms / 2, times[-1] + frame_ms / 2,
                    0, len(profile) * band_px * um_per_pixel),
        )
        valid = np.isfinite(profile).any(axis=0)
        peaks = np.argmax(np.where(np.isfinite(profile), profile, -np.inf), axis=0)
        positions = (peaks + 0.5) * band_px * um_per_pixel
        ax.plot(times[valid], positions[valid], "o-", color="#59d9ff", ms=3, lw=0.9,
                label="Maximum band intensity")
        ax.axvline(0, color="white", ls="--", lw=1, label="Spike frame")
        ax.set_title(label)
        ax.set_xlabel("Time relative to spike frame (ms)")
        ax.set_ylabel("Position along axis (µm)")
    axes[0, 0].legend(loc="upper left", fontsize=8, facecolor="#eeeeee", framealpha=0.85)
    fig.colorbar(image, ax=list(axes.flat), shrink=0.85, label="Mean MED intensity (shared 1st–99th percentile scale)")
    fig.suptitle(
        f"{title}\nFour-axis space–time profiles | 20 Hz | band width {band_px} px ({band_px * um_per_pixel:.1f} µm)\n"
        "Whole image: all pixels included; MED intensity averaged within each band. No CAT mask.\n"
        "No temporal smoothing. Diagonal end bands contain fewer pixels. Peak trace is argmax, not a tracked object.\n"
        "Exploratory adaptation of Matityahu et al. 2023: a diagonal ridge is a candidate, not a confirmed wave.",
        fontsize=11,
    )
    return fig


def plot_directional_change(result, title) -> Figure:
    """Show fixed-sector gains/losses in consecutive exported CAT frames."""
    count = len(result["labels"])
    offsets = result["offsets"]
    fig = Figure(figsize=(19, 10), layout="constrained")
    grid = fig.add_gridspec(3, len(offsets), height_ratios=(1.25, 1, 0.8))
    cy, cx = result["center"]
    radius = result["radius"]
    angles = np.arange(count) * 2 * np.pi / count
    cmap = ListedColormap(["#151b26", "#7e8796", "#26cc96", "#ed728a"])
    limit = max(result["gain"].max(), result["loss"].max(), 1) * 1.12
    net_limit = max(np.abs(result["net"]).max(), 1) * 1.12
    for col, offset in enumerate(offsets):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(result["changes"][col], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.add_patch(Circle((cx, cy), radius, fill=False, color="white", lw=1))
        for angle in angles + np.pi / count:
            ax.plot([cx, cx + radius * np.sin(angle)], [cy, cy - radius * np.cos(angle)],
                    color="white", lw=0.6, alpha=0.7)
        ax.plot(cx, cy, "+", color="yellow", ms=9)
        ax.set_title(f"Frame {offset:+d} to {offset + 1:+d}\nChange inside circle: {result['coverage'][col]:.0%}")
        ax.set_xticks([])
        ax.set_yticks([])
        polar = fig.add_subplot(grid[1, col], projection="polar")
        polar.set_theta_zero_location("N")
        polar.set_theta_direction(-1)
        closed = np.r_[angles, angles[0]]
        for key, color in (("gain", "#149d72"), ("loss", "#d34b69")):
            polar.plot(closed, np.r_[result[key][col], result[key][col, 0]], color=color, label=key)
        polar.set_xticks(angles, result["labels"])
        polar.set_ylim(0, limit)
        polar.tick_params(labelsize=8)
        polar.set_title(f"Gain nonuniformity CV = {result['cv'][col]:.2f}", fontsize=10, pad=17)
        if col == 0:
            polar.legend(loc="lower left", bbox_to_anchor=(-0.3, -0.2), fontsize=8)
        bar = fig.add_subplot(grid[2, col])
        net = result["net"][col]
        bar.bar(result["labels"], net, color=np.where(net >= 0, "#149d72", "#d34b69"))
        bar.axhline(0, color="gray", lw=0.7)
        bar.set_ylim(-net_limit, net_limit)
        bar.set_title("Net = gain - loss", fontsize=10)
        bar.tick_params(labelsize=8)
        if col == 0:
            bar.set_ylabel("Pixels / frame")
    fig.suptitle(
        f"{title} | {count} directions | CAT-bright median response\n"
        "Green: newly bright | Pink: lost bright | Gray: retained bright | White circle: analyzed area\n"
        "Fixed spike-frame centroid; N = image top. Polar units: pixels/frame (shared scale).\n"
        "CV = SD/mean of gain per sector area; 0 = uniform. Descriptive, not an isotropy significance test.",
        fontsize=12,
    )
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


# ── Spontaneous zone maps (-> spontaneous/zone_maps/) ────────────────────────


def _tint_background(background: np.ndarray, color: str) -> np.ndarray:
    """Normalize to [0, 1]; if not gray, put it in one RGB channel."""
    bg_norm = (background - background.min()) / (background.max() - background.min())
    if color == "gray":
        return bg_norm
    bg_rgb = np.zeros((*bg_norm.shape, 3), dtype=bg_norm.dtype)
    bg_rgb[..., {"red": 0, "green": 1, "blue": 2}[color]] = bg_norm
    return bg_rgb


def _label_zone(ax: mpl.axes.Axes, zone_id: int, centroid: tuple[float, float]) -> None:
    cy, cx = centroid
    ax.text(cx, cy, str(zone_id), color="white", fontsize=9, fontweight="bold",
            ha="center", va="center", bbox={"boxstyle": "circle", "fc": "black", "alpha": 0.6})


def zone_colors(zone_ids: list[int]) -> dict[int, tuple]:
    """Fixed tab20 color per zone id (sorted order), shared by the overlay and single-zone maps."""
    palette = mpl.colormaps["tab20"].colors
    return {z: palette[i % len(palette)] for i, z in enumerate(sorted(zone_ids))}


def plot_zone_overlay(zone_masks: dict[int, np.ndarray], zone_centroids: dict[int, tuple[float, float]],
                      background: np.ndarray, bg_color: str, title: str, um_per_px: float) -> Figure:
    """All zones as translucent fills (largest painted first so small zones stay on top) + scale bar."""
    from skimage.color import label2rgb

    bg_tinted = _tint_background(background, bg_color)
    zone_label_map = np.zeros(background.shape, dtype=np.int32)
    for zone_id in sorted(zone_masks, key=lambda z: zone_masks[z].sum(), reverse=True):
        zone_label_map[zone_masks[zone_id]] = zone_id

    overlay = label2rgb(zone_label_map, image=bg_tinted, bg_label=0, alpha=0.5,
                        colors=mpl.colormaps["tab20"].colors, saturation=1)

    fig = Figure(figsize=(11, 11), layout="tight")
    ax = fig.add_subplot()
    ax.imshow(overlay)
    for zone_id in sorted(zone_masks):
        if zone_id in zone_centroids:
            _label_zone(ax, zone_id, zone_centroids[zone_id])
    _add_scale_bar(um_per_px, ax, background.shape[1], background.shape[0])
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_single_zone(zone_id: int, mask: np.ndarray, color: tuple, centroid: tuple[float, float] | None,
                     background: np.ndarray, bg_color: str, title: str, um_per_px: float) -> Figure:
    """One zone's outline over the background + scale bar."""
    fig = Figure(figsize=(11, 11), layout="tight")
    ax = fig.add_subplot()
    ax.imshow(_tint_background(background, bg_color), cmap="gray" if bg_color == "gray" else None)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2.5)
    if centroid is not None:
        _label_zone(ax, zone_id, centroid)
    _add_scale_bar(um_per_px, ax, background.shape[1], background.shape[0])
    ax.set_title(title)
    ax.axis("off")
    return fig


# ── Spontaneous zone stats (-> spontaneous/spontaneous_stats.png) ───────────

# Fixed color per sensor (color follows the entity, never its rank); validated all-pairs
# with the dataviz palette validator. Unknown sensors fall back to neutral gray.
SENSOR_COLORS = {"GACh3.0": "#2a78d6", "iAChSnFR": "#eb6834", "rACh1h": "#1baf7a"}
_INK_PRIMARY, _INK_SECONDARY, _INK_GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_zone_stats(zones, title: str) -> Figure:
    """Zone area (log) and event frequency per sensor: every zone as a dot + median/IQR box.

    zones: pooled zone table with columns recording, sensor, area_um2, mean_freq_hz.
    """
    sensors = [s for s in SENSOR_COLORS if s in set(zones["sensor"])]
    sensors += sorted(set(zones["sensor"]) - set(sensors))
    rng = np.random.default_rng(0)  # fixed jitter so reruns look identical

    fig = Figure(figsize=(11, 5), layout="constrained", facecolor=_SURFACE)
    panels = (("area_um2", "Zone area (µm², log scale)", True), ("mean_freq_hz", "Zone event frequency (Hz, log scale)", True))
    for i, (col, ylabel, log) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, facecolor=_SURFACE)
        for x, sensor in enumerate(sensors):
            rows = zones[zones["sensor"] == sensor].dropna(subset=[col])
            if log:
                rows = rows[rows[col] > 0]
            if rows.empty:
                continue
            vals = rows[col].to_numpy(dtype=float)
            flagged = rows["high_freq_flag"].to_numpy(dtype=bool) if "high_freq_flag" in rows else np.zeros(len(rows), bool)
            color = SENSOR_COLORS.get(sensor, "#8a8984")
            ax.boxplot(vals, positions=[x], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "none", "edgecolor": _INK_SECONDARY, "linewidth": 1},
                       medianprops={"color": _INK_PRIMARY, "linewidth": 2},
                       whiskerprops={"color": _INK_SECONDARY, "linewidth": 1},
                       capprops={"color": _INK_SECONDARY, "linewidth": 1})
            jitter = x + rng.uniform(-0.18, 0.18, vals.size)
            ax.scatter(jitter[~flagged], vals[~flagged], s=22, color=color, alpha=0.75,
                       edgecolors=_SURFACE, linewidths=0.8, zorder=3)
            ax.scatter(jitter[flagged], vals[flagged], s=34, facecolors=_SURFACE, edgecolors=color,
                       linewidths=1.6, zorder=4)  # hollow = high-frequency flag
        if log:
            ax.set_yscale("log")
        n_rec = zones.groupby("sensor")["recording"].nunique()
        n_zone = zones.groupby("sensor").size()
        ax.set_xticks(range(len(sensors)),
                      [f"{s}\n{n_zone.get(s, 0)} zones / {n_rec.get(s, 0)} rec." for s in sensors],
                      color=_INK_PRIMARY)
        ax.set_ylabel(ylabel, color=_INK_PRIMARY)
        ax.tick_params(colors=_INK_SECONDARY)
        ax.grid(axis="y", color=_INK_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_INK_GRID)
    fig.suptitle(title, color=_INK_PRIMARY)
    if "high_freq_flag" in zones and zones["high_freq_flag"].any():
        fig.text(0.5, -0.02, f"Hollow dots: {int(zones['high_freq_flag'].sum())} zones flagged high-frequency "
                 "(mean_freq_hz > 1 Hz, e.g. manually induced hotspots) -- kept, not removed.",
                 ha="center", va="top", color=_INK_SECONDARY, fontsize=9)
    return fig
