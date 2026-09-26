"""
Headless matplotlib export figures (plain Figure objects, no PySide6; the GUI canvas is classes/mpl_canvas.py).

  Step 1. Spatial     : hotspot-area trace + spike-4..spike+4 CAT panels + Vm overlay  (-> spatial/)
  Step 2. Flow        : striatum / CAT-mask quivers + speed (*_FLOW.png), streamlines + pattern (*_STREAMLINES.png);
                        MED shown as baseline z  (-> flow/)
  Step 3. Reliability : per-segment detection montage + Vm success vs failure  (-> reliability/)
  Step 4. Spikes      : full-length Vm with picked / skipped / collapsed spikes  (-> spikes/)
  Step 5. Prototypes  : kymographs / wave persistence used only by prototype_*.py
  Step 6. Spontaneous : zone overlay / single-zone maps + per-sensor zone stats  (-> spontaneous/)

Example:
    >>> fig = plot_flow_panels(med_stack, flow_pairs, title_info, frame_duration_ms, spike_frame_idx, striatum)
    >>> fig.savefig(out_dir / f"{stem}_FLOW.png")
"""

## Modules
# Standard library imports
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes.spatial_categorization import SpatialCategorizer

# Third-party imports
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter

# Local imports
from classes.region_analyzer import (
    MIN_DECAY_FIT_R2,
    PIXEL_SCALE,
    RegionAnalyzer,
    _decay_model,
)
from classes.spatial_categorization import BASELINE_TRIM_PCT
from functions.ap_threshold import baseline_vm, find_ap_threshold
from functions.flow_pattern import FLOW_PATTERN_BLOCK, block_mean

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 1: spatial -------------------------------------------------------
# Cluster fill/outline colors, cycled by cluster index (red, green, blue, orange, purple)
CLUSTER_RGBA = [
    (0.91, 0.30, 0.24, 0.45),
    (0.18, 0.80, 0.44, 0.45),
    (0.20, 0.60, 0.86, 0.45),
    (0.95, 0.61, 0.07, 0.45),
    (0.61, 0.35, 0.71, 0.45),
]

# --- Step 2: flow ----------------------------------------------------------
FLOW_QUIVER_STEP = 24         # px between drawn arrows
FLOW_AUTO_ARROW_FRAC = 0.9    # auto scale: each panel's p95 arrow is this fraction of FLOW_QUIVER_STEP
FLOW_STREAM_DENSITY = 2.5     # matplotlib streamplot density
FLOW_TITLE_FS = 13            # panel title font size
FLOW_SUPTITLE_FS = 16         # figure title font size
FLOW_CBAR_LABEL_FS = 13       # speed colorbar label font size
FLOW_CBAR_TICK_FS = 11        # speed colorbar tick font size
FLOW_SCALE_BAR_FS = 12        # scale-bar text font size
FLOW_CROSSHAIR_FS = 11        # 0/90/180/270° crosshair label font size
FLOW_MED_Z_LOW = 1.0          # MED rows: gray range starts at baseline + this many sigmas (black)

# --- Step 3: reliability ---------------------------------------------------
MONTAGE_NCOLS = 8                                     # panels per montage row
MONTAGE_MAX_ROWS = 4                                  # rows per montage page
MONTAGE_PAGE_SIZE = MONTAGE_NCOLS * MONTAGE_MAX_ROWS  # segments per PNG, before starting a new one
VM_WINDOW_MS = 50                                     # ± ms around each trace's own peak

# --- Step 6: spontaneous ---------------------------------------------------
# Fixed color per sensor (color follows the entity, never its rank); validated all-pairs
# with the dataviz palette validator. Unknown sensors fall back to neutral gray.
SENSOR_COLORS = {"GACh3.0": "#2a78d6", "iAChSnFR": "#eb6834", "rACh1h": "#1baf7a"}
_INK_PRIMARY, _INK_SECONDARY, _INK_GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ===========================================================================
#
#   SHARED -- SCALE BAR
#
# ===========================================================================


def _add_scale_bar(
    pixel_size_um: float,
    ax: mpl.axes.Axes,
    img_width: int,
    img_height: int,
    font_size: int | None = None,
    bar_height: int | None = None,
) -> None:
    """Lime scale bar (~20% of the image width, rounded to a nice µm value) in the bottom-right corner.

    font_size / bar_height default to values auto-scaled from the image size (px).
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


# ===========================================================================
#
#   STEP 1 -- SPATIAL  (-> spatial/)
#
# ===========================================================================


def plot_spatiotemporal_summary(
    categorizer: "SpatialCategorizer",
    region_analyzer: RegionAnalyzer,
    spike_frame_idx: int,
    title_info: dict,
    vm_segments: list[tuple[np.ndarray, np.ndarray]],
    frame_duration_ms: float,
) -> Figure:
    """3 rows: density-gated hotspot-area trace (why the critical frame was picked) +
    spike-4..spike+4 CAT panels (cluster shading on spike and spike+1) + overlaid Vm of every segment.

    Args:
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial".
        vm_segments: per-segment (time_ms_relative_to_spike_frame, Vm) pairs, from AbfClip.get_vm_segments().
        frame_duration_ms: ms per frame (e.g. AbfClip.ts_imgs * 1000), for the frame gridlines.
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
    # Same units the decay-tau fit below was fit on (hotspot_area_um2) -- see RegionAnalyzer._compute_hotspot_area_trace.
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
            # spike and spike+1 frames each get their own independent cluster shading
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
    """'hotspots: <µm²> (<%>)' for the spike / spike+1 panel titles; None for other panels (no label_frame)."""
    if label_frame is None:
        return None
    hotspot_px = np.count_nonzero(label_frame >= 0)
    hotspot_um2 = hotspot_px * um_per_pixel ** 2
    hotspot_pct = 100.0 * hotspot_px / label_frame.size
    return f"hotspots: {hotspot_um2:.0f} µm² ({hotspot_pct:.1f}%)"


def _draw_cluster_shading(ax: mpl.axes.Axes, label_frame: np.ndarray, centroids: list[tuple[float, float]]) -> None:
    """Translucent per-cluster fill (DBSCAN label map) + centroid cross and index label."""
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
    """Dashed exponential decay curve over the post-peak hotspot-area trace, with tau/R² in the label.

    Draws only a "fit failed" note if fit_decay_tau() couldn't fit (see RegionAnalyzer.__init__).
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


# ===========================================================================
#
#   STEP 2 -- FLOW  (-> flow/)
#
# ===========================================================================

# --- 2-shared. MED z display range -----------------------------------------


def _med_z_display(med_stack: np.ndarray, flow_pairs: list[dict],
                   spike_frame_idx: int) -> tuple[np.ndarray, tuple[float, float]]:
    """MED as z (trimmed baseline mean / std of the pre-spike frames) + gray range (FLOW_MED_Z_LOW, upper).

    upper = median over the flow pairs of each idx_from frame's max z inside its keep_mask; pairs with an
    empty keep_mask are skipped (all empty -> max z over the idx_from frames).
    """
    baseline = med_stack[:spike_frame_idx].astype(np.float64).ravel()
    lo, hi = np.percentile(baseline, BASELINE_TRIM_PCT)
    kept = baseline[(baseline >= lo) & (baseline <= hi)]
    z_stack = (med_stack - kept.mean()) / kept.std()

    maxima = [float(z_stack[p["idx_from"]][p["keep_mask"]].max()) for p in flow_pairs if p["keep_mask"].any()]
    if not maxima:
        maxima = [float(z_stack[p["idx_from"]].max()) for p in flow_pairs] or [FLOW_MED_Z_LOW + 1.0]
    upper = max(float(np.median(maxima)), FLOW_MED_Z_LOW + 1e-3)
    return z_stack, (FLOW_MED_Z_LOW, upper)


def _add_med_z_colorbar(fig: Figure, axes_row: list, z_range: tuple[float, float]) -> None:
    """Gray colorbar for the MED z rows."""
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*z_range), cmap="gray")
    cbar = fig.colorbar(sm, ax=axes_row, shrink=0.8)
    cbar.set_label("MED z (baseline σ)", fontsize=FLOW_CBAR_LABEL_FS)
    cbar.ax.tick_params(labelsize=FLOW_CBAR_TICK_FS)


# --- 2a. quivers + speed (*_FLOW.png) --------------------------------------


def _draw_flow_quivers(ax: mpl.axes.Axes, background: np.ndarray, pair: dict, draw: np.ndarray,
                       grid_y: np.ndarray, grid_x: np.ndarray, um_per_pixel: float,
                       value_range: tuple[float, float]) -> None:
    """Gray background (MED frame or CAT mask, shown over value_range) + red auto-scaled arrows at the grid points
    in `draw` + scale bar."""
    ax.imshow(background, cmap="gray", origin="upper", vmin=value_range[0], vmax=value_range[1])
    if draw.any():
        u_grid, v_grid = pair["u"][grid_y, grid_x][draw], pair["v"][grid_y, grid_x][draw]
        scale = max(float(np.percentile(np.hypot(u_grid, v_grid), 95)), 1e-3) / (FLOW_AUTO_ARROW_FRAC * FLOW_QUIVER_STEP)
        ax.quiver(
            grid_x[draw], grid_y[draw], u_grid, v_grid,
            color="red", angles="xy", scale_units="xy", scale=scale, width=0.003,
            headwidth=2.5, headlength=3, headaxislength=2.5,
        )
    _add_scale_bar(um_per_pixel, ax, background.shape[1], background.shape[0], font_size=FLOW_SCALE_BAR_FS)


def plot_flow_panels(
    med_stack: np.ndarray,
    flow_pairs: list[dict],
    title_info: dict,
    frame_duration_ms: float,
    spike_frame_idx: int,
    striatum: np.ndarray | None = None,
) -> Figure:
    """One column per flow pair; rows: striatum arrows on MED z, CAT-mask arrows on the CAT mask, striatum speed (µm/s).

    striatum None (no outline, e.g. 40X / 60X) -> rows 1 and 3 cover the full FOV. Row 1 shows the MED as baseline z
    (range from _med_z_display()). Row-2 arrows sit at grid points within FLOW_QUIVER_STEP of keep_mask (a dilation,
    so a coarse grid can't miss a thin hotspot). Arrows auto-scale per panel; speed panels share one color scale.

    Args:
        flow_pairs: dicts with "label", "idx_from", "u", "v", "keep_mask" (from compute_flow_pairs()).
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial".
        frame_duration_ms: imaging frame duration, converts px/frame -> µm/s.
        spike_frame_idx: spike frame in med_stack; earlier frames are the z baseline.
        striatum: (H, W) bool display mask for rows 1 and 3.
    """
    n_panels = max(len(flow_pairs), 1)
    fig = Figure(figsize=(6 * n_panels, 18), dpi=110, layout="constrained")
    axes = fig.subplots(3, n_panels, squeeze=False)
    height, width = med_stack.shape[1], med_stack.shape[2]
    grid_y, grid_x = np.mgrid[0:height:FLOW_QUIVER_STEP, 0:width:FLOW_QUIVER_STEP]
    um_per_pixel = 1.0 / PIXEL_SCALE[title_info["obj"]]
    px_per_frame_to_um_per_s = um_per_pixel * 1000.0 / frame_duration_ms
    z_stack, z_range = _med_z_display(med_stack, flow_pairs, spike_frame_idx)
    region = "striatum" if striatum is not None else "full FOV"
    in_region = striatum[grid_y, grid_x] if striatum is not None else np.ones(grid_y.shape, dtype=bool)

    speeds = [np.hypot(p["u"], p["v"]) * px_per_frame_to_um_per_s for p in flow_pairs]
    if striatum is not None:
        speeds = [np.where(striatum, s, np.nan) for s in speeds]
    vmax = float(np.nanpercentile(np.concatenate([s.ravel() for s in speeds]), 99.5)) if speeds else 1.0
    im = None

    for i, (pair, speed) in enumerate(zip(flow_pairs, speeds, strict=True)):
        z_frame = z_stack[pair["idx_from"]]
        near_bright = maximum_filter(pair["keep_mask"], size=FLOW_QUIVER_STEP)[grid_y, grid_x]

        _draw_flow_quivers(axes[0, i], z_frame, pair, in_region, grid_y, grid_x, um_per_pixel, z_range)
        axes[0, i].set_title(f"{pair['label']}  ({region})", fontsize=FLOW_TITLE_FS)
        _draw_flow_quivers(axes[1, i], pair["keep_mask"], pair, near_bright, grid_y, grid_x, um_per_pixel, (0, 1))
        axes[1, i].set_title(f"{pair['label']}  (CAT mask)", fontsize=FLOW_TITLE_FS)
        im = axes[2, i].imshow(speed, cmap="magma", vmin=0, vmax=vmax)
        axes[2, i].set_title(f"flow speed (µm/s), {region}", fontsize=FLOW_TITLE_FS)
        _add_scale_bar(um_per_pixel, axes[2, i], width, height, font_size=FLOW_SCALE_BAR_FS)
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])

    _add_med_z_colorbar(fig, axes[0, :].tolist(), z_range)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[2, :].tolist(), shrink=0.8)
        cbar.set_label("µm/s", fontsize=FLOW_CBAR_LABEL_FS)
        cbar.ax.tick_params(labelsize=FLOW_CBAR_TICK_FS)
    fig.suptitle(
        f"Flow Analysis: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
        f"  (unmasked TV-L1; rows 1 + 3 in {region}, row 2 on the CAT mask)",
        fontsize=FLOW_SUPTITLE_FS,
    )
    return fig


# --- 2b. streamlines + pattern (*_STREAMLINES.png) -------------------------


def _draw_flow_streamlines(ax: mpl.axes.Axes, background: np.ndarray, pair: dict, region: np.ndarray | None,
                           um_per_pixel: float, value_range: tuple[float, float]) -> None:
    """Gray background (MED frame or CAT mask, shown over value_range) + red streamlines (block-averaged flow;
    only blocks touching `region`, all blocks if None) + scale bar."""
    height, width = background.shape
    ax.imshow(background, cmap="gray", origin="upper", vmin=value_range[0], vmax=value_range[1])
    u_small = block_mean(pair["u"])
    v_small = block_mean(pair["v"])
    if region is not None:
        keep_small = block_mean(region.astype(float)) > 0
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
    _add_scale_bar(um_per_pixel, ax, width, height, font_size=FLOW_SCALE_BAR_FS)


def _crosshair_labels(orientation: dict | None) -> dict[str, str]:
    """Arm label per image side: D / V / M / L from the bd entry's "dorsal" / "medial", else 0° / 90° / 180° / 270°."""
    if orientation is None:
        return {"right": "0°", "up": "90°", "left": "180°", "down": "270°"}
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    dorsal, medial = orientation["dorsal"], orientation["medial"]
    return {dorsal: "D", opposite[dorsal]: "V", medial: "M", opposite[medial]: "L"}


def _draw_angle_crosshair(ax: mpl.axes.Axes, width: int, height: int, arm_labels: dict[str, str]) -> None:
    """Dashed crosshair at the frame centre with a label at each arm end (arm_labels: image side -> text)."""
    cx, cy, arm = width / 2, height / 2, 0.3 * min(width, height)
    style = {"color": "cyan", "linewidth": 1.5, "linestyle": "--", "zorder": 5}
    ax.plot([cx - arm, cx + arm], [cy, cy], **style)
    ax.plot([cx, cx], [cy - arm, cy + arm], **style)
    bbox = {"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
    ends = {"right": (cx + arm, cy), "up": (cx, cy - arm), "left": (cx - arm, cy), "down": (cx, cy + arm)}
    for side, (x, y) in ends.items():
        ax.text(x, y, arm_labels[side], ha="center", va="center", fontsize=FLOW_CROSSHAIR_FS, color="teal",
                fontweight="bold", bbox=bbox, zorder=6)


def _pattern_title(pair: dict) -> str:
    """' · source' / ' · sink' / ' · anisotropic L 25° D' (DV / ML, else ' · anisotropic 295°'); '' if unfitted."""
    pattern = pair.get("pattern", {})
    label = pattern.get("label")
    if label is None:
        return ""
    if label == "anisotropic":
        if "dv_ml" in pair:
            pole, tilt, toward = pair["dv_ml"]
            return f" · anisotropic {pole} {tilt:.0f}° {toward}" if toward else f" · anisotropic {pole}"
        return f" · anisotropic {pattern['drift_angle_deg']:.0f}°"
    return f" · {label}"


def plot_flow_streamlines(
    med_stack: np.ndarray,
    flow_pairs: list[dict],
    title_info: dict,
    spike_frame_idx: int,
    striatum: np.ndarray | None = None,
    orientation: dict | None = None,
) -> Figure:
    """One column per flow pair; rows: striatum streamlines on MED z, CAT-mask streamlines on the CAT mask
    (title = CAT-fit pattern).

    striatum None (no outline, e.g. 40X / 60X) -> row 1 covers the full FOV. Row 1 shows the MED as baseline z
    (range from _med_z_display()). Streamlines use FLOW_PATTERN_BLOCK block-averaged u, v; anisotropic row-2
    panels get a D / V / M / L crosshair (0/90/180/270° without orientation).

    Args:
        flow_pairs: dicts with "label", "idx_from", "u", "v", "keep_mask", "pattern" (compute_flow_pairs + fit_flow_pattern),
            optional "dv_ml" (pole, tilt_deg, toward) for the title.
        title_info: dict with keys "animal_id", "slice", "at", "obj", "tiff_serial", "abf_serial".
        spike_frame_idx: spike frame in med_stack; earlier frames are the z baseline.
        striatum: (H, W) bool display mask for row 1.
        orientation: bd entry with "dorsal" / "medial" image sides (e.g. "right" / "up").
    """
    n_panels = max(len(flow_pairs), 1)
    fig = Figure(figsize=(6 * n_panels, 12), dpi=110, layout="constrained")
    axes = fig.subplots(2, n_panels, squeeze=False)
    um_per_pixel = 1.0 / PIXEL_SCALE[title_info["obj"]]
    z_stack, z_range = _med_z_display(med_stack, flow_pairs, spike_frame_idx)
    region = "striatum" if striatum is not None else "full FOV"
    arm_labels = _crosshair_labels(orientation)

    for i, pair in enumerate(flow_pairs):
        frame = z_stack[pair["idx_from"]]
        _draw_flow_streamlines(axes[0, i], frame, pair, striatum, um_per_pixel, z_range)
        axes[0, i].set_title(f"{pair['label']}  ({region})", fontsize=FLOW_TITLE_FS)
        _draw_flow_streamlines(axes[1, i], pair["keep_mask"], pair, pair["keep_mask"], um_per_pixel, (0, 1))
        if pair.get("pattern", {}).get("label") == "anisotropic":
            _draw_angle_crosshair(axes[1, i], frame.shape[1], frame.shape[0], arm_labels)
        axes[1, i].set_title(f"{pair['label']}  (CAT mask){_pattern_title(pair)}", fontsize=FLOW_TITLE_FS)
        for ax in axes[:, i]:
            ax.set_xticks([])
            ax.set_yticks([])

    _add_med_z_colorbar(fig, axes[0, :].tolist(), z_range)
    fig.suptitle(
        f"Flow Streamlines: {title_info['animal_id']} {title_info['slice']} {title_info['at']} "
        f"{title_info['obj']} TIFF_{title_info['tiff_serial']} ABF_{title_info['abf_serial']}"
        f"  (unmasked TV-L1; row 1 in {region}, row 2 on the CAT mask)",
        fontsize=FLOW_SUPTITLE_FS,
    )
    return fig


# ===========================================================================
#
#   STEP 3 -- RELIABILITY  (-> reliability/)
#
# ===========================================================================

# --- 3a. per-segment detection montage (*_RELIABILITY.png) -----------------


def plot_segment_reliability_montage(
    seg_results: list[dict],
    rec_stem: str,
    window_px: int,
    density_thresh: float,
    sigma_mult: float,
    obj: str,
) -> list[Figure]:
    """One panel per raw segment: bright mask + cluster shading, green (detected) / red border, winning frame.

    Pages of MONTAGE_PAGE_SIZE (4x8) panels -> one Figure per page; every suptitle shows the overall reliability%.

    Args:
        seg_results: dicts from SpikeReliabilityChecker.check() ("detected", "frame_offset", "bright_mask",
            "label_frame", "centroids", "n_clusters").
        window_px, density_thresh, sigma_mult: detection settings, for the title only.
        obj: objective ("10X" / "40X" / "60X"), for the per-panel scale bar.
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


# --- 3b. Vm success vs failure (*_VM_SUCCESS_FAIL.png) ---------------------


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


# ===========================================================================
#
#   STEP 4 -- SPIKES  (-> spikes/)
#
# ===========================================================================


def plot_spike_detection_summary(
    rec_time: np.ndarray,
    vm: np.ndarray,
    picked: tuple[np.ndarray, np.ndarray],
    skipped: tuple[np.ndarray, np.ndarray],
    collapsed: tuple[np.ndarray, np.ndarray],
    title: str,
) -> Figure:
    """Full-length Vm trace (s) with picked / skipped / collapsed spikes marked.

    Args:
        picked: (times, values) of spikes kept for analysis.
        skipped: (times, values) of spikes dropped for insufficient baseline margin.
        collapsed: (times, values) of extra spikes sharing a frame with an earlier spike.
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


# ===========================================================================
#
#   STEP 5 -- PROTOTYPES  (prototype_*.py only; not used by the pipeline or GUI)
#
# ===========================================================================

# --- 5a. kymographs / wave persistence -------------------------------------


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


# ===========================================================================
#
#   STEP 6 -- SPONTANEOUS  (-> spontaneous/)
#
# ===========================================================================

# --- 6a. zone maps (-> spontaneous/{stem}_ZONE_MAPS.tif) -------------------


def _label_zone(ax: mpl.axes.Axes, zone_id: int, centroid: tuple[float, float]) -> mpl.text.Text:
    """Zone id in a black circle at the zone centroid (row, col)."""
    cy, cx = centroid
    return ax.text(cx, cy, str(zone_id), color="white", fontsize=9, fontweight="bold",
                   ha="center", va="center", bbox={"boxstyle": "circle", "fc": "black", "alpha": 0.6})


def zone_colors(zone_ids: list[int]) -> dict[int, tuple]:
    """Fixed tab20 color per zone id (sorted order), shared by the overlay and single-zone maps."""
    palette = mpl.colormaps["tab20"].colors
    return {z: palette[i % len(palette)] for i, z in enumerate(sorted(zone_ids))}


def _z_page(z_image: np.ndarray, vmin: float, vmax: float, title: str, um_per_px: float) -> tuple[Figure, mpl.axes.Axes]:
    """Gray z-score image on the shared (vmin, vmax) scale + 'z' colorbar + scale bar."""
    fig = Figure(figsize=(11, 11), layout="tight")
    ax = fig.add_subplot()
    image = ax.imshow(z_image, cmap="gray", vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, shrink=0.8, label="z")
    _add_scale_bar(um_per_px, ax, z_image.shape[1], z_image.shape[0])
    ax.set_title(title)
    ax.axis("off")
    return fig, ax


def plot_zone_overview(z_image: np.ndarray, zone_masks: dict[int, np.ndarray],
                       zone_centroids: dict[int, tuple[float, float]], colors: dict[int, tuple], vmin: float,
                       vmax: float, title: str, um_per_px: float, striatum_outline: np.ndarray | None = None,
                       axis_labels: tuple[str, str] | None = None) -> Figure:
    """All zones as translucent fills (largest painted first so small zones stay on top) over the z image.

    striatum_outline: closed (N, 2) x / y polygon from the Striatum Boundary export, drawn white dashed.
    axis_labels: (x, y) anatomical direction labels; shown without ticks or frame.
    """
    fig, ax = _z_page(z_image, vmin, vmax, title, um_per_px)
    fill = np.zeros((*z_image.shape, 4))
    for zone_id in sorted(zone_masks, key=lambda z: zone_masks[z].sum(), reverse=True):
        fill[zone_masks[zone_id]] = (*colors[zone_id][:3], 0.5)
    ax.imshow(fill)
    if striatum_outline is not None:
        limits = ax.get_xlim(), ax.get_ylim()
        closed = np.vstack([striatum_outline, striatum_outline[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="white", ls="--", lw=1.5)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])  # the outline sits on the pixel border; keep the image extent
    if axis_labels is not None:
        ax.axis("on")  # axis("off") would hide the labels too
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel(axis_labels[0], fontsize=12)
        ax.set_ylabel(axis_labels[1], fontsize=12)
    for zone_id in sorted(zone_masks):
        if zone_id in zone_centroids:
            _label_zone(ax, zone_id, zone_centroids[zone_id])
    return fig


def frame_zone_figures(pages, shape: tuple[int, int],
                       zone_masks: dict[int, np.ndarray], zone_centroids: dict[int, tuple[float, float]],
                       colors: dict[int, tuple], vmin: float, vmax: float, um_per_px: float) -> Iterator[Figure]:
    """One frame per page: contours of the zones hit in this frame + thin white outline of the frame's own hotspots.

    pages: (z_frame, frame_zone_ids, hotspot_mask, title) per frame. Yields the SAME Figure, updated per page --
    render it before pulling the next one. Figure, colorbar and zone contours / labels are built once.
    """
    fig, ax = _z_page(np.zeros(shape, dtype=np.float32), vmin, vmax, "", um_per_px)
    image = ax.images[0]
    zone_artists = {}
    for zone_id in sorted(zone_masks):
        artists = [ax.contour(zone_masks[zone_id].astype(float), levels=[0.5], colors=[colors[zone_id]], linewidths=2.5)]
        if zone_id in zone_centroids:
            artists.append(_label_zone(ax, zone_id, zone_centroids[zone_id]))
        for artist in artists:
            artist.set_visible(False)
        zone_artists[zone_id] = artists

    for z_frame, frame_zone_ids, hotspot_mask, title in pages:
        image.set_data(z_frame)
        ax.set_title(title)
        for zone_id, artists in zone_artists.items():
            for artist in artists:
                artist.set_visible(zone_id in frame_zone_ids)
        rows, cols = np.nonzero(hotspot_mask)  # contour only the hotspots' bounding box (+1 px) -> same outline
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, shape[0])
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, shape[1])
        hotspots = ax.contour(np.arange(c0, c1), np.arange(r0, r1), hotspot_mask[r0:r1, c0:c1].astype(float),
                              levels=[0.5], colors="white", linewidths=1)
        yield fig
        hotspots.remove()


# --- 6b. zone stats (not exported for now) ---------------------------------


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
