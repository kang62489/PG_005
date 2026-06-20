## Modules
# Standard library imports
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from classes.spatial_categorization import SpatialCategorizer

# Third-party imports
import matplotlib as mpl
import numpy as np
import polars as pl
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PySide6.QtCore import QTimer
from PySide6.QtGui import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from rich.console import Console

from classes.region_analyzer import RegionAnalyzer

# Local application imports

# Set backend to QtAgg for interactive plotting
mpl.use("QtAgg")

# Set save dialog to remember last directory
mpl.rcParams["savefig.directory"] = ""

# Set rich console
cs = Console()


def center_on_screen(window: QMainWindow) -> None:
    screen = window.screen()
    screen_geometry = screen.availableGeometry()
    window_geometry = window.frameGeometry()
    window_geometry.moveCenter(screen_geometry.center())
    window.move(window_geometry.topLeft())


# customized toolbar
class CustomToolbar(NavigationToolbar):
    toolitems: ClassVar[list[tuple[str, str, str, str]]] = [("Save", "Save the figure", "filesave", "save_figure")]


class WindowToolbar(QToolBar):
    """Toolbar with save button that captures the entire window."""

    last_directory: ClassVar[str] = ""

    def __init__(self, window: QMainWindow, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.window = window
        save_icon = self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton)
        self.addAction(save_icon, "Save", self.save_window)

    def save_window(self) -> None:
        """Save the entire window as an image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Window", WindowToolbar.last_directory, "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)"
        )
        if file_path:
            from pathlib import Path

            WindowToolbar.last_directory = str(Path(file_path).parent)
            pixmap = self.window.grab()
            pixmap.save(file_path)


class MplCanvas(FigureCanvasQTAgg):
    """Class to create a canvas for matplotlib plots."""

    def __init__(self, parent: QWidget | None = None, width: int = 14, height: int = 4, dpi: int = 100) -> None:
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


# Common functions



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


# ── Static export figures ───────────────────────────────────────────────────


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

    fig = Figure(figsize=(2.4 * n_cols, 7), dpi=100)
    gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 1.2], wspace=0.0)

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
    """Row-1 panel: one frame's categorized image + its own independent largest bright/dim regions."""
    cat_frame = categorizer.categorized_frames[frame_idx]
    regions = region_analyzer.find_largest_regions(cat_frame)
    bright = regions["bright"]
    dim = regions["dim"]

    cmap_cat = ListedColormap(["black", "gray", "white"])
    ax.imshow(cat_frame, cmap=cmap_cat, vmin=0, vmax=2, interpolation="nearest")
    _overlay_region(ax, bright, contour_color="magenta", span_color="yellow", centroid_color="black")
    _overlay_region(ax, dim, contour_color="cyan", span_color="lime", centroid_color="white")

    bright_line = (
        f"Bright [x (µm), y (µm), area (µm²)];\n"
        f"({bright['x_span_um']:.1f}, {bright['y_span_um']:.1f}, {bright['area_um2']:.1f})"
        if bright is not None
        else "Bright [x (µm), y (µm), area (µm²)];\nnone detected"
    )
    dim_line = (
        f"Dim [x (µm), y (µm), area (µm²)];\n"
        f"({dim['x_span_um']:.1f}, {dim['y_span_um']:.1f}, {dim['area_um2']:.1f})"
        if dim is not None
        else "Dim [x (µm), y (µm), area (µm²)];\nnone detected"
    )
    frame_label = "(SPIKE) Frame 0" if offset == 0 else f"Frame {offset:+d}"
    ax.set_title(
        f"{frame_label}\n{bright_line}\n{dim_line}",
        fontsize=8,
        fontweight="bold" if offset == 0 else "normal",
        color="red" if offset == 0 else "black",
    )
    ax.axis("off")

    legend_elements = [
        Line2D([], [], color="magenta", linewidth=1.5, label="Bright contour"),
        Line2D([], [], color="cyan", linewidth=1.5, label="Dim contour"),
        Line2D(
            [], [], marker="+", color="black", linestyle="", markersize=8, markeredgewidth=2, label="Bright centroid"
        ),
        Line2D([], [], marker="+", color="white", linestyle="", markersize=8, markeredgewidth=2, label="Dim centroid"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=5)
    _add_scale_bar(um_per_pixel, ax, cat_frame.shape[1], cat_frame.shape[0])


def _overlay_region(
    ax: mpl.axes.Axes,
    region: dict | None,
    contour_color: str,
    span_color: str,
    centroid_color: str,
) -> None:
    """Draw a region's contour, centroid, and x/y-span crosshair on ax."""
    if region is None:
        return

    contour = region["contour"]
    if contour is not None:
        ax.plot(contour[:, 1], contour[:, 0], color=contour_color, linewidth=1.5)

    y, x = region["centroid"]
    ax.scatter(x, y, c=centroid_color, s=60, marker="+", linewidths=2, zorder=20)

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

    if bright_peak_rel is not None and dim_peak_rel is not None:
        latency_ms = (dim_peak_rel - bright_peak_rel) * frame_duration_ms
        latency_line = f"Peak Latency: {latency_ms:.1f} ms"
    else:
        latency_line = "Peak Latency: N/A (region not detected)"

    ax.set_xlim(-window, window)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Mean z-score")
    ax.set_title(f"Temporal change of bright and dim area\n{latency_line}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def _nanargmax_relative(trace: np.ndarray, spike_frame_idx: int) -> int | None:
    """Index (relative to the spike frame) of trace's peak, or None if trace is all-NaN."""
    if np.all(np.isnan(trace)):
        return None
    return int(np.nanargmax(trace)) - spike_frame_idx


class PlotPeaks(QMainWindow):
    """Class to plot results from cluster analysis."""

    def __init__(
        self, df_list: list, title: str = "untitled", xlabel: str = "Time (ms)", ylabel: str = "Voltage (mV)"
    ) -> None:
        super().__init__()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.df_list = df_list
        self.MAX_ACTIONS = 6

        self.layout_main = QVBoxLayout()

        plotted_canvas_widget = self.plotting()

        self.layout_main.addWidget(plotted_canvas_widget)

        # A widget to hold everything
        w_main = QWidget()
        w_main.setLayout(self.layout_main)

        # Set the central widget of the Window.
        self.setCentralWidget(w_main)
        self.setWindowTitle(self.title)  # Set window title
        self.show()
        center_on_screen(self)

    def plotting(self) -> QWidget:
        """Plot all the dataframes in the list."""
        layout_plotting = QVBoxLayout()  # layout for hold navigation toolbar and canvas

        canvas_0 = MplCanvas()
        self.canvas = canvas_0  # Store canvas for later access
        self.fig = canvas_0.figure  # Store figure for saving

        df_vm: pl.DataFrame = self.df_list[0]
        df_peaks: pl.DataFrame = self.df_list[1]
        canvas_0.axes.plot(df_vm["Time"].to_numpy(), df_vm["Vm"].to_numpy(), color="blue", label="Vm")
        canvas_0.axes.scatter(
            df_peaks["Time"].to_numpy(),
            df_peaks["Peaks"].to_numpy(),
            marker="o",
            facecolors="none",
            edgecolors="red",
            label="Spikes",
        )
        canvas_0.axes.set_title(self.title)
        canvas_0.axes.set_xlabel(self.xlabel)
        canvas_0.axes.set_ylabel(self.ylabel)
        canvas_0.axes.legend()
        canvas_0.axes.minorticks_on()
        canvas_0.axes.grid(True, which="major")
        canvas_0.axes.grid(True, which="minor", alpha=0.3)

        toolbar = NavigationToolbar(canvas_0, self)
        # Disable the problematic configure subplots button
        actions = toolbar.actions()
        if len(actions) > self.MAX_ACTIONS:  # Configure subplots is usually the 7th action
            toolbar.removeAction(actions[6])
        layout_plotting.addWidget(toolbar)
        layout_plotting.addWidget(canvas_0)

        holding_widget = QWidget()
        holding_widget.setLayout(layout_plotting)
        return holding_widget


class PlotSegs(QMainWindow):
    """Class to plot image segments."""

    def __init__(
        self,
        lst_img_segments: list[np.ndarray],
        lst_time_segments: list[np.ndarray],
        lst_abf_segments: list[np.ndarray],
        df_picked_spikes: pl.DataFrame,
        title: str = "Aligned Segments (Image, ABF)",
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.lst_img_segments = lst_img_segments
        self.lst_time_segments = lst_time_segments
        self.lst_abf_segments = lst_abf_segments
        self.df_picked_spikes = df_picked_spikes
        self.MAX_ACTIONS = 6

        self.lo_main = QVBoxLayout()
        self.w_tools = QWidget()
        self.w_tools.setLayout(QHBoxLayout())
        self.lo_main.addWidget(self.w_tools, 0, Qt.AlignCenter)

        self.cb_seg = QComboBox()
        self.cb_seg.addItems(
            f"Spike {idx} @ Frame {row['Spike_Frame_Index']}"
            for idx, row in enumerate(self.df_picked_spikes.iter_rows(named=True))
        )
        self.w_tools.layout().addWidget(self.cb_seg, 0, Qt.AlignCenter)

        # Add play button
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.w_tools.layout().addWidget(self.btn_play)

        # Timer for auto-play
        self.play_timer = QTimer()
        self.play_timer.setInterval(200)  # 1 second per frame

        self.sw_plots = QStackedWidget()
        self.lo_main.addWidget(self.sw_plots)

        w_main = QWidget()
        w_main.setLayout(self.lo_main)
        self.setCentralWidget(w_main)

        self.plotting()
        self.connect_signals()

        # Set initial page to 0
        self.sw_plots.setCurrentIndex(0)
        self.cb_seg.setCurrentIndex(0)

        self.show()
        center_on_screen(self)

    def connect_signals(self) -> None:
        self.cb_seg.currentIndexChanged.connect(self.switch_page)
        self.btn_play.toggled.connect(self.toggle_play)
        self.play_timer.timeout.connect(self.play_next)

    def switch_page(self, idx: int) -> None:
        """Switch to a different page."""
        if idx >= 0:  # Prevent invalid index
            self.sw_plots.setCurrentIndex(idx)

    def toggle_play(self) -> None:
        """Start or stop auto-play."""
        checked = self.btn_play.isChecked()
        if checked:
            self.btn_play.setText("⏸ Pause")
            self.play_timer.start()
        else:
            self.btn_play.setText("▶ Play")
            self.play_timer.stop()

    def play_next(self) -> None:
        """Advance to the next segment during playback."""
        current_idx = self.cb_seg.currentIndex()
        next_idx = current_idx + 1

        if next_idx >= self.cb_seg.count():
            # Reached the end, stop playing
            self.btn_play.setChecked(False)
        else:
            # Move to next segment - this triggers currentIndexChanged signal
            self.cb_seg.setCurrentIndex(next_idx)

    def plotting(self) -> None:
        # Calculate percentile-based vmin/vmax across all segments (ignores outliers)
        all_data = np.concatenate([seg.flatten() for seg in self.lst_img_segments])
        vmin, vmax = np.percentile(all_data, [1, 99])

        for idx, (img_seg, time_seg, abf_seg) in enumerate(
            zip(self.lst_img_segments, self.lst_time_segments, self.lst_abf_segments, strict=True)
        ):
            # Get the spike info for this segment
            spike_row = self.df_picked_spikes.row(idx, named=True)
            spike_frame = spike_row["Spike_Frame_Index"]
            interval = spike_row["Set_Interval_Frames"]

            # Calculate actual frame indices for this segment
            left_frame = spike_frame - interval
            right_frame = spike_frame + interval
            actual_frame_indices = list(range(left_frame, right_frame + 1))

            # Get number of frames in this segment
            n_frames = img_seg.shape[0]

            # Create figure with custom canvas for THIS segment
            fig = Figure(figsize=(3 * n_frames, 8), dpi=100)
            canvas = FigureCanvasQTAgg(fig)

            # Create GridSpec: 2 rows (images on top, voltage on bottom)
            gs = GridSpec(2, n_frames, figure=fig, height_ratios=[1, 0.6], hspace=0.3, wspace=0.0)

            # Top row: Plot image frames with ACTUAL frame numbers
            for frame_idx in range(n_frames):
                ax_img = fig.add_subplot(gs[0, frame_idx])
                ax_img.imshow(img_seg[frame_idx], cmap="gray", vmin=vmin, vmax=vmax)

                # Use actual frame index and highlight spike frame
                actual_idx = actual_frame_indices[frame_idx]
                if actual_idx == spike_frame:
                    ax_img.set_title(f"Frame {actual_idx}\n(SPIKE)", fontweight="bold", color="red")
                else:
                    ax_img.set_title(f"Frame {actual_idx}")
                ax_img.axis("off")

            # Bottom row: Plot voltage trace (spans all columns)
            ax_vm = fig.add_subplot(gs[1, :])
            # Convert time from seconds to milliseconds
            time_seg_ms = time_seg * 1000
            ax_vm.plot(time_seg_ms, abf_seg, label="Vm", color="blue")
            ax_vm.set_xlabel("Time (ms)")
            ax_vm.set_ylabel("Vm (mV)")
            ax_vm.set_title(f"Spike {idx} @ Frame {spike_frame}")

            # Set time axis limits to match the frame timing
            ax_vm.set_xlim(time_seg_ms[0], time_seg_ms[-1])

            # Draw vertical lines for each frame (one line per frame at the start)
            samples_per_frame = len(time_seg_ms) // n_frames
            for frame_idx in range(n_frames):
                time_idx = frame_idx * samples_per_frame
                time_boundary = time_seg_ms[time_idx]

                # Highlight spike frame in red, others in gray
                if actual_frame_indices[frame_idx] == spike_frame:
                    ax_vm.axvline(
                        x=time_boundary, color="red", linestyle="-", linewidth=2, alpha=0.7, label="Spike Frame"
                    )
                else:
                    ax_vm.axvline(x=time_boundary, color="green", linestyle=":", linewidth=2, alpha=0.7)

            ax_vm.legend()
            ax_vm.minorticks_on()
            ax_vm.grid(True, which="major")
            ax_vm.grid(True, which="minor", alpha=0.3)

            # Add canvas as a new PAGE in the stacked widget
            self.sw_plots.addWidget(canvas)
            cs.print(f"Added segment {idx} to stacked widget")

            fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, hspace=0.1, wspace=0)


class PlotSpatialDist(QMainWindow):
    def __init__(
        self,
        categorizor: "SpatialCategorizer",
        spike_traces: list[tuple[np.ndarray, np.ndarray]],
        title: str = "Spatial Distribution",
        obj: str = "10X",
        zscore_range: tuple[float, float] | None = None,
        exp_date: str | None = None,
        abf_serial: str | None = None,
        img_serial: str | None = None,
        n_spikes: int | None = None,
        *,
        show: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.sc_ins = categorizor
        self.spike_traces = spike_traces
        self.obj = obj
        self.zscore_range = zscore_range  # (vmin, vmax) for consistent color scaling
        self.exp_date = exp_date
        self.abf_serial = abf_serial
        self.img_serial = img_serial
        self.n_spikes = n_spikes

        self.ra_ins = RegionAnalyzer(self.sc_ins.categorized_frames, obj=obj)

        self.lo_main = QVBoxLayout()

        self.w_main = QWidget()
        self.w_main.setLayout(self.lo_main)
        self.setCentralWidget(self.w_main)

        self.plotting()

        if show:
            self.show()
            center_on_screen(self)

    def plotting(self) -> None:
        if not self.sc_ins.categorized_frames:
            msg = "No results to plot. Call fit() first."
            raise RuntimeError(msg)

        # Get number of frames in this segment
        n_frames = len(self.sc_ins.source_frames)

        # Create figure with custom canvas
        fig = Figure(figsize=(8 * n_frames, 8), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(1400, 800)
        mpl_toolbar = CustomToolbar(canvas, self)

        gs = GridSpec(
            3,
            n_frames + 1,
            figure=fig,
            height_ratios=[1, 1, 0.8],
            width_ratios=[1] * n_frames + [0.05],
            hspace=0.4,
            wspace=0.05,
        )

        cmap_cat = ListedColormap(["black", "cyan", "magenta"])
        im_z_ref = None

        # Use provided zscore_range or calculate from data
        if self.zscore_range is not None:
            vmin, vmax = self.zscore_range
        else:
            all_data = np.concatenate([f.flatten() for f in self.sc_ins.source_frames])
            vmin, vmax = np.percentile(all_data, [1, 99])

        for frame_idx, (orig, cat) in enumerate(
            zip(self.sc_ins.source_frames, self.sc_ins.categorized_frames, strict=True)
        ):
            # Top row: original z-scored frames (clean overview, no contours)
            ax_img = fig.add_subplot(gs[0, frame_idx])
            im_z_ref = ax_img.imshow(orig, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

            # Frame number title (only on top row)
            centered_frame_idx = frame_idx - n_frames // 2
            if centered_frame_idx == 0:
                ax_img.set_title(
                    f"(SPIKE)\nZ-Scored\nFrame {centered_frame_idx}", fontweight="bold", color="red", fontsize=9
                )
            else:
                ax_img.set_title(f"Z-Scored\nFrame {centered_frame_idx}", fontweight="bold", fontsize=9)
            ax_img.axis("off")

            # Second row: categorized frames (median)
            ax_cat = fig.add_subplot(gs[1, frame_idx])
            ax_cat.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)

            # Title without area info
            if centered_frame_idx == 0:
                ax_cat.set_title(f"(SPIKE)\nFrame {centered_frame_idx}", fontweight="bold", color="red", fontsize=8)
            else:
                ax_cat.set_title(f"Frame {centered_frame_idx}", fontweight="bold", fontsize=8)
            ax_cat.axis("off")

        self.lo_main.addWidget(mpl_toolbar)
        self.lo_main.addWidget(canvas)

        # Z-score colorbar (right of top row)
        if im_z_ref is not None:
            ax_cbar_z = fig.add_subplot(gs[0, n_frames])
            fig.colorbar(im_z_ref, cax=ax_cbar_z, label="Z-score")

        # Categorization colorbar (right of middle row)
        ax_cbar_cat = fig.add_subplot(gs[1, n_frames])
        sm = cm.ScalarMappable(cmap=cmap_cat, norm=mpl.colors.BoundaryNorm([0, 1, 2, 3], cmap_cat.N))
        sm.set_array([])
        cbar_cat = fig.colorbar(sm, cax=ax_cbar_cat)
        cbar_cat.set_ticks([0.5, 1.5, 2.5])
        cbar_cat.set_ticklabels(["BK", "Dim", "Bright"], rotation=90, va="center")
        cbar_cat.ax.set_ylabel("Cluster", fontsize=8)

        # Bottom row: Plot voltage trace (spans all columns)
        ax_vm = fig.add_subplot(gs[2, :n_frames])

        # Plot all traces - each trace is (time_centered, voltage)
        n_traces = len(self.spike_traces)
        colors = cm.tab20(np.linspace(0, 1, n_traces))

        for idx, (time_centered, voltage) in enumerate(self.spike_traces):
            ax_vm.plot(time_centered, voltage, linewidth=0.8, color=colors[idx])

        ax_vm.set_xlabel("Time (ms)")
        ax_vm.set_ylabel("Vm (mV)")
        spike_count_str = f" (n={self.n_spikes})" if self.n_spikes is not None else ""
        ax_vm.set_title(f"All Spikes Overlay (centered at Frame 0){spike_count_str}")

        # Calculate time limits based on number of frames (50ms per frame)
        half_frames = n_frames // 2
        frame_duration = 50.0  # ms
        time_min = -half_frames * frame_duration
        time_max = (n_frames - half_frames) * frame_duration
        ax_vm.set_xlim(time_min, time_max)

        # Draw vertical lines at frame boundaries
        for i in range(n_frames + 1):
            t = (i - half_frames) * frame_duration
            if i == half_frames:  # Spike frame start
                ax_vm.axvline(x=t, color="red", linestyle="-", linewidth=2, alpha=0.6, label="Spike Frame")
            else:
                ax_vm.axvline(x=t, color="black", linestyle="--", linewidth=3, alpha=0.8)

        ax_vm.legend()
        ax_vm.minorticks_on()
        ax_vm.grid(True, which="major")
        ax_vm.grid(True, which="minor", alpha=0.3)

        # Main title with threshold info and metadata
        title = f"Spatial Categorization: {self.sc_ins.grouping_method.upper()}"
        if self.exp_date and self.abf_serial and self.img_serial:
            title += f" | {self.exp_date} abf{self.abf_serial}_img{self.img_serial}"
        if self.sc_ins.thresholds_used:
            thresh_dim, thresh_bright = self.sc_ins.thresholds_used
            title += f" | Thresholds: dim>{thresh_dim:.2f}, bright>{thresh_bright:.2f}"
        title += f" | {self.obj}"
        fig.suptitle(title, fontweight="bold", fontsize=11)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, hspace=0.1, wspace=0)


class PlotRegion(QMainWindow):
    """Detailed frame-by-frame viewer with contours, centroids, and legend."""

    def __init__(
        self,
        categorizer: "SpatialCategorizer",
        region_analyzer: "RegionAnalyzer",
        spike_traces: list[tuple[np.ndarray, np.ndarray]],
        title: str = "Region Detail View",
        obj: str = "10X",
        zscore_range: tuple[float, float] | None = None,
        n_spikes: int | None = None,
        *,
        zscore_only: bool = False,
        show: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.sc_ins = categorizer
        self.ra_ins: RegionAnalyzer = region_analyzer
        self.spike_traces = spike_traces
        self.obj = obj
        self.zscore_range = zscore_range  # (vmin, vmax) for consistent color scaling
        self.n_spikes = n_spikes
        self.zscore_only = zscore_only

        # Get frame info
        self.n_frames = len(self.sc_ins.source_frames)
        self.half_frames = self.n_frames // 2
        self.frame_duration = 50.0  # ms per frame

        # Main layout
        self.lo_main = QVBoxLayout()

        # Row 1: ComboBox for frame selection
        self.cb_frame = QComboBox()
        frame_items = []
        for i in range(self.n_frames):
            centered_idx = i - self.half_frames
            if centered_idx == 0:
                frame_items.append(f"Frame {centered_idx} (SPIKE)")
            else:
                frame_items.append(f"Frame {centered_idx}")
        self.cb_frame.addItems(frame_items)
        self.lo_main.addWidget(self.cb_frame, 0, Qt.AlignCenter)

        # Row 2: stacked panels (zscore always; cat + voltage only when not zscore_only)
        self.lo_stacks = QHBoxLayout()
        self.sw_zscore = QStackedWidget()
        self.lo_stacks.addWidget(self.sw_zscore)
        if not zscore_only:
            self.sw_cat = QStackedWidget()
            self.sw_voltage = QStackedWidget()
            self.lo_stacks.addWidget(self.sw_cat)
            self.lo_stacks.addWidget(self.sw_voltage)
        self.lo_main.addLayout(self.lo_stacks)

        # Set up main widget
        w_main = QWidget()
        w_main.setLayout(self.lo_main)
        self.setCentralWidget(w_main)

        # Create plots
        self.plotting()

        # Add toolbar for saving the whole window (above combo box)
        self.toolbar = WindowToolbar(self)
        self.lo_main.insertWidget(0, self.toolbar)  # Before combo box

        # Connect signals
        self.cb_frame.currentIndexChanged.connect(self.switch_frame)

        # Set initial frame to spike frame (center)
        self.cb_frame.setCurrentIndex(self.half_frames)

        self.resize(500 if zscore_only else 1400, 480)
        if show:
            self.show()
            center_on_screen(self)

    def switch_frame(self, idx: int) -> None:
        """Switch all visible stacks to the selected frame."""
        if idx >= 0:
            self.sw_zscore.setCurrentIndex(idx)
            if not self.zscore_only:
                self.sw_cat.setCurrentIndex(idx)
                self.sw_voltage.setCurrentIndex(idx)

    def plotting(self) -> None:
        """Create canvases for each frame and add to stacks."""
        # Use provided zscore_range or calculate from data
        if self.zscore_range is not None:
            vmin, vmax = self.zscore_range
        else:
            all_data = np.concatenate([f.flatten() for f in self.sc_ins.source_frames])
            vmin, vmax = np.percentile(all_data, [1, 99])

        for frame_idx in range(self.n_frames):
            self._create_zscore_canvas(frame_idx, vmin, vmax)
            if not self.zscore_only:
                self._create_categorized_canvas(frame_idx)
                self._create_voltage_canvas(frame_idx)

    def _create_zscore_canvas(self, frame_idx: int, vmin: float, vmax: float) -> None:
        """Stack 1: Z-scored image + contours + centroids + colorbar."""
        centered_idx = frame_idx - self.half_frames
        frame_result = self.ra_ins.get_frame_results(frame_idx)
        orig = self.sc_ins.source_frames[frame_idx]

        fig_z = Figure(figsize=(5, 4), dpi=100)
        canvas_z = FigureCanvasQTAgg(fig_z)
        ax_z = fig_z.add_subplot(111)
        im_z = ax_z.imshow(orig, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        fig_z.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04, label="Z-Score")

        # Draw contour for largest dim region
        dim_largest = frame_result["dim_largest"]
        if dim_largest is not None:
            contour = dim_largest["contour"]
            if contour is not None:
                ax_z.plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=1.5)
            dy, dx = dim_largest["centroid"]
            ax_z.scatter(dx, dy, c="white", s=80, marker="+", linewidths=2, zorder=20)

        # Draw contour for largest bright region
        bright_largest = frame_result["bright_largest"]
        if bright_largest is not None:
            contour = bright_largest["contour"]
            if contour is not None:
                ax_z.plot(contour[:, 1], contour[:, 0], color="magenta", linewidth=1.5)

            # Draw centroid of largest bright region
            y, x = bright_largest["centroid"]
            ax_z.scatter(x, y, c="black", s=80, marker="+", linewidths=2, zorder=20)

            # Draw span lines (cross-hair from centroid)
            x_span_px = bright_largest["x_span_px"]
            y_span_px = bright_largest["y_span_px"]
            x_span_um = bright_largest["x_span_um"]
            y_span_um = bright_largest["y_span_um"]

            # Calculate span extents (half span in each direction from centroid)
            # x_span is horizontal (column direction)
            # y_span is vertical (row direction)
            x_west = x - x_span_px / 2
            x_east = x + x_span_px / 2
            y_north = y - y_span_px / 2
            y_south = y + y_span_px / 2

            # Draw x-span line (horizontal)
            ax_z.plot([x_west, x_east], [y, y], color="red", linewidth=2, linestyle="-", zorder=15, alpha=0.8)
            # Draw y-span line (vertical)
            ax_z.plot([x, x], [y_north, y_south], color="red", linewidth=2, linestyle="-", zorder=15, alpha=0.8)

            bright_centroid_str = f"({x * self.ra_ins.um_per_pixel:.1f} µm, {y * self.ra_ins.um_per_pixel:.1f} µm)"
            span_str = f"x-span: {x_span_um:.1f} µm | y-span: {y_span_um:.1f} µm"
        else:
            bright_centroid_str = "(None)"
            span_str = ""

        if centered_idx == 0:
            title_text = f"Z-Scored Frame {centered_idx} (SPIKE)\nBright centroid: {bright_centroid_str}"
            if span_str:
                title_text += f"\n{span_str}"
            ax_z.set_title(title_text, fontweight="bold", color="red", fontsize=10)
        else:
            title_text = f"Z-Scored Frame {centered_idx}\nBright centroid: {bright_centroid_str}"
            if span_str:
                title_text += f"\n{span_str}"
            ax_z.set_title(title_text, fontweight="bold", fontsize=10)
        ax_z.axis("off")

        # Add legend for contour, centroid, and spans
        ax_z.plot([], [], color="cyan", linewidth=1.5, label="Largest dim contour")
        ax_z.plot([], [], color="magenta", linewidth=1.5, label="Largest bright contour")
        ax_z.plot(
            [], [], marker="+", color="black", linestyle="", markersize=8, markeredgewidth=2, label="Bright centroid"
        )
        ax_z.plot([], [], color="yellow", linewidth=2, label="x-span (horizontal)")
        ax_z.plot([], [], color="lime", linewidth=2, label="y-span (vertical)")
        ax_z.legend(loc="lower left", fontsize=7)

        fig_z.tight_layout()
        _add_scale_bar(self.ra_ins.um_per_pixel, ax_z, orig.shape[1], orig.shape[0])
        self.sw_zscore.addWidget(canvas_z)

    def _create_categorized_canvas(self, frame_idx: int) -> None:
        """Stack 2: Categorized image + legend."""
        centered_idx = frame_idx - self.half_frames
        cat = self.sc_ins.categorized_frames[frame_idx]

        cmap_cat = ListedColormap(["black", "cyan", "magenta"])

        fig_c = Figure(figsize=(4.5, 4), dpi=100)
        canvas_c = FigureCanvasQTAgg(fig_c)
        ax_c = fig_c.add_subplot(111)
        ax_c.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)

        if centered_idx == 0:
            ax_c.set_title(f"Frame {centered_idx} (SPIKE)", fontweight="bold", color="red", fontsize=10)
        else:
            ax_c.set_title(f"Frame {centered_idx}", fontweight="bold", fontsize=10)
        ax_c.axis("off")

        # Add legend
        legend_elements = [
            Patch(facecolor="black", edgecolor="white", label="Background"),
            Patch(facecolor="cyan", label="Dim"),
            Patch(facecolor="magenta", label="Bright"),
        ]
        ax_c.legend(handles=legend_elements, loc="lower left", ncol=1, fontsize=8)
        fig_c.tight_layout()
        _add_scale_bar(self.ra_ins.um_per_pixel, ax_c, cat.shape[1], cat.shape[0])
        self.sw_cat.addWidget(canvas_c)

    def _create_voltage_canvas(self, frame_idx: int) -> None:
        """Stack 3: Voltage trace with xlim for current frame."""
        centered_idx = frame_idx - self.half_frames

        fig_v = Figure(figsize=(4, 4), dpi=100)
        canvas_v = FigureCanvasQTAgg(fig_v)
        ax_v = fig_v.add_subplot(111)

        # Plot all spike traces
        n_traces = len(self.spike_traces)
        colors = cm.tab20(np.linspace(0, 1, n_traces))
        for idx, (time_centered, voltage) in enumerate(self.spike_traces):
            ax_v.plot(time_centered, voltage, linewidth=0.8, color=colors[idx])

        # Calculate xlim for this frame
        xlim_min = centered_idx * self.frame_duration
        xlim_max = (centered_idx + 1) * self.frame_duration
        ax_v.set_xlim(xlim_min, xlim_max)

        # Draw frame boundaries
        ax_v.axvline(x=xlim_min, color="red", linestyle="-", linewidth=2, alpha=0.7)
        ax_v.axvline(x=xlim_max, color="red", linestyle="-", linewidth=2, alpha=0.7)

        ax_v.set_xlabel("Time (ms)")
        ax_v.set_ylabel("Vm (mV)")
        spike_count_str = f" (n={self.n_spikes})" if self.n_spikes is not None else ""
        if centered_idx == 0:
            ax_v.set_title(f"Voltage @ Frame {centered_idx} (SPIKE){spike_count_str}", fontweight="bold", color="red")
        else:
            ax_v.set_title(f"Voltage @ Frame {centered_idx}{spike_count_str}", fontweight="bold")
        ax_v.minorticks_on()
        ax_v.grid(True, which="major")
        ax_v.grid(True, which="minor", alpha=0.3)
        fig_v.tight_layout()
        self.sw_voltage.addWidget(canvas_v)
