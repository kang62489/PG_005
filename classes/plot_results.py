## Modules
# Standard library imports
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from classes.spatial_categorization import SpatialCategorizer

# Third-party imports
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
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
def center_on_screen(window: QMainWindow) -> None:
    """Center the window on the current screen."""
    screen = window.screen()
    screen_geometry = screen.availableGeometry()
    window_geometry = window.frameGeometry()
    center_point = screen_geometry.center()
    window_geometry.moveCenter(center_point)
    window.move(window_geometry.topLeft())


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
        self.df_list[0].plot(
            ax=canvas_0.axes,
            x="Time",
            y="Vm",
            kind="line",
            color="blue",
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            label="Vm",
        )
        self.df_list[1].plot(
            ax=canvas_0.axes,
            x="Time",
            y="Peaks",
            kind="scatter",
            marker="o",
            c="none",
            edgecolors="red",
            label="Spikes",
        )
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
        df_picked_spikes: pd.DataFrame,
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
            f"Spike {idx} @ Frame {row['Spike_Frame_Index']}" for idx, row in self.df_picked_spikes.iterrows()
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
            spike_row = self.df_picked_spikes.iloc[idx]
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
        *,
        show: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.sc_ins = categorizor
        self.spike_traces = spike_traces
        self.obj = obj
        self.zscore_range = zscore_range  # (vmin, vmax) for consistent color scaling

        # Fit RegionAnalyzer for contours, centroids, and area
        self.ra_ins = RegionAnalyzer(obj=obj)
        self.ra_ins.fit(self.sc_ins.categorized_frames)

        self.lo_main = QVBoxLayout()

        self.w_main = QWidget()
        self.w_main.setLayout(self.lo_main)
        self.setCentralWidget(self.w_main)

        self.plotting()

        if show:
            self.show()
            center_on_screen(self)

    def plotting(self) -> None:  # noqa: PLR0915
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

        gs = GridSpec(3, n_frames, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0)

        cmap_cat = ListedColormap(["black", "cyan", "magenta"])

        # Use provided zscore_range or calculate from data
        if self.zscore_range is not None:
            vmin, vmax = self.zscore_range
        else:
            all_data = np.concatenate([f.flatten() for f in self.sc_ins.source_frames])
            vmin, vmax = np.percentile(all_data, [1, 99])

        for frame_idx, (orig, cat) in enumerate(
            zip(self.sc_ins.source_frames, self.sc_ins.categorized_frames, strict=True)
        ):
            # Get region analysis for this frame
            frame_result = self.ra_ins.get_frame_results(frame_idx)

            # Top row: original z-scored frames (clean overview, no contours)
            ax_img = fig.add_subplot(gs[0, frame_idx])
            ax_img.imshow(orig, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

            # Get category data for area calculation
            dim_cat = frame_result["dim_category"]
            bright_cat = frame_result["bright_category"]

            # Frame number title (only on top row)
            centered_frame_idx = frame_idx - n_frames // 2
            if centered_frame_idx == 0:
                ax_img.set_title(
                    f"(SPIKE)\nZ-Scored\nFrame {centered_frame_idx}", fontweight="bold", color="red", fontsize=9
                )
            else:
                ax_img.set_title(f"Z-Scored\nFrame {centered_frame_idx}", fontweight="bold", fontsize=9)
            ax_img.axis("off")

            # Get total areas from category-level analysis
            dim_total_area = dim_cat["total_area_um2"]
            bright_total_area = bright_cat["total_area_um2"]

            # Second row: categorized frames (median) with area info in title
            ax_cat = fig.add_subplot(gs[1, frame_idx])
            ax_cat.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)

            # Title with area info
            if centered_frame_idx == 0:
                ax_cat.set_title(
                    f"(SPIKE) Frame {centered_frame_idx}\nDim: {dim_total_area:.1f} µm²\nBright: {bright_total_area:.1f} µm²",
                    fontweight="bold",
                    color="red",
                    fontsize=8,
                )
            else:
                ax_cat.set_title(
                    f"Frame {centered_frame_idx}\nDim: {dim_total_area:.1f} µm²\nBright: {bright_total_area:.1f} µm²",
                    fontweight="bold",
                    fontsize=8,
                )
            ax_cat.axis("off")

        self.lo_main.addWidget(mpl_toolbar)
        self.lo_main.addWidget(canvas)

        legend_elements = [
            Patch(facecolor="black", edgecolor="white", label="Background"),
            Patch(facecolor="cyan", label="Dim"),
            Patch(facecolor="magenta", label="Bright"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.38),
            ncol=3,
            fontsize=9,
            frameon=True,
            facecolor="white",
            framealpha=0.7,
            edgecolor="gray",
        )

        # Bottom row: Plot voltage trace (spans all columns)
        ax_vm = fig.add_subplot(gs[2, :])

        # Plot all traces - each trace is (time_centered, voltage)
        n_traces = len(self.spike_traces)
        colors = cm.tab20(np.linspace(0, 1, n_traces))

        for idx, (time_centered, voltage) in enumerate(self.spike_traces):
            ax_vm.plot(time_centered, voltage, linewidth=0.8, color=colors[idx])

        ax_vm.set_xlabel("Time (ms)")
        ax_vm.set_ylabel("Vm (mV)")
        ax_vm.set_title("All Spikes Overlay (centered at Frame 0)")

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

        # Main title with threshold info
        title = f"Spatial Categorization: {self.sc_ins.method.upper()}"
        if self.sc_ins.thresholds_used:
            thresh_dim, thresh_bright = self.sc_ins.thresholds_used
            title += f" | Thresholds: dim>{thresh_dim:.2f}, bright>{thresh_bright:.2f}"
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
        *,
        show: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.sc_ins = categorizer
        self.ra_ins: RegionAnalyzer = region_analyzer
        self.spike_traces = spike_traces
        self.obj = obj
        self.zscore_range = zscore_range  # (vmin, vmax) for consistent color scaling

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

        # Row 2: Three QStackedWidgets side by side
        self.lo_stacks = QHBoxLayout()
        self.sw_zscore = QStackedWidget()
        self.sw_cat = QStackedWidget()
        self.sw_voltage = QStackedWidget()

        self.lo_stacks.addWidget(self.sw_zscore)
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

        # Set window size to accommodate all three stacks
        self.resize(1400, 480)
        if show:
            self.show()
            center_on_screen(self)

    def switch_frame(self, idx: int) -> None:
        """Switch all three stacks to the selected frame."""
        if idx >= 0:
            self.sw_zscore.setCurrentIndex(idx)
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

        # Draw contours
        for contour in frame_result["dim_contours"]:
            ax_z.plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=1.5)
        for contour in frame_result["bright_contours"]:
            ax_z.plot(contour[:, 1], contour[:, 0], color="magenta", linewidth=1.5)

        # Draw centroids of largest regions
        dim_largest = frame_result["dim_largest"]
        bright_largest = frame_result["bright_largest"]
        if dim_largest is not None:
            y, x = dim_largest["centroid"]
            ax_z.scatter(x, y, c="black", s=60, marker="x", linewidths=2, zorder=20)
            dim_centroid_str = f"({x * self.ra_ins.um_per_pixel:.1f} µm, {y * self.ra_ins.um_per_pixel:.1f} µm)"
        else:
            dim_centroid_str = "(None)"

        if bright_largest is not None:
            y, x = bright_largest["centroid"]
            ax_z.scatter(x, y, c="black", s=80, marker="+", linewidths=2, zorder=20)
            bright_centroid_str = f"({x * self.ra_ins.um_per_pixel:.1f} µm, {y * self.ra_ins.um_per_pixel:.1f} µm)"
        else:
            bright_centroid_str = "(None)"

        if centered_idx == 0:
            ax_z.set_title(
                f"Z-Scored Frame {centered_idx} (SPIKE)\nBright centroid: {bright_centroid_str}\nDim centroid: {dim_centroid_str}",
                fontweight="bold",
                color="red",
                fontsize=10,
            )
        else:
            ax_z.set_title(
                f"Z-Scored Frame {centered_idx}\nBright centroid: {bright_centroid_str}\nDim centroid: {dim_centroid_str})",
                fontweight="bold",
                fontsize=10,
            )
        ax_z.axis("off")

        # Add legend for contours and centroids
        ax_z.plot([], [], color="cyan", linewidth=1.5, label="Dim contour")
        ax_z.plot([], [], color="magenta", linewidth=1.5, label="Bright contour")
        ax_z.plot(
            [], [], marker="x", color="black", linestyle="", markersize=6, markeredgewidth=2, label="Dim centroid"
        )
        ax_z.plot(
            [], [], marker="+", color="black", linestyle="", markersize=8, markeredgewidth=2, label="Bright centroid"
        )
        ax_z.legend(loc="lower left", fontsize=7)

        fig_z.tight_layout()
        _add_scale_bar(self.ra_ins.um_per_pixel, ax_z, orig.shape[1], orig.shape[0])
        self.sw_zscore.addWidget(canvas_z)

    def _create_categorized_canvas(self, frame_idx: int) -> None:
        """Stack 2: Categorized image + legend + area info."""
        centered_idx = frame_idx - self.half_frames
        frame_result = self.ra_ins.get_frame_results(frame_idx)
        cat = self.sc_ins.categorized_frames[frame_idx]

        cmap_cat = ListedColormap(["black", "cyan", "magenta"])

        fig_c = Figure(figsize=(4.5, 4), dpi=100)
        canvas_c = FigureCanvasQTAgg(fig_c)
        ax_c = fig_c.add_subplot(111)
        ax_c.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)

        dim_cat = frame_result["dim_category"]
        bright_cat = frame_result["bright_category"]
        dim_area = dim_cat["total_area_um2"]
        bright_area = bright_cat["total_area_um2"]

        if centered_idx == 0:
            ax_c.set_title(
                f"Frame {centered_idx} (SPIKE)\nDim: {dim_area:.1f} µm² | Bright: {bright_area:.1f} µm²",
                fontweight="bold",
                color="red",
                fontsize=10,
            )
        else:
            ax_c.set_title(
                f"Frame {centered_idx}\nDim: {dim_area:.1f} µm² | Bright: {bright_area:.1f} µm²",
                fontweight="bold",
                fontsize=10,
            )
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
        if centered_idx == 0:
            ax_v.set_title(f"Voltage @ Frame {centered_idx} (SPIKE)", fontweight="bold", color="red")
        else:
            ax_v.set_title(f"Voltage @ Frame {centered_idx}", fontweight="bold")
        ax_v.minorticks_on()
        ax_v.grid(True, which="major")
        ax_v.grid(True, which="minor", alpha=0.3)
        fig_v.tight_layout()
        self.sw_voltage.addWidget(canvas_v)
