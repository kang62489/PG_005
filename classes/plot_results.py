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
from matplotlib.patches import Patch
from PySide6.QtCore import QTimer
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QMainWindow, QPushButton, QStackedWidget, QVBoxLayout, QWidget
from rich.console import Console

# Local application imports

# Set backend to QtAgg for interactive plotting
mpl.use("QtAgg")

# Set rich console
cs = Console()


# customized toolbar
class CustomToolbar(NavigationToolbar):
    toolitems: ClassVar[list[tuple[str, str, str, str]]] = [("Save", "Save the figure", "filesave", "save_figure")]


class MplCanvas(FigureCanvasQTAgg):
    """Class to create a canvas for matplotlib plots."""

    def __init__(self, parent: QWidget | None = None, width: int = 14, height: int = 4, dpi: int = 100) -> None:
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


def center_on_screen(window: QMainWindow) -> None:
    """Center the window on the current screen."""
    screen = window.screen()
    screen_geometry = screen.availableGeometry()
    window_geometry = window.frameGeometry()
    center_point = screen_geometry.center()
    window_geometry.moveCenter(center_point)
    window.move(window_geometry.topLeft())


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


class PlotSpatialDist(QMainWindow):
    def __init__(
        self,
        categorizor: "SpatialCategorizer",
        spike_traces: list[tuple[np.ndarray, np.ndarray]],
        title: str = "Spatial Distribution",
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.sc_ins = categorizor
        self.spike_traces = spike_traces

        self.lo_main = QVBoxLayout()

        self.w_main = QWidget()
        self.w_main.setLayout(self.lo_main)
        self.setCentralWidget(self.w_main)

        self.plotting()

        self.show()
        center_on_screen(self)

    def plotting(self) -> None:  # noqa: PLR0915
        if not self.sc_ins.categorized_frames:
            msg = "No results to plot. Call fit() first."
            raise RuntimeError(msg)

        # Get number of frames in this segment
        n_frames = len(self.sc_ins.frames)

        # Create figure with custom canvas
        fig = Figure(figsize=(8 * n_frames, 8), dpi=100)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(1400, 800)
        mpl_toolbar = CustomToolbar(canvas, self)

        gs = GridSpec(4, n_frames, figure=fig, height_ratios=[1, 1, 0.1, 0.8], hspace=0.4, wspace=0)

        cmap_cat = ListedColormap(["black", "limegreen", "yellow"])
        all_data = np.concatenate([f.flatten() for f in self.sc_ins.frames])
        vmin, vmax = np.percentile(all_data, [1, 99])

        for frame_idx, (orig, cat) in enumerate(zip(self.sc_ins.frames, self.sc_ins.categorized_frames, strict=True)):
            # Top row: original z-scored frames (median)
            ax_img = fig.add_subplot(gs[0, frame_idx])
            ax_img.imshow(orig, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

            # Frame number title (only on top row)
            centered_frame_idx = frame_idx - n_frames // 2
            if centered_frame_idx == 0:
                ax_img.set_title(
                    f"(SPIKE)\nZ-Scored\nFrame {centered_frame_idx}", fontweight="bold", color="red", fontsize=9
                )
            else:
                ax_img.set_title(f"Z-Scored\nFrame {centered_frame_idx}", fontweight="bold", fontsize=9)
            ax_img.axis("off")

            # Bottom row: categorized frames (median)
            ax_cat = fig.add_subplot(gs[1, frame_idx])
            ax_cat.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2)
            # Frame number title (only on top row)
            if centered_frame_idx == 0:
                ax_cat.set_title(
                    f"(SPIKE)\nSpatial Dist\nFrame {centered_frame_idx}", fontweight="bold", color="red", fontsize=9
                )
            else:
                ax_cat.set_title(f"Spatial Dist\nFrame {centered_frame_idx}", fontweight="bold", fontsize=9)
            ax_cat.axis("off")

        self.lo_main.addWidget(mpl_toolbar)
        self.lo_main.addWidget(canvas)

        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis("off")

        legend_elements = [
            Patch(facecolor="black", edgecolor="white", label="Background"),
            Patch(facecolor="limegreen", label="Dim"),
            Patch(facecolor="yellow", label="Bright"),
        ]
        ax_legend.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9)

        # Bottom row: Plot voltage trace (spans all columns)
        ax_vm = fig.add_subplot(gs[3, :])

        # Plot all traces - each trace is (time_centered, voltage)
        n_traces = len(self.spike_traces)
        colors = cm.tab20(np.linspace(0, 1, n_traces))

        for idx, (time_centered, voltage) in enumerate(self.spike_traces):
            ax_vm.plot(time_centered, voltage, alpha=0.8, linewidth=0.8, color=colors[idx])

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
