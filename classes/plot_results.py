## Modules
# Third-party imports
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

# Set backend to QtAgg for interactive plotting
mpl.use("QtAgg")


class MplCanvas(FigureCanvasQTAgg):
    """Class to create a canvas for matplotlib plots."""

    def __init__(
        self,
        parent: QWidget | None = None,
        width: int = 14,
        height: int = 4,
        dpi: int = 100,
    ) -> None:
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class PlotResults(QMainWindow):
    """Class to plot results from cluster analysis."""

    def __init__(
        self,
        df_list: list,
        title: str = "untitled",
        xlabel: str = "Time (ms)",
        ylabel: str = "Voltage (mV)",
    ) -> None:
        super().__init__()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.df_list = df_list

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
        if len(actions) > 6:  # Configure subplots is usually the 7th action
            toolbar.removeAction(actions[6])
        layout_plotting.addWidget(toolbar)
        layout_plotting.addWidget(canvas_0)

        holding_widget = QWidget()
        holding_widget.setLayout(layout_plotting)
        return holding_widget
