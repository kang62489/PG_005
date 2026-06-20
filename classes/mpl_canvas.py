"""Matplotlib canvas widget for embedding plots in PySide6 GUIs."""

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget

# Set backend to QtAgg for interactive plotting
mpl.use("QtAgg")

# Set save dialog to remember last directory
mpl.rcParams["savefig.directory"] = ""


class MplCanvas(FigureCanvasQTAgg):
    """Class to create a canvas for matplotlib plots."""

    def __init__(self, parent: QWidget | None = None, width: int = 14, height: int = 4, dpi: int = 100) -> None:
        self.parent = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
