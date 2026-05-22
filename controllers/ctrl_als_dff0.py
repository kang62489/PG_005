## Modules
# Third-party imports
from rich.console import Console

console = Console()

class CtrlAlsDff0:
    def __init__(self, view):
        self.view = view
        self.connect_signals()

    def connect_signals(self):
        self.view.cb_switch_roi.activated.connect(self.on_switch_roi)

    def on_switch_roi(self, index):
        self.view.lo_als_plot.setCurrentIndex(index)