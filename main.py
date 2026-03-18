## Modules
# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from rich.console import Console

from controllers import CtrlDorQuery

# Local application imports
from utils import APP_STATUS_MESSAGE, UISizes
from views import ViewCheckList, ViewDorQuery

# Setup rich console
console = Console()


class Main(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Analyzer for Cholinergic Influence Domain (ACID)")
        self.setFixedSize(UISizes.MAIN_WINDOW_SIZE[0], UISizes.MAIN_WINDOW_SIZE[1])

        # Set status bar message
        self.statusBar().showMessage(APP_STATUS_MESSAGE)

        # Set central widget as a tab widget
        self.w_main = QTabWidget()

        # Add tabs to the tab widget
        self.tab_dor_query = QWidget()
        self.tab_pick_raws = QWidget()
        self.tab_check_list = QWidget()
        self.tab_im_preproc = QWidget()
        self.tab_spike_align = QWidget()

        self.w_main.addTab(self.tab_dor_query, "Query by DOR")
        self.w_main.addTab(self.tab_check_list, "Check Pick List")
        self.w_main.addTab(self.tab_im_preproc, "Image Preprocessing")
        self.w_main.addTab(self.tab_spike_align, "Spike Alignment Analysis")

        # Setup tabs
        self.view_dor_query = ViewDorQuery(self.tab_dor_query)
        self.ctrl_dor_query = CtrlDorQuery(self.view_dor_query)

        self.view_check_list = ViewCheckList(self.tab_check_list)

        self.setCentralWidget(self.w_main)
        self.show()

app = QApplication()
app.setStyle("Fusion")
app.styleHints().setColorScheme(Qt.ColorScheme.Light)

window = Main()
app.exec()
