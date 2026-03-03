## Modules
# Standard library imports
import os

# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from rich.console import Console

# Local application imports
from utils.params import APP_STATUS_MESSAGE, UISizes
from views import ViewDorQuery, ViewPickRaws

# Setup rich console
console = Console()


class Main(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Cholinergic Domain Resolver of Morphology (CD-ROM)")
        self.setFixedSize(UISizes.MAIN_WINDOW_SIZE[0], UISizes.MAIN_WINDOW_SIZE[1])

        # Set status bar message
        self.statusBar().showMessage(APP_STATUS_MESSAGE)

        # Set central widget as a tab widget
        self.w_main = QTabWidget()

        # Add tabs to the tab widget
        self.tab_dor_query = QWidget()
        self.tab_pick_raws = QWidget()
        self.tab_check_pick_list = QWidget()
        self.tab_im_preproc = QWidget()
        self.tab_spike_align = QWidget()
        
        self.w_main.addTab(self.tab_dor_query, "Query by DOR")
        self.w_main.addTab(self.tab_pick_raws, "Pick Raws from REC Summary")
        self.w_main.addTab(self.tab_check_pick_list, "Check Pick List")
        self.w_main.addTab(self.tab_im_preproc, "Image Preprocessing")
        self.w_main.addTab(self.tab_spike_align, "Spike Alignment Analysis")
        
        # Setup tabs
        self.view_dor_query = ViewDorQuery(self.tab_dor_query)
        
        self.view_pick_raws = ViewPickRaws(self.tab_pick_raws)
        
        self.setCentralWidget(self.w_main)
        self.show()


app = QApplication()
app.setStyle("Fusion")
app.styleHints().setColorScheme(Qt.ColorScheme.Light)

window = Main()
app.exec()
