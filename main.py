## Modules
# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from rich.console import Console

from controllers import CtrlAlsCal, CtrlDataSelector, CtrlDorQuery, CtrlImgProc

# Local application imports
from utils import APP_STATUS_MESSAGE, UISizes
from views import ViewAlsCal, ViewDataSelector, ViewDorQuery, ViewImgProc

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
        self.tab_data_selector = QWidget()
        self.tab_im_proc = QWidget()
        self.tab_als_cal = QWidget()
        self.tab_align_spike = QWidget()


        self.w_main.addTab(self.tab_dor_query, "Query by DOR")
        self.w_main.addTab(self.tab_data_selector, "Data Selector")
        self.w_main.addTab(self.tab_im_proc, "Image Processing")
        self.w_main.addTab(self.tab_als_cal, "ALS Correction")
        self.w_main.addTab(self.tab_align_spike, "Spike-aligned Analysis")

        # Setup tabs
        self.view_dor_query = ViewDorQuery(self.tab_dor_query)
        self.ctrl_dor_query = CtrlDorQuery(self.view_dor_query)

        self.view_data_selector = ViewDataSelector(self.tab_data_selector)
        self.ctrl_data_selector = CtrlDataSelector(self.view_data_selector)

        self.view_img_proc = ViewImgProc(self.tab_im_proc)
        self.ctrl_img_proc = CtrlImgProc(self.view_img_proc)

        self.view_als_cal = ViewAlsCal(self.tab_als_cal)
        self.ctrl_als_cal = CtrlAlsCal(self.view_als_cal)

        # Connect dor_changed signal from ctrl_dor_query to ctrl_data_selector
        self.ctrl_dor_query.dor_changed.connect(self.ctrl_data_selector.on_dor_changed)

        # Connect pick_confirmed signal from ctrl_data_selector to ctrl_img_proc
        self.ctrl_data_selector.pick_confirmed.connect(self.ctrl_img_proc.load_pick_list)
        self.ctrl_data_selector.pick_confirmed.connect(lambda _: self.w_main.setCurrentIndex(2))

        self.setCentralWidget(self.w_main)
        self.show()


app = QApplication()
app.setStyle("Fusion")
app.styleHints().setColorScheme(Qt.ColorScheme.Light)

window = Main()
app.exec()
