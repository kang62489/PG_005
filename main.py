## Modules
# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from rich.console import Console

# Local application imports
from controllers import CtrlAlignSpike, CtrlAlsCorrect, CtrlDataSelector, CtrlDorQuery, CtrlImgProc, CtrlMain
from utils import APP_STATUS_MESSAGE, UISizes
from views import ViewAlignSpike, ViewAlsCorrect, ViewDataSelector, ViewDorQuery, ViewImgProc, ViewMain

# Setup rich console
console = Console()


class Main(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Analyzer for Cholinergic Influence Domain (ACID)")
        self.setFixedSize(UISizes.MAIN_WINDOW_SIZE[0], UISizes.MAIN_WINDOW_SIZE[1])

        # Load and apply stylesheet
        # with Path.open(STYLES_DIR / "styles.qss") as f:
        #     self.setStyleSheet(f.read())

        # Set status bar message
        self.statusBar().showMessage(APP_STATUS_MESSAGE)

        # Set central widget as the shell view (DOR list + window-opening buttons)
        self.w_main = QWidget()
        self.view_main = ViewMain(self.w_main)

        # Query by DOR popout window (block_2 of the former tab): built eagerly so
        # lw_dor gets populated and stays live even before the window is first shown
        self.popwin_exp_info = QWidget()
        self.popwin_exp_info.setWindowTitle("Query by DOR")
        self.popwin_exp_info.setFixedSize(*UISizes.POPWIN_DOR_DETAIL_SIZE)

        # Log View popout window: metadata + log tools, opened from within the DOR detail window
        self.popwin_log_view = QWidget()
        self.popwin_log_view.setWindowTitle("Log View")
        self.popwin_log_view.resize(*UISizes.POPWIN_LOG_VIEW_SIZE)

        self.view_dor_query = ViewDorQuery(self.popwin_exp_info, self.view_main.lw_dor, self.popwin_log_view)
        self.ctrl_dor_query = CtrlDorQuery(self.view_dor_query)

        # Data Selector popout window
        self.popwin_data_selector = QWidget()
        self.popwin_data_selector.setWindowTitle("Data Selector")
        popwin_data_selector_width, popwin_data_selector_height = UISizes.POPWIN_DATA_SELECTOR_SIZE
        self.popwin_data_selector.resize(popwin_data_selector_width, popwin_data_selector_height)
        self.popwin_data_selector.setFixedWidth(popwin_data_selector_width)

        self.view_data_selector = ViewDataSelector(self.popwin_data_selector)
        self.ctrl_data_selector = CtrlDataSelector(self.view_data_selector)

        # Image Processing popout window
        self.popwin_img_proc = QWidget()
        self.popwin_img_proc.setWindowTitle("Image Processing")
        self.popwin_img_proc.setFixedSize(*UISizes.POPWIN_IMG_PROC_SIZE)

        self.view_img_proc = ViewImgProc(self.popwin_img_proc)
        self.ctrl_img_proc = CtrlImgProc(self.view_img_proc)

        # ALS Correction popout window
        self.popwin_als_correct = QWidget()
        self.popwin_als_correct.setWindowTitle("ALS Correction")
        self.popwin_als_correct.setFixedSize(*UISizes.POPWIN_ALS_CORRECT_SIZE)

        self.view_als_correct = ViewAlsCorrect(self.popwin_als_correct)
        self.ctrl_als_correct = CtrlAlsCorrect(self.view_als_correct)

        # Spike-aligned Analysis popout window
        self.popwin_align_spike = QWidget()
        self.popwin_align_spike.setWindowTitle("Spike-aligned Analysis")
        self.popwin_align_spike.setFixedSize(*UISizes.POPWIN_ALIGN_SPIKE_SIZE)

        self.view_align_spike = ViewAlignSpike(self.popwin_align_spike)
        self.ctrl_align_spike = CtrlAlignSpike(self.view_align_spike)

        # Hand every popout off to CtrlMain, which wires each popwins[key] to view_main's
        # matching btn_<key> — adding a new popout later is just one more dict entry here
        self.popwins = {
            "exp_info": self.popwin_exp_info,
            "log_view": self.popwin_log_view,
            "data_selector": self.popwin_data_selector,
            "img_proc": self.popwin_img_proc,
            "als_correct": self.popwin_als_correct,
            "align_spike": self.popwin_align_spike,
        }
        self.ctrl_main = CtrlMain(self.view_main, self.popwins)

        # One-off wiring that doesn't fit the shell-button naming convention
        self.view_dor_query.btn_log_view.clicked.connect(lambda: self.ctrl_main.show_popwin("log_view"))
        self.ctrl_dor_query.dor_changed.connect(self.ctrl_data_selector.on_dor_changed)
        self.ctrl_data_selector.pick_confirmed.connect(self.ctrl_img_proc.load_pick_list)
        self.ctrl_data_selector.pick_confirmed.connect(lambda _: self.ctrl_main.show_popwin("img_proc"))

        self.setCentralWidget(self.w_main)
        self.center_on_screen()
        self.show()

    def center_on_screen(self) -> None:
        """Center the window on the current screen."""
        screen = self.screen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())


app = QApplication()
app.setStyle("Fusion")
app.styleHints().setColorScheme(Qt.ColorScheme.Light)

window = Main()
app.exec()
