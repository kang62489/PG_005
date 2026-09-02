## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
)

# Local application imports
from utils import UISizes


class ViewAlsCorrect:
    def __init__(self, parent=None) -> None:
        self.popwin_container = parent
        self.lo_popwin_container = QHBoxLayout()
        self.popwin_container.setLayout(self.lo_popwin_container)
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()

    def setup_block_1(self) -> None:
        self.lo_block_1 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_1)

        self.btn_load_proc_list = QPushButton("Load Processing List")
        self.lo_block_1.addWidget(self.btn_load_proc_list)

        self.lw_gauss_tiff = QListWidget()
        self.lo_block_1.addWidget(self.lw_gauss_tiff)
        self.lw_gauss_tiff.setFixedWidth(UISizes.LW_GAUSS_TIFFS_WIDTH)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_2)

        self.lo_als_config = QGridLayout()
        self.lo_block_2.addLayout(self.lo_als_config)

        self.lbl_als_lam = QLabel("ALS Lambda: ")
        self.lbl_als_p = QLabel("ALS p: ")
        self.lbl_als_num_iters = QLabel("ALS Num Iters: ")

        self.le_als_lam = QLineEdit("100")
        self.le_als_p = QLineEdit("0.05")
        self.le_als_num_iters = QLineEdit("10")

        self.lo_als_config.addWidget(self.lbl_als_lam, 0, 0)
        self.lo_als_config.addWidget(self.lbl_als_p, 0, 1)
        self.lo_als_config.addWidget(self.lbl_als_num_iters, 0, 2)
        self.lo_als_config.addWidget(self.le_als_lam, 1, 0)
        self.lo_als_config.addWidget(self.le_als_p, 1, 1)
        self.lo_als_config.addWidget(self.le_als_num_iters, 1, 2)


        self.btn_run_als_test = QPushButton("Run ALS Test")
        self.lo_als_config.addWidget(self.btn_run_als_test, 1, 3)

        self.cb_switch_roi = QComboBox()
        self.cb_switch_roi.addItems(["ROI 1", "ROI 2", "ROI 3", "ROI 4", "ROI 5"])
        self.lo_block_2.addWidget(self.cb_switch_roi)

        self.lo_als_plot = QStackedLayout()
        self.lo_block_2.addLayout(self.lo_als_plot)

        from classes import MplCanvas
        self.canvases = [MplCanvas() for _ in range(5)]
        for i, canvas in enumerate(self.canvases):
            canvas.axes.text(0.5, 0.5, f"ROI {i + 1}", ha="center", va="center", fontsize=24, color="gray")
            canvas.axes.set_axis_off()
            self.lo_als_plot.addWidget(canvas)

        self.btn_run_correct = QPushButton("Calibrate dF/f0 slow fluctuation")
        self.btn_run_correct.setFixedHeight(UISizes.BTN_RUN_CORRECT_HEIGHT)
        self.btn_run_correct.setStyleSheet("color: darkgreen; font-weight: bold;")
        self.lo_block_2.addWidget(self.btn_run_correct)

        self.gb_proc_info = QGroupBox("Processing Info")
        self.lo_proc_info = QFormLayout()
        self.gb_proc_info.setLayout(self.lo_proc_info)
        self.lo_block_2.addWidget(self.gb_proc_info)

        self.lbl_run_on = QLabel("Run on:")
        self.le_run_on = QLineEdit()
        self.le_run_on.setReadOnly(True)

        self.lbl_als_params = QLabel("ALS Parameters:")
        self.le_als_params = QLineEdit()
        self.le_als_params.setReadOnly(True)

        self.lbl_current_total = QLabel("Current/Total:")
        self.le_curret_total = QLineEdit()
        self.le_curret_total.setReadOnly(True)

        self.lbl_processing_file = QLabel("Processing file:")
        self.le_processing_file = QLineEdit()
        self.le_processing_file.setReadOnly(True)

        self.lbl_processing_step = QLabel("Processing step:")
        self.le_processing_step = QLineEdit()
        self.le_processing_step.setReadOnly(True)

        self.lo_proc_info.addRow(self.lbl_run_on, self.le_run_on)
        self.lo_proc_info.addRow(self.lbl_current_total, self.le_curret_total)
        self.lo_proc_info.addRow(self.lbl_als_params, self.le_als_params)
        self.lo_proc_info.addRow(self.lbl_processing_file, self.le_processing_file)
        self.lo_proc_info.addRow(self.lbl_processing_step, self.le_processing_step)

