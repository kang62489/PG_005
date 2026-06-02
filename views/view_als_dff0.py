## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
)


class ViewAlsDff0:
    def __init__(self, parent=None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QHBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()

    def setup_block_1(self) -> None:
        self.lo_block_1 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_1)

        self.btn_load_proc_list = QPushButton("Load Processing List")
        self.lo_block_1.addWidget(self.btn_load_proc_list)

        self.lw_gauss_tiff = QListWidget()
        self.lo_block_1.addWidget(self.lw_gauss_tiff)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

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
        self.lo_als_config.addWidget(self.btn_run_als_test, 0, 3)
        self.btn_cal_dff0_all = QPushButton("Calculate dF/F0 for All Gauss TIFFs")
        self.lo_als_config.addWidget(self.btn_cal_dff0_all, 1, 3)

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

