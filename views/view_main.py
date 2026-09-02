## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Local application imports
from utils import UISizes


class ViewMain:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.container = parent
        self.lo_main = QHBoxLayout()
        self.container.setLayout(self.lo_main)
        self.setup_layouts()
        self.setup_uis()

    def setup_layouts(self) -> None:
        self.lo_dor = QVBoxLayout()
        self.lo_buttons = QVBoxLayout()

        self.lo_main.addLayout(self.lo_dor)
        self.lo_main.addLayout(self.lo_buttons)

    def setup_uis(self) -> None:
        # Setup DOR list block (Left block)
        self.lbl_dor = QLabel("Date of Record: ")
        self.lo_dor.addWidget(self.lbl_dor)

        self.lw_dor = QListWidget()
        self.lo_dor.addWidget(self.lw_dor)
        self.lw_dor.setFixedWidth(UISizes.LW_DOR_SHELL_WIDTH)


        # Add a spacing of lbl_dor height + spacing between lbl_dor and lw_dor before the first button in the right block
        self.lo_buttons.addSpacing(self.lbl_dor.sizeHint().height() + self.lo_dor.spacing())

        self.btn_exp_info = QPushButton("Query by\nDOR")
        self.btn_data_selector = QPushButton("Data\nSelector")
        self.btn_img_proc = QPushButton("Image\nProcessing")
        self.btn_als_correct = QPushButton("ALS\nCorrection")
        self.btn_align_spike = QPushButton("Spike-aligned\nAnalysis")

        btn_w, btn_h = UISizes.BTN_SHELL_SIZE
        for btn in (
            self.btn_exp_info,
            self.btn_data_selector,
            self.btn_img_proc,
            self.btn_als_correct,
            self.btn_align_spike,
        ):
            btn.setFixedSize(btn_w, btn_h)
            self.lo_buttons.addWidget(btn)

        self.lo_buttons.addStretch()
