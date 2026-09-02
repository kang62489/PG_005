## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
)

# Local application imports
from utils.params import UISizes


class ViewImgProc:
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

        # Directory of raw TIFFs
        self.lbl_dir_raw_tiffs = QLabel("Directory of Raw TIFFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_raw_tiffs)

        self.lo_dir_raw_images = QHBoxLayout()
        self.te_dir_raw_images = QTextEdit()
        self.te_dir_raw_images.setFixedHeight(UISizes.TE_DIRS_HEIGHT)
        self.btn_browse_raw_images = QPushButton("Browse...")
        self.lo_dir_raw_images.addWidget(self.te_dir_raw_images)
        self.lo_dir_raw_images.addWidget(self.btn_browse_raw_images)
        self.lo_block_1.addLayout(self.lo_dir_raw_images)

        # Directory of processed TIFFs (Cal and Gauss)
        self.lbl_dir_processed = QLabel("Directory of Processed TIFFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_processed)

        self.lo_dir_processed = QHBoxLayout()
        self.te_dir_processed = QTextEdit()
        self.te_dir_processed.setFixedHeight(UISizes.TE_DIRS_HEIGHT)
        self.btn_browse_processed = QPushButton("Browse...")
        self.lo_dir_processed.addWidget(self.te_dir_processed)
        self.lo_dir_processed.addWidget(self.btn_browse_processed)
        self.lo_block_1.addLayout(self.lo_dir_processed)

        self.lo_buttons = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_buttons)

        self.btn_load_pick_list = QPushButton("Load Pick List")
        self.lo_buttons.addWidget(self.btn_load_pick_list)

        self.btn_refresh_status = QPushButton("Refresh Status")
        self.lo_buttons.addWidget(self.btn_refresh_status)

        self.btn_export_proc_list = QPushButton("Export Proc List")
        self.lo_buttons.addWidget(self.btn_export_proc_list)

        self.lbl_pick_list = QLabel("Status of Picked Files: ")
        self.lo_block_1.addWidget(self.lbl_pick_list)

        self.tv_pick_list = QTableView()
        self.lo_block_1.addWidget(self.tv_pick_list)


    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_2)

        self.btn_start_processing = QPushButton("Start Processing")
        self.lo_block_2.addWidget(self.btn_start_processing)

        self.gb_proc_info = QGroupBox("Processing Info")
        self.lo_proc_info = QFormLayout()
        self.gb_proc_info.setLayout(self.lo_proc_info)
        self.lo_block_2.addWidget(self.gb_proc_info)

        self.lbl_run_on = QLabel("Run on:")
        self.le_run_on = QLineEdit()
        self.le_run_on.setReadOnly(True)

        self.lbl_current_total = QLabel("Current/Total:")
        self.le_curret_total = QLineEdit()
        self.le_curret_total.setReadOnly(True)

        self.lbl_mode = QLabel("Mode:")
        self.le_mode = QLineEdit()
        self.le_mode.setReadOnly(True)

        self.lbl_processing_file = QLabel("Processing file:")
        self.le_processing_file = QLineEdit()
        self.le_processing_file.setReadOnly(True)

        self.lbl_processing_step = QLabel("Processing step:")
        self.le_processing_step = QLineEdit()
        self.le_processing_step.setReadOnly(True)


        self.lo_proc_info.addRow(self.lbl_run_on, self.le_run_on)
        self.lo_proc_info.addRow(self.lbl_current_total, self.le_curret_total)
        self.lo_proc_info.addRow(self.lbl_mode, self.le_mode)
        self.lo_proc_info.addRow(self.lbl_processing_file, self.le_processing_file)
        self.lo_proc_info.addRow(self.lbl_processing_step, self.le_processing_step)








