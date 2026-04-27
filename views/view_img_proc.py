## Modules
# Third-party imports
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableView, QTextEdit, QVBoxLayout, QWidget

# Local application imports
from utils.params import UISizes


class ViewImgProc:
    def __init__(self, parent: QWidget | None = None) -> None:
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

        # Directory of raw TIFFs
        self.lbl_dir_raw_tiffs = QLabel("Directory of Raw TIFFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_raw_tiffs)

        self.lo_dir_raw_tiffs = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_dir_raw_tiffs)
        self.btn_browse_raw_tiffs = QPushButton("Browse...")

        self.le_dir_raw_images = QTextEdit()
        self.le_dir_raw_images.setFixedHeight(UISizes.TE_DIRS_HEIGHT)

        self.lo_dir_raw_tiffs.addWidget(self.le_dir_raw_images)
        self.lo_dir_raw_tiffs.addWidget(self.btn_browse_raw_tiffs)

        # Directory of raw ABFs
        self.lbl_dir_raw_abfs = QLabel("Directory of Raw ABFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_raw_abfs)

        self.lo_dir_raw_abfs = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_dir_raw_abfs)

        self.le_dir_raw_abfs = QTextEdit()
        self.le_dir_raw_abfs.setFixedHeight(UISizes.TE_DIRS_HEIGHT)

        self.btn_browse_raw_abfs = QPushButton("Browse...")
        self.lo_dir_raw_abfs.addWidget(self.le_dir_raw_abfs)
        self.lo_dir_raw_abfs.addWidget(self.btn_browse_raw_abfs)

        # Directory of processed TIFFs (Cal and Gauss)
        self.lbl_dir_processed = QLabel("Directory of Processed TIFFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_processed)

        self.lo_dir_processed = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_dir_processed)

        self.le_dir_processed = QTextEdit()
        self.le_dir_processed.setFixedHeight(UISizes.TE_DIRS_HEIGHT)

        self.btn_browse_processed = QPushButton("Browse...")
        self.lo_dir_processed.addWidget(self.le_dir_processed)
        self.lo_dir_processed.addWidget(self.btn_browse_processed)


        self.lbl_data_selector = QLabel("Picked Data: ")
        self.lo_block_1.addWidget(self.lbl_data_selector)

        self.btn_load_pick_list = QPushButton("Load Pick List")
        self.lo_block_1.addWidget(self.btn_load_pick_list)

        self.btn_check_file_status = QPushButton("Check File Status")
        self.lo_block_1.addWidget(self.btn_check_file_status)

        self.btn_check_bleach_corr = QPushButton("Check Bleach Correction")
        self.lo_block_1.addWidget(self.btn_check_bleach_corr)

        self.tv_data_selector = QTableView()
        self.lo_block_1.addWidget(self.tv_data_selector)

        self.btn_start_processing = QPushButton("Start Processing")
        self.lo_block_1.addWidget(self.btn_start_processing)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

        self.lbl_preview_corr = QLabel("Preview of Bleaching Correction: ")
        self.lo_block_2.addWidget(self.lbl_preview_corr)

        self.w_preview_corr = QWidget()
        self.lo_block_2.addWidget(self.w_preview_corr)
        self.lo_preview_corr = QVBoxLayout(self.w_preview_corr)

