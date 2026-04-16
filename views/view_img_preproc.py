## Modules
# Third-party imports
from PySide6.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QTableView

# Local application imports
from utils.params import UISizes


class ViewImgPreproc:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QHBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()

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

        # Directory of preprocessed TIFFs (Cal and Gauss)
        self.lbl_dir_preprocessed = QLabel("Directory of Preprocessed TIFFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_preprocessed)

        self.lo_dir_preprocessed = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_dir_preprocessed)

        self.le_dir_preprocessed = QTextEdit()
        self.le_dir_preprocessed.setFixedHeight(UISizes.TE_DIRS_HEIGHT)
        
        self.btn_browse_preprocessed = QPushButton("Browse...")
        self.lo_dir_preprocessed.addWidget(self.le_dir_preprocessed)
        self.lo_dir_preprocessed.addWidget(self.btn_browse_preprocessed)

        
        self.lbl_data_selector = QLabel("Picked Data: ")
        self.lo_block_1.addWidget(self.lbl_data_selector)

        self.btn_load_pick_list = QPushButton("Load Pick List")
        self.lo_block_1.addWidget(self.btn_load_pick_list)

        self.tv_data_selector = QTableView()
        self.lo_block_1.addWidget(self.tv_data_selector)

        self.lo_btn_row = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_btn_row)
        self.btn_rm_selected = QPushButton("Remove Selected")
        self.lo_btn_row.addWidget(self.btn_rm_selected)
        self.btn_clear_list = QPushButton("Clear List")
        self.lo_btn_row.addWidget(self.btn_clear_list)

        self.btn_start_check = QPushButton("Start Check")
        self.lo_block_1.addWidget(self.btn_start_check)


