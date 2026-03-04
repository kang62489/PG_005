## Modules
# Third-party imports
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableView, QTextEdit, QVBoxLayout, QWidget


class ViewCheckList:
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

        self.lbl_pick_list = QLabel("Pick List: ")
        self.lo_block_1.addWidget(self.lbl_pick_list)

        self.tv_pick_list = QTableView()
        self.lo_block_1.addWidget(self.tv_pick_list)

        self.lo_btn_row = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_btn_row)
        self.btn_rm_selected = QPushButton("Remove Selected")
        self.lo_btn_row.addWidget(self.btn_rm_selected)
        self.btn_clear_list = QPushButton("Clear List")
        self.lo_btn_row.addWidget(self.btn_clear_list)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

        # Directory of raw TIFFs
        self.lbl_dir_raw_tiffs = QLabel("Directory of Raw TIFFs: ")
        self.lo_block_2.addWidget(self.lbl_dir_raw_tiffs)

        self.lo_dir_raw_tiffs = QHBoxLayout()
        self.lo_block_2.addLayout(self.lo_dir_raw_tiffs)
        self.btn_browse_raw_tiffs = QPushButton("Browse...")
        self.le_dir_raw_images = QTextEdit()
        self.lo_dir_raw_tiffs.addWidget(self.le_dir_raw_images)
        self.lo_dir_raw_tiffs.addWidget(self.btn_browse_raw_tiffs)

        # Directory of raw ABFs
        self.lbl_dir_raw_abfs = QLabel("Directory of Raw ABFs: ")
        self.lo_block_2.addWidget(self.lbl_dir_raw_abfs)

        self.lo_dir_raw_abfs = QHBoxLayout()
        self.lo_block_2.addLayout(self.lo_dir_raw_abfs)

        self.le_dir_raw_abfs = QTextEdit()
        self.btn_browse_raw_abfs = QPushButton("Browse...")
        self.lo_dir_raw_abfs.addWidget(self.le_dir_raw_abfs)
        self.lo_dir_raw_abfs.addWidget(self.btn_browse_raw_abfs)

        # Directory of preprocessed TIFFs (Cal and Gauss)
        self.lbl_dir_preprocessed = QLabel("Directory of Preprocessed TIFFs: ")
        self.lo_block_2.addWidget(self.lbl_dir_preprocessed)

        self.lo_dir_preprocessed = QHBoxLayout()
        self.lo_block_2.addLayout(self.lo_dir_preprocessed)
        self.le_dir_preprocessed = QTextEdit()
        self.btn_browse_preprocessed = QPushButton("Browse...")
        self.lo_dir_preprocessed.addWidget(self.le_dir_preprocessed)
        self.lo_dir_preprocessed.addWidget(self.btn_browse_preprocessed)

        self.btn_start_check = QPushButton("Start Check")
        self.lo_block_2.addWidget(self.btn_start_check)
