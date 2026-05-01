## Modules
# Third-party imports
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableView, QTextEdit, QVBoxLayout

# Local application imports
from utils.params import UISizes


class ViewImgProc:
    def __init__(self, parent=None) -> None:
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

        self.le_dir_raw_images = QTextEdit()
        self.le_dir_raw_images.setFixedHeight(UISizes.TE_DIRS_HEIGHT)
        self.lo_block_1.addWidget(self.le_dir_raw_images)

        # Directory of processed TIFFs (Cal and Gauss)
        self.lbl_dir_processed = QLabel("Directory of Processed TIFFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_processed)

        self.le_dir_processed = QTextEdit()
        self.le_dir_processed.setFixedHeight(UISizes.TE_DIRS_HEIGHT)
        self.lo_block_1.addWidget(self.le_dir_processed)


        self.lbl_pick_list = QLabel("Picked Data: ")
        self.lo_block_1.addWidget(self.lbl_pick_list)

        self.btn_load_pick_list = QPushButton("Load Pick List")
        self.lo_block_1.addWidget(self.btn_load_pick_list)

        self.tv_pick_list = QTableView()
        self.lo_block_1.addWidget(self.tv_pick_list)

        self.btn_start_processing = QPushButton("Start Processing")
        self.lo_block_1.addWidget(self.btn_start_processing)

