## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Local application imports
from classes import CheckableDropdown

FILTER_COLUMNS = ["OBJ", "EXC", "EMI"]


class ViewCheckList:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QHBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)
        self.filter_columns = FILTER_COLUMNS
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()
        self.setup_block_3()

    def setup_block_1(self) -> None:
        self.lo_block_1 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_1)

        # Filter Panel
        self.lo_panels = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_panels)

        self.gb_filter_panel = QGroupBox("Filter Panel")
        self.lo_panels.addWidget(self.gb_filter_panel)
        self.gb_filter_panel.setLayout(QHBoxLayout())

        for col in self.filter_columns:
            dropdown = CheckableDropdown(col)
            dropdown.setObjectName(f"dd_{col}")
            setattr(self, f"dd_{col}", dropdown)
            self.gb_filter_panel.layout().addWidget(dropdown)

        self.dd_shown_cols = CheckableDropdown("Show Columns")
        self.gb_filter_panel.layout().addWidget(self.dd_shown_cols)

        self.btn_reset_all_filters = QPushButton("Reset All Filters")
        self.gb_filter_panel.layout().addWidget(self.btn_reset_all_filters)
        self.lo_db_view = QVBoxLayout()
        self.lo_block_1.addLayout(self.lo_db_view)

        # Pick List Control
        self.gb_pick_list = QGroupBox("Pick List Control")
        self.gb_pick_list.setLayout(QHBoxLayout())
        self.btn_pick_selected = QPushButton("Pick Selected")
        self.btn_open_pick_list = QPushButton("Open Pick List")
        self.gb_pick_list.layout().addWidget(self.btn_pick_selected)
        self.gb_pick_list.layout().addWidget(self.btn_open_pick_list)
        self.lo_panels.addWidget(self.gb_pick_list)

        # REC Summary
        self.lbl_rec_summary = QLabel("REC Summary: ")
        self.lo_db_view.addWidget(self.lbl_rec_summary)

        self.tv_rec_summary = QTableView()
        self.lo_db_view.addWidget(self.tv_rec_summary)
        self.tv_rec_summary.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_rec_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

        self.lbl_check_list = QLabel("Check List: ")
        self.lo_block_2.addWidget(self.lbl_check_list)

        self.btn_load_pick_list = QPushButton("Load Pick List")
        self.lo_block_2.addWidget(self.btn_load_pick_list)

        self.tv_check_list = QTableView()
        self.lo_block_2.addWidget(self.tv_check_list)

        self.lo_btn_row = QHBoxLayout()
        self.lo_block_2.addLayout(self.lo_btn_row)
        self.btn_rm_selected = QPushButton("Remove Selected")
        self.lo_btn_row.addWidget(self.btn_rm_selected)
        self.btn_clear_list = QPushButton("Clear List")
        self.lo_btn_row.addWidget(self.btn_clear_list)

    def setup_block_3(self) -> None:
        self.lo_block_3 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_3)

        # Directory of raw TIFFs
        self.lbl_dir_raw_tiffs = QLabel("Directory of Raw TIFFs: ")
        self.lo_block_3.addWidget(self.lbl_dir_raw_tiffs)

        self.lo_dir_raw_tiffs = QHBoxLayout()
        self.lo_block_3.addLayout(self.lo_dir_raw_tiffs)
        self.btn_browse_raw_tiffs = QPushButton("Browse...")
        self.le_dir_raw_images = QTextEdit()
        self.lo_dir_raw_tiffs.addWidget(self.le_dir_raw_images)
        self.lo_dir_raw_tiffs.addWidget(self.btn_browse_raw_tiffs)

        # Directory of raw ABFs
        self.lbl_dir_raw_abfs = QLabel("Directory of Raw ABFs: ")
        self.lo_block_3.addWidget(self.lbl_dir_raw_abfs)

        self.lo_dir_raw_abfs = QHBoxLayout()
        self.lo_block_3.addLayout(self.lo_dir_raw_abfs)

        self.le_dir_raw_abfs = QTextEdit()
        self.btn_browse_raw_abfs = QPushButton("Browse...")
        self.lo_dir_raw_abfs.addWidget(self.le_dir_raw_abfs)
        self.lo_dir_raw_abfs.addWidget(self.btn_browse_raw_abfs)

        # Directory of preprocessed TIFFs (Cal and Gauss)
        self.lbl_dir_preprocessed = QLabel("Directory of Preprocessed TIFFs: ")
        self.lo_block_3.addWidget(self.lbl_dir_preprocessed)

        self.lo_dir_preprocessed = QHBoxLayout()
        self.lo_block_3.addLayout(self.lo_dir_preprocessed)
        self.le_dir_preprocessed = QTextEdit()
        self.btn_browse_preprocessed = QPushButton("Browse...")
        self.lo_dir_preprocessed.addWidget(self.le_dir_preprocessed)
        self.lo_dir_preprocessed.addWidget(self.btn_browse_preprocessed)

        self.btn_start_check = QPushButton("Start Check")
        self.lo_block_3.addWidget(self.btn_start_check)
