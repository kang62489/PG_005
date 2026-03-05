## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

# Local application imports
from classes import CheckableDropdown
from utils import UISizes

FILTER_COLUMNS = ["OBJ", "EXC", "EMI"]


class ViewDorQuery:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QHBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)
        self.filter_columns = FILTER_COLUMNS
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()

    def setup_block_1(self) -> None:
        self.lo_block_1 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_1)

        self.lbl_dor = QLabel("Date of Recording: ")
        self.lo_block_1.addWidget(self.lbl_dor)

        self.lw_dor = QListWidget()
        self.lo_block_1.addWidget(self.lw_dor)
        self.lw_dor.setFixedWidth(UISizes.LW_DOR_WIDTH)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

        self.lbl_animals = QLabel("Animals: ")
        self.lo_block_2.addWidget(self.lbl_animals)

        self.tv_animals = QTableView()
        self.lo_block_2.addWidget(self.tv_animals)
        self.tv_animals.setFixedHeight(UISizes.TV_ANIMALS_HEIGHT)
        self.tv_animals.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_animals.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tv_animals.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.lbl_injections = QLabel("Injection History: ")
        self.lo_block_2.addWidget(self.lbl_injections)

        self.tv_injections = QTableView()
        self.lo_block_2.addWidget(self.tv_injections)
        self.tv_injections.setFixedHeight(UISizes.TV_INJECTIONS_HEIGHT)
        self.tv_injections.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_injections.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tv_injections.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.gb_filter_panel = QGroupBox("Filter Panel")
        self.lo_block_2.addWidget(self.gb_filter_panel)
        self.gb_filter_panel.setLayout(QHBoxLayout())

        for col in self.filter_columns:
            dropdown = CheckableDropdown(col)
            dropdown.setObjectName(f"dd_{col}")
            setattr(self, f"dd_{col}", dropdown)
            self.gb_filter_panel.layout().addWidget(dropdown)

        self.btn_reset_all_filters = QPushButton("Reset All Filters")
        self.gb_filter_panel.layout().addWidget(self.btn_reset_all_filters)

        self.lo_db_view = QVBoxLayout()
        self.lo_block_2.addLayout(self.lo_db_view)

        self.lbl_rec_summary = QLabel("REC Summary: ")
        self.lo_db_view.addWidget(self.lbl_rec_summary)

        self.tv_rec_summary = QTableView()
        self.lo_db_view.addWidget(self.tv_rec_summary)
        self.tv_rec_summary.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_rec_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
