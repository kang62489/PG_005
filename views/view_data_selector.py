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
    QVBoxLayout,
    QWidget,
)

# Local application imports
from classes import CheckableDropdown
from utils.params import UISizes

# Constants
FILTER_COLUMNS = ["OBJ", "EXC", "EMI"]


class ViewDataSelector:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.popwin_container = parent
        self.lo_popwin_container = QHBoxLayout()
        self.popwin_container.setLayout(self.lo_popwin_container)
        self.filter_columns = FILTER_COLUMNS
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()

    def setup_block_1(self) -> None:
        self.lo_block_1 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_1)

        # Filter Panel
        self.lo_panels = QHBoxLayout()
        self.lo_block_1.addLayout(self.lo_panels)

        self.lo_panels.addStretch()

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
        self.gb_filter_panel.layout().addStretch()
        self.lo_db_view = QVBoxLayout()
        self.lo_block_1.addLayout(self.lo_db_view)

        # Pick List Control
        self.gb_pick_list = QGroupBox("Pick List Control")
        self.gb_pick_list.setLayout(QHBoxLayout())
        self.gb_pick_list.setFixedWidth(UISizes.GB_PICK_LIST_WIDTH)
        self.btn_pick_selected = QPushButton("Pick Selected")
        self.btn_open_pick_list = QPushButton("Open Pick List")
        self.gb_pick_list.layout().addWidget(self.btn_pick_selected, 1)
        self.gb_pick_list.layout().addWidget(self.btn_open_pick_list, 1)
        self.lo_panels.addWidget(self.gb_pick_list)
        self.lo_panels.addStretch()

        # REC Summary
        self.lbl_rec_summary = QLabel("REC Summary: ")
        self.lo_db_view.addWidget(self.lbl_rec_summary)

        self.tv_rec_summary = QTableView()

        self.lo_db_view.addWidget(self.tv_rec_summary)
        self.tv_rec_summary.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_rec_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
