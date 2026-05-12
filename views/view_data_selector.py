## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QTextEdit,
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
        self.tv_rec_summary.setFixedWidth(UISizes.TV_REC_SUMMARY_WIDTH)

        self.lo_db_view.addWidget(self.tv_rec_summary)
        self.tv_rec_summary.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_rec_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

        self.gb_processing_brief = QGroupBox("Processing Brief")
        self.gb_processing_brief.setLayout(QVBoxLayout())
        self.lo_block_2.addWidget(self.gb_processing_brief)

        # Form fields
        self.lo_processing_brief_form = QFormLayout()
        self.le_title = QLineEdit()
        self.te_purposes = QTextEdit()
        self.le_date_created = QLineEdit()
        self.lo_processing_brief_form.addRow(QLabel("Date Created:"), self.le_date_created)
        self.lo_processing_brief_form.addRow(QLabel("Title:"), self.le_title)
        self.lo_processing_brief_form.addRow(QLabel("Purpose:"), self.te_purposes)
        self.gb_processing_brief.layout().addLayout(self.lo_processing_brief_form)

        # Preview
        self.lbl_preview_processing_brief = QLabel("Preview")
        self.gb_processing_brief.layout().addWidget(self.lbl_preview_processing_brief)
        self.te_processing_brief = QTextEdit()
        self.te_processing_brief.setReadOnly(True)
        self.gb_processing_brief.layout().addWidget(self.te_processing_brief)

        # Buttons
        self.lo_processing_brief_btns = QHBoxLayout()
        self.btn_brief_gen = QPushButton("Generate Brief")
        self.btn_brief_export = QPushButton("Export Processing Brief")
        self.lo_processing_brief_btns.addWidget(self.btn_brief_gen)
        self.lo_processing_brief_btns.addWidget(self.btn_brief_export)
        self.gb_processing_brief.layout().addLayout(self.lo_processing_brief_btns)
