## Modules
# Third-party imports
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QListWidget, QPushButton, QTableView, QVBoxLayout, QWidget


class ViewPickRaws:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QVBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)

        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()

    def setup_block_1(self) -> None:
        self.gb_filter_panel = QGroupBox("Filter Panel")
        self.lo_tab_container.addWidget(self.gb_filter_panel)
        self.gb_filter_panel.setLayout(QHBoxLayout())

        for col in ["OBJ", "EXC", "EMI"]:
            lo_cond = QVBoxLayout()
            self.gb_filter_panel.layout().addLayout(lo_cond)

            lbl_cond_name = QLabel(col)
            lo_cond.addWidget(lbl_cond_name)

            lw_cond = QListWidget()
            lw_cond.setObjectName(f"lw_{col}")
            setattr(self, f"lw_{col}", lw_cond)
            lo_cond.addWidget(lw_cond)

            lo_btn_row = QHBoxLayout()
            btn_all = QPushButton("All")
            btn_all.setObjectName(f"btn_all_{col}")
            setattr(self, f"btn_all_{col}", btn_all)

            btn_clear = QPushButton("Clear")
            btn_clear.setObjectName(f"btn_clear_{col}")
            setattr(self, f"btn_clear_{col}", btn_clear)

            lo_btn_row.addWidget(btn_all)
            lo_btn_row.addWidget(btn_clear)

            lo_cond.addLayout(lo_btn_row)

        self.btn_reset_all_filters = QPushButton("Reset All Filters")
        self.gb_filter_panel.layout().addWidget(self.btn_reset_all_filters)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QHBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)

        self.lo_db_view = QVBoxLayout()
        self.lo_block_2.addLayout(self.lo_db_view)

        self.lbl_rec_summary = QLabel("REC Summary: ")
        self.lo_db_view.addWidget(self.lbl_rec_summary)

        self.tv_rec_summary = QTableView()
        self.lo_db_view.addWidget(self.tv_rec_summary)

        self.btn_add_to_analysis_list = QPushButton("Add to Pick List")
        self.lo_block_2.addWidget(self.btn_add_to_analysis_list)
