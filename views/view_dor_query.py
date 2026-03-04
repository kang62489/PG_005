## Modules
# Third-party imports
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QTableView, QVBoxLayout, QWidget

# Local application imports
from utils.params import UISizes


class ViewDorQuery:
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
        self.tv_animals.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_animals.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.lbl_injections = QLabel("Injection History: ")
        self.lo_block_2.addWidget(self.lbl_injections)

        self.tv_injections = QTableView()
        self.lo_block_2.addWidget(self.tv_injections)
        self.tv_injections.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_injections.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
