## Modules
# Third-party imports
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QTableView, QVBoxLayout, QWidget

# Local application imports


class ViewPickRaws:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QVBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)

        # self.setup_blocks()

        # self.btn_add_to_analysis_list = QPushButton("Add to Pick List")
        # self.lo_block_2.addWidget(self.btn_add_to_analysis_list)
