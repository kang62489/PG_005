## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
)

class ViewAlignSpike:
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

        self.btn_load_proc_list = QPushButton("Load Processed List")
        self.lo_block_1.addWidget(self.btn_load_proc_list)