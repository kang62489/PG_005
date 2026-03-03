## Modules

# Third-party imports
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QTableView

# Local application imports


class ViewDorQuery:
    def __init__(self, parent: QWidget | None = None) -> None:
        self.tab_container = parent
        self.lo_tab_container = QHBoxLayout()
        self.tab_container.setLayout(self.lo_tab_container)
        
        self.setup_blocks()
        
        
    def setup_blocks(self):
        self.setup_block_1()
        self.setup_block_2()
        
    def setup_block_1(self):
        self.lo_block_1 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_1)
        
        self.lbl_dor = QLabel("Date of Recording: ")
        self.lo_block_1.addWidget(self.lbl_dor)
        
        
        self.lv_dor = QListWidget()
        self.lo_block_1.addWidget(self.lv_dor)
        
    def setup_block_2(self):
        self.lo_block_2 = QVBoxLayout()
        self.lo_tab_container.addLayout(self.lo_block_2)
        
        self.lbl_animals = QLabel("Animals: ")
        self.lo_block_2.addWidget(self.lbl_animals)
        
        self.tv_exp_info = QTableView()
        self.lo_block_2.addWidget(self.tv_exp_info)
        
        self.lbl_injection_history = QLabel("Injection History: ")
        self.lo_block_2.addWidget(self.lbl_injection_history)
        
        self.tv_inj_history = QTableView()
        self.lo_block_2.addWidget(self.tv_inj_history)