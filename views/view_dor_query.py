## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Local application imports
from utils import UISizes


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
        # Animal Records
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

        # Injection History
        self.lbl_injections = QLabel("Injection History: ")
        self.lo_block_2.addWidget(self.lbl_injections)

        self.tv_injections = QTableView()
        self.lo_block_2.addWidget(self.tv_injections)
        self.tv_injections.setFixedHeight(UISizes.TV_INJECTIONS_HEIGHT)
        self.tv_injections.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tv_injections.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tv_injections.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Data Folder Properties
        self.lo_data_folder_props = QHBoxLayout()
        self.lo_block_2.addLayout(self.lo_data_folder_props)

        self.lo_data_folder_props_left = QVBoxLayout()
        self.lo_data_folder_props_right = QVBoxLayout()
        self.lo_data_folder_props.addLayout(self.lo_data_folder_props_left)
        self.lo_data_folder_props.addLayout(self.lo_data_folder_props_right)

        # Left side of data folder properties: metadata and log insertion
        self.lo_metadata = QFormLayout()
        self.lo_data_folder_props_left.addLayout(self.lo_metadata)

        self.le_last_modified = QLineEdit()
        self.le_last_modified.setReadOnly(True)

        self.le_system = QLineEdit()
        self.le_keywords = QLineEdit()

        self.le_prj = QLineEdit()

        self.te_descriptions = QTextEdit()
        self.te_descriptions.setFixedHeight(UISizes.TE_DESCRIPTIONS_HEIGHT)

        self.btn_update_descriptions = QPushButton("Update")
        self.lo_descriptions = QHBoxLayout()
        self.lo_descriptions.addWidget(self.te_descriptions)
        self.lo_descriptions.addWidget(self.btn_update_descriptions)

        self.te_findings = QTextEdit()
        self.btn_update_findings = QPushButton("Update")
        self.lo_findings = QHBoxLayout()
        self.lo_findings.addWidget(self.te_findings)
        self.lo_findings.addWidget(self.btn_update_findings)

        self.lo_metadata.addRow("Last Modified: ", self.le_last_modified)
        self.lo_metadata.addRow("System: ", self.le_system)
        self.lo_metadata.addRow("Keywords: ", self.le_keywords)
        self.lo_metadata.addRow("Project: ", self.le_prj)
        self.lo_metadata.addRow("Descriptions: ", self.lo_descriptions)
        self.lo_metadata.addRow("Findings: ", self.lo_findings)

        # Right side of data folder properties: log viewer
        self.lbl_insert_log = QLabel("Insert Log: ")
        self.te_insert_log = QTextEdit()
        self.lo_data_folder_props_right.addWidget(self.lbl_insert_log)
        self.lo_data_folder_props_right.addWidget(self.te_insert_log)

        self.lo_buttons = QHBoxLayout()
        self.lo_data_folder_props_right.addLayout(self.lo_buttons)
        self.btn_insert_log = QPushButton("Insert Log")
        self.btn_clear_log = QPushButton("Clear")
        self.lo_buttons.addWidget(self.btn_insert_log)
        self.lo_buttons.addWidget(self.btn_clear_log)

        self.lbl_log_date = QLabel("Jump to Log Date: ")
        self.lbl_log_date.setFixedHeight(24)
        self.cb_log_date = QComboBox()
        self.lo_data_folder_props_right.addWidget(self.lbl_log_date)
        self.lo_data_folder_props_right.addWidget(self.cb_log_date)

        self.te_log_contents = QTextEdit()
        self.te_log_contents.setReadOnly(True)
        self.te_log_contents.setFixedHeight(UISizes.TE_LOG_CONTENTS_HEIGHT)
        self.lo_data_folder_props_right.addWidget(self.te_log_contents)


        self.btn_scan_files = QPushButton("Scan Files in the Data Folder")
        self.te_folder_structure = QTextEdit()
        self.te_folder_structure.setReadOnly(True)
        self.te_folder_structure.setFixedHeight(UISizes.TE_FOLDER_STRUCTURE_HEIGHT)

        self.lo_data_folder_props_right.addWidget(self.btn_scan_files)
        self.lo_data_folder_props_right.addWidget(self.te_folder_structure)
