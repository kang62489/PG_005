## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
)

# Local application imports
from utils import UISizes


class ViewAlignSpike:
    def __init__(self, parent=None) -> None:
        self.popwin_container = parent
        self.lo_popwin_container = QVBoxLayout()
        self.popwin_container.setLayout(self.lo_popwin_container)
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()

    def setup_block_1(self) -> None:
        self.lo_block_1 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_1)

        self.lbl_dir_raw_abfs = QLabel("Directory of Raw ABFs: ")
        self.lo_block_1.addWidget(self.lbl_dir_raw_abfs)

        self.lo_dir_raw_abfs = QHBoxLayout()
        self.te_dir_raw_abfs = QTextEdit()
        self.te_dir_raw_abfs.setFixedHeight(UISizes.TE_DIRS_HEIGHT)
        self.btn_browse_raw_abfs = QPushButton("Browse...")
        self.btn_browse_raw_abfs.setFixedHeight(UISizes.BTN_RUN_ANALYSIS_HEIGHT)
        self.lo_dir_raw_abfs.addWidget(self.te_dir_raw_abfs)
        self.lo_dir_raw_abfs.addWidget(self.btn_browse_raw_abfs)
        self.lo_block_1.addLayout(self.lo_dir_raw_abfs)

        self.lo_proc_list_btns = QHBoxLayout()
        self.btn_load_proc_list = QPushButton("Load Processed List")
        self.btn_load_proc_list.setFixedHeight(UISizes.BTN_RUN_ANALYSIS_HEIGHT)
        self.btn_refresh_status = QPushButton("Refresh Status")
        self.btn_refresh_status.setFixedHeight(UISizes.BTN_RUN_ANALYSIS_HEIGHT)
        self.lo_proc_list_btns.addWidget(self.btn_load_proc_list)
        self.lo_proc_list_btns.addWidget(self.btn_refresh_status)
        self.lo_block_1.addLayout(self.lo_proc_list_btns)

        self.tv_proc_list = QTableView()
        self.lo_block_1.addWidget(self.tv_proc_list)

    def setup_block_2(self) -> None:
        self.lo_block_2 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_2)

        self.lo_conditions = QFormLayout()
        self.lo_block_2.addLayout(self.lo_conditions)

        self.lbl_cond_2 = QLabel("Z-score Normalization:")
        self.gb_norm = QButtonGroup()
        self.rb_norm_1 = QRadioButton("dF/F0 (Gauss)")
        self.rb_norm_2 = QRadioButton("dF/F0 (Gauss + ALS)")
        self.gb_norm.addButton(self.rb_norm_1)
        self.gb_norm.addButton(self.rb_norm_2)

        self.rb_norm_1.setChecked(True)  # Default to Gauss-only normalization

        self.lo_norm = QHBoxLayout()
        self.lo_norm.addWidget(self.rb_norm_1)
        self.lo_norm.addWidget(self.rb_norm_2)

        self.lo_conditions.addRow(self.lbl_cond_2, self.lo_norm)

        self.te_export_path = QTextEdit()
        self.te_export_path.setFixedHeight(UISizes.TE_EXPORT_PATH_HEIGHT)
        self.btn_export_browse = QPushButton("Browse...")
        self.btn_export_browse.setFixedWidth(UISizes.BTN_BROWSE_WIDTH)
        self.btn_export_browse.setFixedHeight(UISizes.BTN_RUN_ANALYSIS_HEIGHT)
        self.lo_export_path = QHBoxLayout()
        self.lo_export_path.addWidget(self.te_export_path)
        self.lo_export_path.addWidget(self.btn_export_browse)
        self.lo_conditions.addRow(self.lo_export_path)

        self.btn_confirm_analyzing_list = QPushButton("Export Analyzing List")
        self.btn_confirm_analyzing_list.setFixedHeight(UISizes.BTN_RUN_ANALYSIS_HEIGHT)
        self.lo_conditions.addRow(self.btn_confirm_analyzing_list)

        self.btn_run_analysis = QPushButton("Run Analysis")
        self.btn_run_analysis.setFixedHeight(UISizes.BTN_RUN_ANALYSIS_HEIGHT)
        self.btn_run_analysis.setStyleSheet(
            "QPushButton { color: darkgreen; font-weight: bold; } "
            "QPushButton:disabled { color: gray; font-weight: normal; }"
        )
        self.lo_conditions.addRow(self.btn_run_analysis)

        self.lo_processing_info = QFormLayout()
        self.lo_block_2.addLayout(self.lo_processing_info)

        self.lbl_processing_info = QLabel("Processing Info:")
        self.lo_processing_info.addRow(self.lbl_processing_info)

        self.lbl_current_total = QLabel("Current/Total: ")
        self.lbl_current_total.setFixedHeight(UISizes.LE_PROCESSING_INFO_HEIGHT)
        self.le_current_total = QLineEdit()
        self.le_current_total.setFixedHeight(UISizes.LE_PROCESSING_INFO_HEIGHT)

        self.lbl_status = QLabel("Status: ")
        self.lbl_status.setFixedHeight(UISizes.LE_PROCESSING_INFO_HEIGHT)
        self.le_status = QLineEdit()
        self.le_status.setFixedHeight(UISizes.LE_PROCESSING_INFO_HEIGHT)

        self.lo_processing_info.addRow(self.lbl_current_total, self.le_current_total)
        self.lo_processing_info.addRow(self.lbl_status, self.le_status)
