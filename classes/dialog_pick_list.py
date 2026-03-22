## Modules
# Third-party imports
import pandas as pd
from PySide6.QtCore import QFileSystemWatcher, QModelIndex, Qt
from PySide6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QTableView, QVBoxLayout
from rich.console import Console

# Local application imports
from utils import MODELS_DIR

from .model_from_dataframe import ModelFromDataFrame

# Set up rich console
console = Console()


class DialogPickList(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setup_view()
        self.load_pick_list()
        self.resize_to_table_content()
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.addPath(str(MODELS_DIR / "pick_list.json"))
        self.connect_signals()

    def setup_view(self) -> None:
        self.setWindowTitle("Pick List")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.lo_main = QVBoxLayout()
        self.setLayout(self.lo_main)

        self.tv_pick_list = QTableView()
        self.tv_pick_list.verticalHeader().setVisible(False)
        self.tv_pick_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tv_pick_list.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.tv_pick_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tv_pick_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.lo_main.addWidget(self.tv_pick_list)

        self.bbox_OK = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.lo_main.addWidget(self.bbox_OK, 1, Qt.AlignCenter)

    def connect_signals(self) -> None:
        self.bbox_OK.accepted.connect(self.accept)
        self.file_watcher.fileChanged.connect(self.load_pick_list)

    def load_pick_list(self) -> None:
        json_pick_list_path = MODELS_DIR / "pick_list.json"
        if not json_pick_list_path.exists():
            console.print("[bold red]Pick list JSON file not found![/bold red]")
            self.model_tv_pick_list = ModelFromDataFrame(None)
            self.tv_pick_list.setModel(self.model_tv_pick_list)
            return

        with json_pick_list_path.open() as f:
            self.model_tv_pick_list = ModelFromDataFrame(pd.read_json(f, orient="records", dtype=str))
            self.tv_pick_list.setModel(self.model_tv_pick_list)

        self.resize_to_table_content()

    def resize_to_table_content(self) -> None:
        # Calculate the width with number of columns of the table
        width_cols = 0
        for col in range(self.model_tv_pick_list.columnCount()):
            width_cols += self.tv_pick_list.columnWidth(col)

        # Add some padding + scrollbar width
        width_padding = 30
        scrollbar_width = self.tv_pick_list.verticalScrollBar().sizeHint().width()
        width_window = max(400, width_cols + width_padding + scrollbar_width)

        # Calculate the height of the table
        height_h_header = self.tv_pick_list.horizontalHeader().height()
        total_row_number = min(20, self.model_tv_pick_list.rowCount(QModelIndex()))
        height_total_rows = total_row_number * self.tv_pick_list.verticalHeader().defaultSectionSize()
        height_table = max(200, height_h_header + height_total_rows)
        height_btn_OK = 50

        total_height = height_h_header + height_total_rows + height_btn_OK

        # Set the size of the dialog
        self.tv_pick_list.setFixedHeight(height_table)
        self.resize(width_window, min(800, max(200, total_height)))
        self.setMinimumWidth(width_window)
