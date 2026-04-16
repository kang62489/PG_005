## Modules
# Standard library imports
import json

# Third-party imports
import polars as pl
from PySide6.QtCore import QFileSystemWatcher, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QVBoxLayout,
)
from rich.console import Console

# Local application imports
from utils import MODELS_DIR

from .model_from_dataframe import ModelFromDataFrame

# Set up rich console
console = Console()

PICK_LIST_JSON_PATH = MODELS_DIR / "pick_list.json"


class DialogPickList(QDialog):
    pick_list_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.df = pl.DataFrame()
        self.setup_view()
        self.load_pick_list()
        self.resize_to_table_content()
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.addPath(str(PICK_LIST_JSON_PATH))
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

        # Row removal buttons
        self.lo_edit_btns = QHBoxLayout()
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_clear_all = QPushButton("Clear All")
        self.lo_edit_btns.addWidget(self.btn_remove_selected)
        self.lo_edit_btns.addWidget(self.btn_clear_all)
        self.lo_main.addLayout(self.lo_edit_btns)

        self.bbox_OK = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.lo_main.addWidget(self.bbox_OK, 1, Qt.AlignmentFlag.AlignCenter)

    def connect_signals(self) -> None:
        self.bbox_OK.accepted.connect(self.accept)
        self.file_watcher.fileChanged.connect(self.load_pick_list)
        self.btn_remove_selected.clicked.connect(self.remove_selected)
        self.btn_clear_all.clicked.connect(self.clear_all)

    def load_pick_list(self) -> None:
        if not PICK_LIST_JSON_PATH.exists():
            console.print("[bold red]Pick list JSON file not found![/bold red]")
            self.df = pl.DataFrame()
            self.model_tv_pick_list = ModelFromDataFrame(None)
            self.tv_pick_list.setModel(self.model_tv_pick_list)
            return

        raw = pl.read_json(PICK_LIST_JSON_PATH)
        self.df = raw.with_columns(pl.all().cast(pl.Utf8)) if not raw.is_empty() else pl.DataFrame()
        self.model_tv_pick_list = ModelFromDataFrame(self.df if not self.df.is_empty() else None)
        self.tv_pick_list.setModel(self.model_tv_pick_list)
        self.resize_to_table_content()

    def remove_selected(self) -> None:
        selected = self.tv_pick_list.selectionModel().selectedRows()
        if not selected or self.df.is_empty():
            return
        rows_to_remove = {idx.row() for idx in selected}
        rows_to_keep = [i for i in range(len(self.df)) if i not in rows_to_remove]
        self._write_and_notify(self.df[rows_to_keep])

    def clear_all(self) -> None:
        self._write_and_notify(pl.DataFrame())

    def _write_and_notify(self, df: pl.DataFrame) -> None:
        """Write updated DataFrame to JSON and notify the controller."""
        PICK_LIST_JSON_PATH.write_text(json.dumps(df.to_dicts(), indent=4))
        self.pick_list_changed.emit()

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
