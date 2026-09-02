## Modules
# Standard library imports
import datetime
import json
from pathlib import Path

# Third-party imports
import polars as pl
from PySide6.QtCore import QFileSystemWatcher, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QTextEdit,
    QVBoxLayout,
)
from rich.console import Console

# Local application imports
from utils import MODELS_DIR, UISizes

from .model_from_dataframe import ModelFromDataFrame

# Set up rich console
console = Console()

PICK_LIST_JSON_PATH = MODELS_DIR / "pick_list.json"


class DialogPickList(QDialog):
    pick_list_changed = Signal()
    pick_confirmed = Signal(str)

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
        self.lo_main = QVBoxLayout()
        self.setLayout(self.lo_main)

        self.tv_pick_list = QTableView()
        self.tv_pick_list.verticalHeader().setVisible(False)
        self.tv_pick_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tv_pick_list.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        self.tv_pick_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tv_pick_list.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.lo_main.addWidget(self.tv_pick_list)

        # Pick list metadata form (left) + preview (right)
        self.lo_lower = QHBoxLayout()
        self.lo_main.addLayout(self.lo_lower)

        self.lo_form = QVBoxLayout()
        self.lo_lower.addLayout(self.lo_form)

        self.lo_pick_list_form = QFormLayout()
        self.le_title = QLineEdit()
        self.te_purposes = QTextEdit()
        self.le_date_created = QLineEdit()
        self.lo_pick_list_form.addRow(QLabel("Date Created:"), self.le_date_created)
        self.lo_pick_list_form.addRow(QLabel("Title:"), self.le_title)
        self.lo_pick_list_form.addRow(QLabel("Purpose:"), self.te_purposes)
        self.lo_form.addLayout(self.lo_pick_list_form)

        self.lo_preview = QVBoxLayout()
        self.lo_lower.addLayout(self.lo_preview)

        self.lbl_pick_list_preview = QLabel("Preview")
        self.lo_preview.addWidget(self.lbl_pick_list_preview)
        self.te_pick_list_preview = QTextEdit()
        self.te_pick_list_preview.setReadOnly(True)
        self.te_pick_list_preview.setFixedHeight(UISizes.TE_PICK_LIST_PREVIEW_HEIGHT)
        self.lo_preview.addWidget(self.te_pick_list_preview)

        # Row removal / export buttons
        self.lo_edit_btns = QHBoxLayout()
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_pick_list_export = QPushButton("Export Pick List and Check File Status")
        self.btn_clear_all.setStyleSheet("color: red; font-weight: bold;")
        self.btn_pick_list_export.setStyleSheet("color: darkgreen; font-weight: bold;")
        self.lo_edit_btns.addWidget(self.btn_remove_selected)
        self.lo_edit_btns.addWidget(self.btn_clear_all)
        self.lo_edit_btns.addWidget(self.btn_pick_list_export)
        self.lo_main.addLayout(self.lo_edit_btns)

    def connect_signals(self) -> None:
        self.file_watcher.fileChanged.connect(self.load_pick_list)
        self.btn_remove_selected.clicked.connect(self.remove_selected)
        self.btn_clear_all.clicked.connect(self.clear_all)
        self.btn_pick_list_export.clicked.connect(self.pick_list_export)
        self.le_title.textChanged.connect(lambda _: self.refresh_preview())
        self.te_purposes.textChanged.connect(self.refresh_preview)

    def load_pick_list(self) -> None:
        if not PICK_LIST_JSON_PATH.exists():
            console.print("[bold red]Pick list JSON file not found![/bold red]")
            self.df = pl.DataFrame()
            self.model_tv_pick_list = ModelFromDataFrame(None)
            self.tv_pick_list.setModel(self.model_tv_pick_list)
            self.refresh_preview()
            self.btn_pick_list_export.setEnabled(False)
            return

        raw = pl.read_json(PICK_LIST_JSON_PATH)
        self.df = raw.with_columns(pl.all().cast(pl.Utf8)) if not raw.is_empty() else pl.DataFrame()
        self.model_tv_pick_list = ModelFromDataFrame(self.df if not self.df.is_empty() else None)
        self.tv_pick_list.setModel(self.model_tv_pick_list)
        self.resize_to_table_content()
        self.refresh_preview()
        self.btn_pick_list_export.setEnabled(not self.df.is_empty())

    def refresh_preview(self) -> None:
        self.le_date_created.setText(datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d"))
        title = self.le_title.text().strip() or "Untitled"
        purposes_raw = self.te_purposes.toPlainText().strip()
        date_created = self.le_date_created.text().strip()

        lines = [f"Date Created: {date_created}"]
        lines.append(f"Analysis: {title}")

        if purposes_raw:
            purpose_lines = [line.strip() for line in purposes_raw.splitlines() if line.strip()]
            indent = " " * 4  # align under first bullet
            lines.append("Purposes:")
            for purpose_line in purpose_lines:
                lines.append(f"{indent} {purpose_line}")

        lines.append("\nPicked: [raw_tiff_name, paired_abf]")

        if self.df.is_empty() or "Filename" not in self.df.columns:
            lines.append("  (No records picked yet)")
        else:
            cols = self.df.columns
            rows = sorted(self.df.to_dicts(), key=lambda r: r["Filename"])
            for row in rows:
                abf = row.get("PAIRED_ABF") or "" if "PAIRED_ABF" in cols else ""
                if abf:
                    dor = row["Filename"].split("-")[0]
                    abf_str = f"{dor}_{abf}.abf"
                else:
                    abf_str = "N/A"
                lines.append(f"[{row['Filename']}, {abf_str}]")
            lines.append(f"\nTotal {len(rows)} records picked")

        self.te_pick_list_preview.setPlainText("\n".join(lines))

    def _parse_pick_list_header(self, path: Path) -> tuple[str, str]:
        """Extract normalized title and purposes from a pick list .txt for duplicate detection."""
        title = ""
        purposes_lines: list[str] = []
        in_purposes = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Analysis:"):
                title = line.split(":", 1)[1].strip()
                in_purposes = False
            elif line.strip() == "Purposes:":
                in_purposes = True
            elif in_purposes:
                stripped = line.strip()
                if not stripped:
                    in_purposes = False
                else:
                    purposes_lines.append(stripped)
        return title, "\n".join(purposes_lines)

    def pick_list_export(self) -> None:
        """Export pick list as .txt, with auto serial suffix."""
        pick_list_context = self.te_pick_list_preview.toPlainText().strip()
        if not pick_list_context:
            console.print("[bold red]Preview is empty — nothing to export.[/bold red]")
            return

        date_created = self.le_date_created.text().strip()
        if not date_created:
            console.print("[bold red]Date Created is empty — please enter a date before exporting.[/bold red]")
            return

        MODELS_DIR.mkdir(exist_ok=True)

        current_title = self.le_title.text().strip() or "Untitled"
        current_purposes = "\n".join(
            line.strip() for line in self.te_purposes.toPlainText().splitlines() if line.strip()
        )

        # Check if any existing pick list from the same date has identical title + purposes
        existing = sorted(
            p for p in MODELS_DIR.glob(f"pick_{date_created}_*.txt")
            if not p.stem.endswith("_checked")
        )
        match_path = next(
            (p for p in existing if self._parse_pick_list_header(p) == (current_title, current_purposes)),
            None,
        )

        if match_path:
            pick_list_path = match_path
            console.print(f"[yellow]Same title+purpose found — overwriting {match_path.name}[/yellow]")
        else:
            last_serial = int(existing[-1].stem.rsplit("_", 1)[-1]) if existing else -1
            pick_list_path = MODELS_DIR / f"pick_{date_created}_{last_serial + 1:03d}.txt"

        pick_list_path.write_text(pick_list_context, encoding="utf-8")
        console.print(f"[bold green]Pick list saved → {pick_list_path}[/bold green]")

        self.pick_confirmed.emit(str(pick_list_path))

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
        # Constrain the table to its content; everything else (form, fixed-height
        # preview, buttons) already has a natural size, so adjustSize() below
        # shrinks/grows the whole dialog to fit automatically.
        width_cols = 0
        for col in range(self.model_tv_pick_list.columnCount()):
            width_cols += self.tv_pick_list.columnWidth(col)

        width_padding = 30
        scrollbar_width = self.tv_pick_list.verticalScrollBar().sizeHint().width()
        width_window = max(700, width_cols + width_padding + scrollbar_width)

        height_h_header = self.tv_pick_list.horizontalHeader().height()
        total_row_number = min(20, self.model_tv_pick_list.rowCount(QModelIndex()))
        height_total_rows = total_row_number * self.tv_pick_list.verticalHeader().defaultSectionSize()
        height_table = max(200, height_h_header + height_total_rows)

        self.tv_pick_list.setFixedHeight(height_table)
        self.tv_pick_list.setMinimumWidth(width_window - width_padding)
        self.adjustSize()
