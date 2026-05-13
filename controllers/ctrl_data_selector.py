## Modules
# Standard library imports
import datetime
import json

# Third-party imports
import polars as pl
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from PySide6.QtWidgets import QAbstractItemView
from rich.console import Console

# Local application imports
from classes import DialogPickList
from utils.params import MODELS_DIR, REC_DB_PATH

# Set up rich console
console = Console()

# Constants
CORE_COLUMNS = ("Filename", "Timestamp", "OBJ", "EXC", "EMI", "FRAMES", "SLICE", "AT", "SENSOR", "PAIRED_ABF")
PICK_LIST_JSON_PATH = MODELS_DIR / "pick_list.json"


class CtrlDataSelector:
    def __init__(self, view) -> None:
        self.view = view
        self.current_dor: str | None = None
        self.df_pick_list = pl.DataFrame()

        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "data_selector_rec_data")
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))
        self.rec_data_db.open()

        self.view.tv_rec_summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.connect_signals()
        self.clear_pick_list()  # Clear pick list on startup to avoid confusion with old entries

    def connect_signals(self) -> None:
        self.view.btn_reset_all_filters.clicked.connect(self.reset_all_filters)

        self.view.btn_pick_selected.clicked.connect(self.pick_selected)
        self.view.btn_open_pick_list.clicked.connect(self.open_pick_list)
        self.view.btn_brief_gen.clicked.connect(self.brief_gen)
        self.view.btn_brief_export.clicked.connect(self.brief_export)

        # Connect filter dropdowns
        for col in self.view.filter_columns:
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.lw.itemChanged.connect(self.apply_filters)
        self.view.dd_shown_cols.lw.itemChanged.connect(self.toggle_shown_columns)

    def on_dor_changed(self, dor: str) -> None:
        self.current_dor = dor
        self.load_rec_summary(dor)

    def load_rec_summary(self, dor: str) -> None:
        # Clear rec summary table when switching DOR
        self.view.tv_rec_summary.setModel(None)

        # Clear filter dropdowns
        for col in self.view.filter_columns:
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.clear_items()

        self.view.dd_shown_cols.clear_items()

        # Display via QSqlTableModel
        model = QSqlTableModel(db=self.rec_data_db)
        tablename = f"REC_{dor}"
        model.setTable(tablename)
        model.select()

        if model.lastError().isValid() or model.rowCount() == 0:
            self.view.lbl_rec_summary.setText(f"Table Name Not Found: {tablename}")
            self.view.lbl_rec_summary.setStyleSheet("color: red; font-weight: bold")
            return

        self.view.tv_rec_summary.setModel(model)
        self.view.lbl_rec_summary.setText(f"Table Name: {tablename}")
        self.view.lbl_rec_summary.setStyleSheet("color: black; font-weight: normal")

        # Populate filter dropdowns
        for col in self.view.filter_columns:
            col_list = [model.record(row).value(col) for row in range(model.rowCount())]
            unique_col = set(col_list)
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.lw.blockSignals(True)
            dropdown.add_items(unique_col)
            dropdown.lw.blockSignals(False)

        # Populate "Show Columns" dropdown
        self.view.dd_shown_cols.lw.blockSignals(True)
        self.view.dd_shown_cols.add_items(
            model.headerData(col, Qt.Orientation.Horizontal) for col in range(model.columnCount())
        )
        self.view.dd_shown_cols.lw.blockSignals(False)

    def apply_filters(self) -> None:
        model = self.view.tv_rec_summary.model()
        if model is None:
            return

        conditions = []
        for col in self.view.filter_columns:
            dropdown = getattr(self.view, f"dd_{col}")
            checked = dropdown.checked_items()
            total = dropdown.lw.count()
            if total == 0 or len(checked) == total:
                continue  # all checked → no restriction needed
            if len(checked) == 0:
                conditions.append("1=0")  # nothing checked → show nothing
                break
            values = ", ".join(f"'{v}'" for v in checked)
            conditions.append(f"{col} IN ({values})")
        model.setFilter(" AND ".join(conditions))
        model.select()

    def reset_all_filters(self) -> None:
        for col in self.view.filter_columns:
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.lw.blockSignals(True)
            for i in range(dropdown.lw.count()):
                dropdown.lw.item(i).setCheckState(Qt.CheckState.Checked)
            dropdown.lw.blockSignals(False)

        self.view.dd_shown_cols.lw.blockSignals(True)
        for i in range(self.view.dd_shown_cols.lw.count()):
            self.view.dd_shown_cols.lw.item(i).setCheckState(Qt.CheckState.Checked)
        self.view.dd_shown_cols.lw.blockSignals(False)

        self.apply_filters()
        self.toggle_shown_columns()

    def toggle_shown_columns(self) -> None:
        model = self.view.tv_rec_summary.model()
        if model is None:
            return

        checked = self.view.dd_shown_cols.checked_items()
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) in checked:
                self.view.tv_rec_summary.showColumn(col)
            else:
                self.view.tv_rec_summary.hideColumn(col)

    # ── Pick list persistence ──────────────────────────────────────────────

    def save_pick_list(self, df: pl.DataFrame) -> None:
        """Persist pick list to JSON, then refresh processing brief."""
        self.df_pick_list = df
        PICK_LIST_JSON_PATH.write_text(json.dumps(df.to_dicts(), indent=4))
        self.brief_gen()

    def brief_gen(self) -> None:
        """Format and display the processing brief in the GUI text area."""
        # Auto-fill creation date and generate brief content based on current pick list and user inputs
        self.view.le_date_created.setText(datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d"))
        title = self.view.le_title.text().strip() or "Untitled"
        purposes_raw = self.view.te_purposes.toPlainText().strip()

        date_created = self.view.le_date_created.text().strip()

        lines = [f"Date Created: {date_created}"]
        lines.append(f"Analysis: {title}")

        if purposes_raw:
            purpose_lines = [line.strip() for line in purposes_raw.splitlines() if line.strip()]
            indent = " " * 4  # align under first bullet
            lines.append("Purposes:")
            for purpose_line in purpose_lines:
                lines.append(f"{indent} {purpose_line}")

        lines.append("\nPicked:")

        if self.df_pick_list.is_empty() or "Filename" not in self.df_pick_list.columns:
            lines.append("  (No records picked yet)")
        else:
            cols = self.df_pick_list.columns
            rows = sorted(self.df_pick_list.to_dicts(), key=lambda r: r["Filename"])
            for row in rows:
                abf = row.get("PAIRED_ABF") or "" if "PAIRED_ABF" in cols else ""
                if abf:
                    dor = row["Filename"].split("-")[0]
                    entry = f"[{row['Filename']}, {dor}_{abf}.abf]"
                else:
                    entry = f"[{row['Filename']}]"
                lines.append(entry)
            lines.append(f"\nTotal {len(rows)} records picked")

        self.view.te_processing_brief.setPlainText("\n".join(lines))

    def brief_export(self) -> None:
        """Export processing brief as .txt into results/, with auto serial suffix."""
        date_created = self.view.le_date_created.text().strip()
        if not date_created:
            console.print("[bold red]Date Created is empty — run Generate Brief first.[/bold red]")
            return

        MODELS_DIR.mkdir(exist_ok=True)

        # Find next available serial number for today's date
        existing = sorted(MODELS_DIR.glob(f"proc_brief_{date_created}_*.txt"))
        if existing:
            last_serial = int(existing[-1].stem.rsplit("_", 1)[-1])
            serial = last_serial + 1
        else:
            serial = 0
        note_path = MODELS_DIR / f"proc_brief_{date_created}_{serial:03d}.txt"

        note_text = self.view.te_processing_brief.toPlainText().strip()
        note_path.write_text(note_text, encoding="utf-8")
        console.print(f"[bold green]Processing brief saved → {note_path}[/bold green]")

    # ── Pick actions ───────────────────────────────────────────────────────

    def check_pick_list(self, df_selected: pl.DataFrame) -> None:
        df_saved = pl.DataFrame()
        if PICK_LIST_JSON_PATH.exists():
            raw = pl.read_json(PICK_LIST_JSON_PATH)
            if not raw.is_empty():
                df_saved = raw.with_columns(pl.all().cast(pl.Utf8)).fill_null("")

        if not df_saved.is_empty():
            new_rows = df_selected.join(df_saved, on="Filename", how="anti")
            df_merged = pl.concat([df_saved, new_rows], how="diagonal").fill_null("")
        else:
            df_merged = df_selected

        # Re-apply CORE_COLUMNS ordering after concat (diagonal concat uses df_saved column order,
        # so new columns from new_rows would be appended instead of placed correctly)
        all_cols = df_merged.columns
        core_in_merged = [c for c in CORE_COLUMNS if c in all_cols]
        extra_cols = sorted(c for c in all_cols if c not in CORE_COLUMNS)
        df_merged = df_merged.select(core_in_merged + extra_cols)

        self.save_pick_list(df_merged.sort("Filename"))

    def pick_selected(self) -> None:
        if self.view.tv_rec_summary.model() is None:
            console.print("[bold red]No table to pick from![/bold red]")
            return

        selected = self.view.tv_rec_summary.selectionModel().selectedRows()
        if not selected:
            console.print("[bold red]No row selected![/bold red]")
            return

        # Build column order: CORE columns first (in defined order), then extras alphabetically
        model = self.view.tv_rec_summary.model()
        all_cols = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(model.columnCount())]
        core_in_table = [c for c in CORE_COLUMNS if c in all_cols]
        extra_cols = sorted(c for c in all_cols if c not in CORE_COLUMNS)
        ordered_cols = core_in_table + extra_cols

        selected_row_data = []
        for idx in sorted(selected):
            record = model.record(idx.row())
            selected_row_data.append(
                {col: (str(record.value(col)) if record.value(col) is not None else "") for col in ordered_cols}
            )

        df_selected = pl.DataFrame(selected_row_data).with_columns(pl.all().cast(pl.Utf8))
        console.print(f"[bold green]Selected {len(df_selected)} row(s) from {self.current_dor}.[/bold green]")
        self.check_pick_list(df_selected)

    def clear_pick_list(self) -> None:
        self.df_pick_list = pl.DataFrame()
        PICK_LIST_JSON_PATH.write_text(json.dumps([], indent=4))

    def open_pick_list(self) -> None:
        self.dlg_pick_list = DialogPickList()
        self.dlg_pick_list.pick_list_changed.connect(self._on_dialog_pick_list_changed)
        self.dlg_pick_list.show()

    def _on_dialog_pick_list_changed(self) -> None:
        """Sync GUI log after the pick list dialog modifies the JSON."""
        if PICK_LIST_JSON_PATH.exists():
            raw = pl.read_json(PICK_LIST_JSON_PATH)
            df = raw.with_columns(pl.all().cast(pl.Utf8)).fill_null("") if not raw.is_empty() else pl.DataFrame()
        else:
            df = pl.DataFrame()
        self.df_pick_list = df
        self.brief_gen()
