## Modules
# Standard library imports
import datetime
import json
import shutil

# Third-party imports
import polars as pl
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from rich.console import Console

# Local application imports
from classes import DialogPickList
from utils.params import MODELS_DIR, REC_DB_PATH, RESULTS_DIR
from views import ViewDataSelector

# Set up rich console
console = Console()

# Constants
CORE_COLUMNS = ("Filename", "Timestamp", "OBJ", "EXC", "EMI", "FRAMES", "SLICE", "AT")
PICK_LIST_JSON_PATH = MODELS_DIR / "pick_list.json"
PICK_LIST_XLSX_PATH = MODELS_DIR / "pick_list.xlsx"


class CtrlDataSelector:
    def __init__(self, view: ViewDataSelector) -> None:
        self.view = view
        self.current_dor: str | None = None
        self.df_pick_list = pl.DataFrame()

        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "data_selector_rec_data")
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))
        self.rec_data_db.open()

        self.connect_signals()
        self.clear_pick_list()  # Clear pick list on startup to avoid confusion with old entries

    def connect_signals(self) -> None:
        self.view.btn_reset_all_filters.clicked.connect(self.reset_all_filters)

        self.view.btn_pick_selected.clicked.connect(self.pick_selected)
        self.view.btn_open_pick_list.clicked.connect(self.open_pick_list)
        self.view.btn_note_gen.clicked.connect(self.note_gen)
        self.view.btn_note_export.clicked.connect(self.note_export)

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
        """Persist pick list to JSON + XLSX, then refresh GUI log."""
        self.df_pick_list = df
        PICK_LIST_JSON_PATH.write_text(json.dumps(df.to_dicts(), indent=4))
        if not df.is_empty():
            try:
                df.write_excel(PICK_LIST_XLSX_PATH)
            except Exception as e:
                console.print(f"[bold yellow]Warning: could not write XLSX — {e}[/bold yellow]")
        self.note_gen()

    def note_gen(self) -> None:
        """Format and display the analysis note in the GUI text area."""
        # Auto-fill creation date and generate note content based on current pick list and user inputs
        self.view.le_date_created.setText(datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%b-%d"))
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
            df_with_folder = self.df_pick_list.with_columns(
                pl.col("Filename").str.split("-").list.first().alias("_folder")
            )
            cols = df_with_folder.columns
            total = 0
            for folder in df_with_folder["_folder"].unique(maintain_order=True).sort().to_list():
                group = df_with_folder.filter(pl.col("_folder") == folder)
                rows = sorted(group.to_dicts(), key=lambda r: r["Filename"])
                lines.append(f"\n{folder}/")
                for row in rows:
                    obj = row.get("OBJ") or "" if "OBJ" in cols else ""
                    abf = row.get("PAIRED_ABF") or "" if "PAIRED_ABF" in cols else ""
                    suffix = "  |  ".join(filter(None, [obj, abf]))
                    entry = f"    {row['Filename']}"
                    if suffix:
                        entry += f"  |  {suffix}"
                    lines.append(entry)
                n = len(rows)
                total += n
                lines.append(f"\n{n} records picked")
            lines.append(f"\nTotal {total} records picked")

        self.view.te_analysis_notes.setPlainText("\n".join(lines))

    def note_export(self) -> None:
        """Export analysis note as .txt and pick list as .xlsx into results/."""
        date_created = self.view.le_date_created.text().strip()
        if not date_created:
            console.print("[bold red]Date Created is empty — run Generate Note first.[/bold red]")
            return

        RESULTS_DIR.mkdir(exist_ok=True)

        # Save analysis note text
        note_text = self.view.te_analysis_notes.toPlainText().strip()
        note_path = RESULTS_DIR / f"analysis_note_{date_created}.txt"
        note_path.write_text(note_text, encoding="utf-8")
        console.print(f"[bold green]Analysis note saved → {note_path}[/bold green]")

        # Move pick list XLSX
        if PICK_LIST_XLSX_PATH.exists():
            dest = RESULTS_DIR / f"pick_list_{date_created}.xlsx"
            shutil.copy2(PICK_LIST_XLSX_PATH, dest)
            console.print(f"[bold green]Pick list copied → {dest}[/bold green]")
        else:
            console.print("[bold yellow]No pick_list.xlsx found to export.[/bold yellow]")

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
        if PICK_LIST_XLSX_PATH.exists():
            PICK_LIST_XLSX_PATH.unlink()

    def open_pick_list(self) -> None:
        self.dlg_pick_list = DialogPickList()
        self.dlg_pick_list.pick_list_changed.connect(self._on_dialog_pick_list_changed)
        self.dlg_pick_list.show()

    def _on_dialog_pick_list_changed(self) -> None:
        """Sync XLSX and GUI log after the pick list dialog modifies the JSON."""
        if PICK_LIST_JSON_PATH.exists():
            raw = pl.read_json(PICK_LIST_JSON_PATH)
            df = raw.with_columns(pl.all().cast(pl.Utf8)).fill_null("") if not raw.is_empty() else pl.DataFrame()
        else:
            df = pl.DataFrame()
        self.df_pick_list = df
        if not df.is_empty():
            try:
                df.write_excel(PICK_LIST_XLSX_PATH)
            except Exception as e:
                console.print(f"[bold yellow]Warning: could not write XLSX — {e}[/bold yellow]")
        self.note_gen()
