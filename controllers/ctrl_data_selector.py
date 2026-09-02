## Modules
# Standard library imports
import json

# Third-party imports
import polars as pl
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from PySide6.QtWidgets import QAbstractItemView
from rich.console import Console

# Local application imports
from classes import DialogPickList
from utils import MODELS_DIR, REC_DB_PATH, ColumnSorter

# Set up rich console
console = Console()

# Constants
PICK_LIST_JSON_PATH = MODELS_DIR / "pick_list.json"
PICK_LIST_STATE_PATH = MODELS_DIR / "pick_list_state.json"


class CtrlDataSelector(QObject):
    pick_confirmed = Signal(str)

    def __init__(self, view) -> None:
        super().__init__()
        self.view = view
        self.current_dor: str | None = None
        self.df_pick_list = pl.DataFrame()

        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "data_selector_rec_data")
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))
        self.rec_data_db.open()

        self.view.tv_rec_summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.dlg_pick_list = DialogPickList()

        self.connect_signals()

        # Only clear stale picks if last session's picks were already exported;
        # otherwise recover them so unexported work isn't silently lost
        self.df_pick_list = self._load_df_pick_list_from_json()
        if not self.df_pick_list.is_empty():
            if self._read_exported_flag():
                self.clear_pick_list()
            else:
                console.print(f"[yellow]Recovered {len(self.df_pick_list)} unexported pick(s) from a previous session.[/yellow]")

    def connect_signals(self) -> None:
        self.view.btn_reset_all_filters.clicked.connect(self.reset_all_filters)

        self.view.btn_pick_selected.clicked.connect(self.pick_selected)
        self.view.btn_open_pick_list.clicked.connect(self.open_pick_list)

        self.dlg_pick_list.pick_list_changed.connect(self._on_dialog_pick_list_changed)
        self.dlg_pick_list.pick_confirmed.connect(self._on_pick_confirmed)

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

    def _load_df_pick_list_from_json(self) -> pl.DataFrame:
        if not PICK_LIST_JSON_PATH.exists():
            return pl.DataFrame()
        raw = pl.read_json(PICK_LIST_JSON_PATH)
        if raw.is_empty():
            return pl.DataFrame()
        return raw.with_columns(pl.all().cast(pl.Utf8)).fill_null("")

    def _read_exported_flag(self) -> bool:
        if not PICK_LIST_STATE_PATH.exists():
            return True  # no state recorded => nothing known to be pending, safe to clear
        try:
            state = json.loads(PICK_LIST_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True
        return bool(state.get("exported", True))

    def _write_exported_flag(self, exported: bool) -> None:
        PICK_LIST_STATE_PATH.write_text(json.dumps({"exported": exported}), encoding="utf-8")

    def save_pick_list(self, df: pl.DataFrame) -> None:
        """Persist pick list to JSON."""
        self.df_pick_list = df
        PICK_LIST_JSON_PATH.write_text(json.dumps(df.to_dicts(), indent=4))
        self._write_exported_flag(df.is_empty())

    def _on_pick_confirmed(self, path: str) -> None:
        self._write_exported_flag(True)
        self.pick_confirmed.emit(path)

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
        core_in_merged = [c for c in ColumnSorter.CORE_COLUMNS if c in all_cols]
        extra_cols = sorted(c for c in all_cols if c not in ColumnSorter.CORE_COLUMNS)
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
        core_in_table = [c for c in ColumnSorter.CORE_COLUMNS if c in all_cols]
        extra_cols = sorted(c for c in all_cols if c not in ColumnSorter.CORE_COLUMNS)
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
        self._write_exported_flag(True)

    def open_pick_list(self) -> None:
        self.dlg_pick_list.show()
        self.dlg_pick_list.raise_()
        self.dlg_pick_list.activateWindow()

    def _on_dialog_pick_list_changed(self) -> None:
        """Sync internal state after the pick list dialog modifies the JSON."""
        self.df_pick_list = self._load_df_pick_list_from_json()
        self._write_exported_flag(self.df_pick_list.is_empty())
