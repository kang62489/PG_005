## Modules
# Standard library imports
import json

# Third-party imports
import polars as pl
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from rich.console import Console
from tabulate import tabulate

# Local application imports
from classes import DialogPickList
from utils.params import MODELS_DIR, REC_DB_PATH
from views import ViewDataSelector

# Set up rich console
console = Console()

# Constants
COLUMNS_TO_PICK = ("Filename", "OBJ", "EMI", "FRAMES", "SLICE", "AT", "ABF_NUMBER")
PICK_LIST_PATH = MODELS_DIR / "pick_list.json"

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

        # Display via QSqlTableModel, hide unwanted columns
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

        # Hide columns
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

    def check_pick_list(self, df_selected: pl.DataFrame) -> None:
        df_saved = pl.DataFrame(schema=dict.fromkeys(COLUMNS_TO_PICK, pl.Utf8))
        if PICK_LIST_PATH.exists():
            df_saved = pl.read_json(PICK_LIST_PATH).with_columns(pl.all().cast(pl.Utf8)).fill_null("")

        if not df_saved.is_empty():
            new_rows = df_selected.join(df_saved, on=list(COLUMNS_TO_PICK), how="anti")
            df_selected = pl.concat([df_saved, new_rows])

        self.df_pick_list = df_selected.sort("Filename")
        PICK_LIST_PATH.write_text(json.dumps(self.df_pick_list.to_dicts(), indent=4))

        console.print(
            "\n[bold green]Pick List (Latest):[/bold green]\n",
            tabulate(self.df_pick_list.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"),
        )

    def pick_selected(self) -> None:
        if self.view.tv_rec_summary.model() is None:
            console.print("[bold red]No table to pick from![/bold red]")
            return

        selected = self.view.tv_rec_summary.selectionModel().selectedRows()
        if not selected:
            console.print("[bold red]No row selected![/bold red]")
            return

        selected_row_data = []
        for idx in sorted(selected):
            record = self.view.tv_rec_summary.model().record(idx.row())
            selected_row_data.append(
                {col: (str(record.value(col)) if record.value(col) is not None else "") for col in COLUMNS_TO_PICK}
            )

        df_selected = pl.DataFrame(selected_row_data).with_columns(pl.all().cast(pl.Utf8))
        console.print(
            "\n[bold green]Selected Rows:[/bold green]\n",
            tabulate(df_selected.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"),
        )

        self.check_pick_list(df_selected)

    def clear_pick_list(self) -> None:
        self.df_pick_list = pl.DataFrame()
        PICK_LIST_PATH.write_text(json.dumps([], indent=4))

    def open_pick_list(self) -> None:
        self.dlg_pick_list = DialogPickList()
        self.dlg_pick_list.show()
