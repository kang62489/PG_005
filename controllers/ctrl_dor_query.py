## Modules
# Standard library imports
import sqlite3
from pathlib import Path

# Third-party imports
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from rich.console import Console
from tabulate import tabulate

# Local application imports
from classes import DialogPickList
from utils import EXP_DB_PATH, MODELS_DIR, REC_DB_PATH
from views import ViewDorQuery

ANIMALS_KEEP = {"Animal_ID", "DOB", "Ages", "Genotype", "Sex"}
INJECTIONS_KEEP = {"DOI", "Inj_Mode", "Side", "Incubated", "Virus_Full"}
COLUMNS_TO_PICK = ("Filename", "OBJ", "EMI", "FRAMES", "SLICE", "AT", "ABF_NUMBER")

# Set up rich console
console = Console()


class CtrlDorQuery:
    def __init__(self, view: ViewDorQuery) -> None:
        self.view = view
        self.exp_info_db = QSqlDatabase.addDatabase("QSQLITE", "access_exp_info")
        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "access_rec_data")
        self.exp_info_db.setDatabaseName(str(EXP_DB_PATH))
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))

        self.exp_info_db.open()
        self.rec_data_db.open()

        self.clear_pick_list()
        self.load_dors()
        self.connect_signals()

    def load_dors(self) -> None:
        conn = sqlite3.connect(EXP_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT DOR FROM BASIC_INFO ORDER BY DOR")
        self.view.lw_dor.addItems([row[0] for row in cursor.fetchall()])
        conn.close()

    def connect_signals(self) -> None:
        self.view.lw_dor.currentTextChanged.connect(self.load_animals)
        self.view.lw_dor.currentTextChanged.connect(self.load_rec_summary)
        self.view.btn_reset_all_filters.clicked.connect(self.reset_all_filters)
        for col in self.view.filter_columns:
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.lw.itemChanged.connect(self.apply_filters)

        self.view.dd_shown_cols.lw.itemChanged.connect(self.toggle_shown_columns)
        self.view.btn_pick_selected.clicked.connect(self.pick_selected)
        self.view.btn_open_pick_list.clicked.connect(self.open_pick_list)

    def load_animals(self, dor: str) -> None:
        # Clear injections table when switching DOR
        self.view.tv_injections.setModel(None)

        # Display via QSqlTableModel, hide unwanted columns
        model = QSqlTableModel(db=self.exp_info_db)
        model.setTable("BASIC_INFO")
        model.setFilter(f"DOR = '{dor}'")
        model.select()
        self.view.tv_animals.setModel(model)
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) not in ANIMALS_KEEP:
                self.view.tv_animals.hideColumn(col)
        self.view.tv_animals.selectionModel().selectionChanged.connect(self.load_injections)

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

    def load_injections(self) -> None:
        selected = self.view.tv_animals.selectionModel().selectedRows()
        if not selected:
            return

        animal_id = self.view.tv_animals.model().record(selected[0].row()).value("Animal_ID")
        # Display via QSqlTableModel, hide unwanted columns
        model = QSqlTableModel(db=self.exp_info_db)
        model.setTable("INJECTION_HISTORY")
        model.setFilter(f"Animal_ID = '{animal_id}'")
        model.select()
        self.view.tv_injections.setModel(model)
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) not in INJECTIONS_KEEP:
                self.view.tv_injections.hideColumn(col)

    def check_pick_list(self, df_selected: pd.DataFrame) -> None:
        path = MODELS_DIR / "pick_list.json"
        df_saved = pd.DataFrame(columns=COLUMNS_TO_PICK, dtype=str)
        if path.exists():
            with Path.open(path) as f:
                df_saved = pd.read_json(f, orient="records", dtype=str).fillna("")

        if not df_saved.empty:
            merged = df_selected.merge(df_saved, how="left", indicator=True)
            new_rows = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
            df_selected = pd.concat([df_saved, new_rows], ignore_index=True)

        self.df_pick_list = df_selected.sort_values(by="Filename", ignore_index=True)
        self.df_pick_list.to_json(path, orient="records", indent=4)

        console.print(
            "\n[bold green]Pick List (Latest):[/bold green]\n",
            tabulate(self.df_pick_list, headers="keys", showindex=False, tablefmt="pretty"),
        )

    def pick_selected(self) -> None:
        if self.view.tv_rec_summary.model() is None:
            console.print("[bold red]No table to pick from![/bold red]")
            return

        selected = self.view.tv_rec_summary.selectionModel().selectedRows()
        if not selected:
            console.print("[bold red]No row selected![/bold red]")
            return

        # Create a table of dataframe for selected rows
        selected_row_data = []
        for idx in sorted(selected):
            record = self.view.tv_rec_summary.model().record(idx.row())
            selected_row_data.append(
                {col: (str(record.value(col)) if record.value(col) is not None else "") for col in COLUMNS_TO_PICK}
            )

        df_selected = pd.DataFrame(selected_row_data, dtype=str)
        console.print(
            "\n[bold green]Selected Rows:[/bold green]\n",
            tabulate(df_selected, headers="keys", showindex=False, tablefmt="pretty"),
        )

        self.check_pick_list(df_selected)

    def clear_pick_list(self) -> None:
        self.df_pick_list = pd.DataFrame(dtype=str)
        self.df_pick_list.to_json(MODELS_DIR / "pick_list.json", orient="records", indent=4)

    def open_pick_list(self) -> None:
        self.dlg_pick_list = DialogPickList()
        self.dlg_pick_list.show()
