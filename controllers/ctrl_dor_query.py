## Modules
# Standard library imports
import sqlite3

# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel

# Local application imports
from utils import EXP_DB_PATH, REC_DB_PATH
from views import ViewDorQuery

ANIMALS_KEEP = {"Animal_ID", "DOB", "Ages", "Genotype", "Sex"}
INJECTIONS_KEEP = {"DOI", "Inj_Mode", "Side", "Incubated", "Virus_Full"}
REC_SUMMARY_KEEP = {}


class CtrlDorQuery:
    def __init__(self, view: ViewDorQuery) -> None:
        self.view = view
        self.exp_info_db = QSqlDatabase.addDatabase("QSQLITE", "access_exp_info")
        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "access_rec_data")
        self.exp_info_db.setDatabaseName(str(EXP_DB_PATH))
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))

        self.exp_info_db.open()
        self.rec_data_db.open()

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
        for col in self.view.filter_columns:
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.clear_items()

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

        for col in self.view.filter_columns:
            col_list = [model.record(row).value(col) for row in range(model.rowCount())]
            unique_col = set(col_list)
            dropdown = getattr(self.view, f"dd_{col}")
            dropdown.lw.blockSignals(True)
            dropdown.add_items(unique_col)
            dropdown.lw.blockSignals(False)

        # for col in range(model.columnCount()):
        #     if model.headerData(col, Qt.Orientation.Horizontal) in REC_SUMMARY_KEEP:
        #         self.view.tv_rec_summary.hideColumn(col)

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
        self.apply_filters()

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
