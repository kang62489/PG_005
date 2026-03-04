## Modules
# Standard library imports
import sqlite3

# Third-party imports
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel

# Local application imports
from utils.params import EXP_DB_PATH
from views import ViewDorQuery

ANIMALS_KEEP = {"Animal_ID", "DOB", "Ages", "Genotype", "SEX"}
INJECTIONS_KEEP = {"DOI", "Inj_Mode", "Side", "Incubated", "Virus_Full"}


class CtrlDorQuery:
    def __init__(self, view: ViewDorQuery) -> None:
        self.view = view
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(str(EXP_DB_PATH))
        self.db.open()
        self.df_animals: pd.DataFrame = pd.DataFrame()
        self.df_injections: pd.DataFrame = pd.DataFrame()
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

    def load_animals(self, dor: str) -> None:
        # Clear injections table when switching DOR
        self.view.tv_injections.setModel(None)
        self.df_injections = pd.DataFrame()

        # Store full data as DataFrame
        conn = sqlite3.connect(EXP_DB_PATH)
        self.df_animals = pd.read_sql("SELECT * FROM BASIC_INFO WHERE DOR = ?", conn, params=(dor,))
        conn.close()

        # Display via QSqlTableModel, hide unwanted columns
        model = QSqlTableModel(db=self.db)
        model.setTable("BASIC_INFO")
        model.setFilter(f"DOR = '{dor}'")
        model.select()
        self.view.tv_animals.setModel(model)
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) not in ANIMALS_KEEP:
                self.view.tv_animals.hideColumn(col)
        self.view.tv_animals.selectionModel().selectionChanged.connect(self.load_injections)

    def load_injections(self) -> None:
        selected = self.view.tv_animals.selectionModel().selectedRows()
        if not selected:
            return
        # Read animal_id from DataFrame (not from Qt model)
        animal_id = self.df_animals.iloc[selected[0].row()]["Animal_ID"]

        # Store full data as DataFrame
        conn = sqlite3.connect(EXP_DB_PATH)
        self.df_injections = pd.read_sql(
            "SELECT * FROM INJECTION_HISTORY WHERE Animal_ID = ?", conn, params=(animal_id,)
        )
        conn.close()

        # Display via QSqlTableModel, hide unwanted columns
        model = QSqlTableModel(db=self.db)
        model.setTable("INJECTION_HISTORY")
        model.setFilter(f"Animal_ID = '{animal_id}'")
        model.select()
        self.view.tv_injections.setModel(model)
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) not in INJECTIONS_KEEP:
                self.view.tv_injections.hideColumn(col)
        self.view.tv_injections.resizeColumnsToContents()
