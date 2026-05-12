# Modules
# Third-party imports
import polars as pl
from PySide6.QtCore import QAbstractTableModel, Qt


class ModelFromDataFrame(QAbstractTableModel):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self._data = df if df is not None else pl.DataFrame()

    def data(self, index, role) -> None | str:
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row(), index.column()])
        return None

    def rowCount(self, _parent=None) -> int:
        return self._data.shape[0]

    def columnCount(self, _parent=None) -> int:
        return self._data.shape[1]

    def flags(self, index) -> Qt.ItemFlag:
        base = super().flags(index)
        col_name = self._data.columns[index.column()]
        if col_name == "PROC":
            return base | Qt.ItemFlag.ItemIsEditable
        if col_name == "MODE" and "PROC" in self._data.columns:
            proc_val = self._data[index.row(), self._data.columns.index("PROC")]
            if proc_val != "SKIP":
                return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index, value: str, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole:
            return False
        col_name = self._data.columns[index.column()]
        self._data = self._data.with_columns(
            pl.when(pl.int_range(pl.len()) == index.row())
            .then(pl.lit(value))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )
        self.dataChanged.emit(index, index, [role])
        return True

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole) -> str | None:
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._data.columns[section])
        return str(section)
