"""
model_from_dataframe.py  --  Qt table model backed by a polars DataFrame.

Rows with a missing_values cell are bold red, else rows with a muted_values cell are gray
(e.g. Spike-aligned: missing=("No",), muted=("N/A",)). Only the PROC / MODE columns are editable.
"""

## Modules
# Third-party imports
import polars as pl
from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtGui import QColor, QFont


class ModelFromDataFrame(QAbstractTableModel):
    """Rows with any cell in missing_values -> bold red; else any cell in muted_values -> gray."""

    def __init__(self, df: pl.DataFrame, missing_values: tuple[str, ...] = (),
                 muted_values: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._data = df if df is not None else pl.DataFrame()
        self._missing_values = missing_values
        self._muted_values = muted_values
        self._update_row_styles()

    def _rows_with(self, values: tuple[str, ...]) -> set[int]:
        """Row indices where any string cell is one of values."""
        str_cols = [name for name, dtype in self._data.schema.items() if dtype == pl.String]
        if not values or self._data.is_empty() or not str_cols:
            return set()
        has_value = pl.any_horizontal(pl.col(str_cols).is_in(values))
        return set(self._data.with_row_index("_row").filter(has_value)["_row"].to_list())

    def _update_row_styles(self) -> None:
        self._missing_rows = self._rows_with(self._missing_values)
        self._muted_rows = self._rows_with(self._muted_values) - self._missing_rows

    def data(self, index, role) -> str | QColor | QFont | None:
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row(), index.column()])
        if index.row() in self._missing_rows:
            if role == Qt.ItemDataRole.ForegroundRole:
                return QColor("red")
            if role == Qt.ItemDataRole.FontRole:
                font = QFont()
                font.setBold(True)
                return font
        if index.row() in self._muted_rows and role == Qt.ItemDataRole.ForegroundRole:
            return QColor("gray")
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
        self._update_row_styles()
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
