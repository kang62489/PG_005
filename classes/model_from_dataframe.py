# Modules
# Third-party imports
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class ModelFromDataFrame(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._data = df if df is not None else pd.DataFrame()

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> None | str:
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def rowCount(self, _parent: QModelIndex | None = None) -> int:
        return self._data.shape[0]

    def columnCount(self, _parent: QModelIndex | None = None) -> int:
        return self._data.shape[1]

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole) -> str | None:
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._data.columns[section])
        return str(self._data.index[section])
