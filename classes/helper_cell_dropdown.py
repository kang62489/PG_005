## Modules
# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QStyledItemDelegate, QWidget


class CellDropdownDelegate(QStyledItemDelegate):
    """A delegate that shows a QComboBox for editable cells."""

    def __init__(self, menu_options: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.menu_options = menu_options

    def createEditor(self, parent, _option, _index) -> QComboBox:
        editor = QComboBox(parent)
        editor.addItems(self.menu_options)
        # Connect the currentIndexChanged signal to commitData to ensure changes are saved immediately
        editor.activated.connect(lambda: self.commitData.emit(editor))
        return editor

    def setEditorData(self, editor: QComboBox, index) -> None:
        current = index.data(Qt.ItemDataRole.DisplayRole)
        idx = editor.findText(current)
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor: QComboBox, model, index) -> None:
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
