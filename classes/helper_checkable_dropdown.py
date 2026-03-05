## Modules
# Third-party imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget


class CheckableDropdown(QPushButton):
    """A button that shows a checkable list popup when clicked."""

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{label} ▼", parent)
        self.popup = QFrame(self, Qt.WindowType.Popup)
        self.popup.setLayout(QVBoxLayout())
        self.lw = QListWidget()
        self.popup.layout().addWidget(self.lw)
        self.clicked.connect(self._show_popup)

    def _show_popup(self) -> None:
        pos = self.mapToGlobal(self.rect().bottomLeft())
        self.popup.move(pos)
        self.popup.show()

    def add_items(self, items: list[str]) -> None:
        for label in items:
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.lw.addItem(item)

    def checked_items(self) -> list[str]:
        return [
            self.lw.item(i).text()
            for i in range(self.lw.count())
            if self.lw.item(i).checkState() == Qt.CheckState.Checked
        ]

    def clear_items(self) -> None:
        self.lw.clear()
