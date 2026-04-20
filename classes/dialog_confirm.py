from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class DialogConfirm(QDialog):
    def __init__(self, title="Dialog", msg="Question", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.lbl_message = QLabel(msg)

        self.buttons = QDialogButtonBox.Yes | QDialogButtonBox.No
        self.buttonBox = QDialogButtonBox(self.buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.lo_main = QVBoxLayout()
        self.lo_main.addWidget(self.lbl_message)
        self.lo_main.addWidget(self.buttonBox, 0, Qt.AlignCenter)
        self.setLayout(self.lo_main)
