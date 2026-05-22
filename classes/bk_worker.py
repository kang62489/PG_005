## Modules
# Third-party imports
from PySide6.QtCore import QThread, Signal

class BackgroundWorker(QThread):
    finished = Signal()  # Signal to emit when the task is finished, passing the result

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        self._fn(*self._args, **self._kwargs)
        self.finished.emit()