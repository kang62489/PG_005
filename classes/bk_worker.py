## Modules
# Third-party imports
from PySide6.QtCore import QThread, Signal


class BackgroundWorker(QThread):
    finished = Signal()
    proc_msgs = Signal(str)

    def __init__(self, fn, *args, use_emitter: bool = False, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._use_emitter = use_emitter

    def run(self) -> None:
        if self._use_emitter:
            self._fn(*self._args, **self._kwargs, emitter=self.proc_msgs.emit)
        else:
            self._fn(*self._args, **self._kwargs)
        self.finished.emit()
