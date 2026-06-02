from PySide6.QtWidgets import QFileDialog


class DialogGetPath(QFileDialog):
    def __init__(self, title="Please select the folder contains .rec files", init_dir=""):
        super().__init__()
        self.setFileMode(QFileDialog.Directory)
        self.setOption(QFileDialog.ShowDirsOnly)
        self.setDirectory(init_dir)

        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setWindowTitle(title)

    def get_path(self):
        if self.exec():
            return self.selectedFiles()[0]
        return ""

class DialogGetFile(QFileDialog):
    def __init__(self, title="Please select a file", init_dir=""):
        super().__init__()
        self._init_dir = init_dir
        self.setFileMode(QFileDialog.ExistingFile)

        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setWindowTitle(title)

    def get_pick_list(self):
        self.setNameFilters(["Text Files (*.txt)",])
        self.setDirectory(self._init_dir)
        if self.exec():
            return self.selectedFiles()[0]
        return ""

    def get_proc_list(self):
        self.setNameFilter("Processing List (proc_*.txt)")
        self.setDirectory(self._init_dir)
        if self.exec():
            return self.selectedFiles()[0]
        return ""
