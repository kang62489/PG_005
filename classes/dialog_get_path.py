from PySide6.QtWidgets import QFileDialog

class DialogGetPath(QFileDialog):
    def __init__(self, title="Please select the folder contains .rec files", init_dir=''):
        super().__init__()
        self.setFileMode(QFileDialog.Directory)
        self.setOption(QFileDialog.ShowDirsOnly)
        self.setDirectory(init_dir)

        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setWindowTitle(title)
        
    def get_path(self):
        if self.exec():
            return self.selectedFiles()[0]
        else:
            return ""
        