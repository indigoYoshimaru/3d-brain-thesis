import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon


class FileDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Open file'
        self.left = 20
        self.top = 20
        self.width = 640
        self.height = 480

    def get_nii_dir(self):
        """
        Read .nii.gz file directory 
        """
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "OpenFile", "", "All Files (*);;NIfTI Files(*.nii.gz)", options=options)
        print(file_name)
        return file_name
