import argparse
import sys
import os
from visualizer.main_window import MainWindow
import PyQt5.QtWidgets as QtWidgets


if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    window = MainWindow(app)
    sys.exit(app.exec_())
