import argparse
import sys
import os
import PyQt5.QtWidgets as QtWidgets
from utils.config import *


if __name__ == "__main__":
    sys_dirs = ['C:\\users\\seito-nonroot\\appdata\\roaming\\python\\python37\\site-packages',
                'D:\\2122sem1\pre_thesis\\brain-reconstruction\\code\\3d-brain-thesis\\']
    for sdir in sys_dirs:
        sys.path.append(os.path.dirname(sdir))
    print(sys.path)
    from model_controller import segtran_inference
    from visualizer.main_window import MainWindow

    app = QtWidgets.QApplication([])
    args, net = segtran_inference.load_model(MODEL_PATH)
    window = MainWindow(app, args, net)
    sys.exit(app.exec_())
