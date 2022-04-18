import argparse
import sys
import os
import PyQt5.QtWidgets as QtWidgets
from utils.config import *


if __name__ == "__main__":
    sys_dirs = [
        './3d-brain-thesis']
    for sdir in sys_dirs:
        sys.path.append(os.path.dirname(sdir))
    print(sys.path)
    from model_controller import inference, model_loader
    from visualizer.main_window import MainWindow

    app = QtWidgets.QApplication([])

    args, net = model_loader.load_unet_model(MODEL_PATH)
    print(net)
    window = MainWindow(app, args, net)
    sys.exit(app.exec_())
