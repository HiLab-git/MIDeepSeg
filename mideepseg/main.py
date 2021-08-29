import os
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from gui import MIDeepSeg


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MIDeepSeg()
    sys.exit(app.exec_())
