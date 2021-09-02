# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   2 Sep., 2021
# Implementation of MIDeepSeg for interactive medical image segmentation and annotation.
# Reference:
#     X. Luo and G. Wang et al. MIDeepSeg: Minimally interactive segmentation of unseen objects
#     from medical images using deep learning. Medical Image Analysis, 2021. DOI:https://doi.org/10.1016/j.media.2021.102102.

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
