import random
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from controler import Controler


class MIDeepSeg(QWidget):

    def __init__(self):
        super().__init__()

        self.graph_maker = Controler()
        self.seed_type = 1  # annotation type
        self.all_datasets = []

        self.initUI()

    def initUI(self):
        self.a = QApplication(sys.argv)

        self.window = QMainWindow()
        # Setup file menu
        self.window.setWindowTitle('MIDeepSeg')
        mainMenu = self.window.menuBar()
        fileMenu = mainMenu.addMenu('&File')

        openButton = QAction(QIcon('exit24.png'), 'Open Image', self.window)
        openButton.setShortcut('Ctrl+O')
        openButton.setStatusTip('Open a file for segmenting.')
        openButton.triggered.connect(self.on_open)
        fileMenu.addAction(openButton)

        saveButton = QAction(QIcon('exit24.png'), 'Save Image', self.window)
        saveButton.setShortcut('Ctrl+S')
        saveButton.setStatusTip('Save file to disk.')
        saveButton.triggered.connect(self.on_save)
        fileMenu.addAction(saveButton)

        closeButton = QAction(QIcon('exit24.png'), 'Exit', self.window)
        closeButton.setShortcut('Ctrl+Q')
        closeButton.setStatusTip('Exit application')
        closeButton.triggered.connect(self.on_close)
        fileMenu.addAction(closeButton)

        mainWidget = QWidget()

        annotationButton = QPushButton("Load Image")
        annotationButton.setStyleSheet("background-color:white")
        annotationButton.clicked.connect(self.on_open)

        segmentButton = QPushButton("Segment")
        segmentButton.setStyleSheet("background-color:white")
        segmentButton.clicked.connect(self.on_segment)

        refinementButton = QPushButton("Refinement")
        refinementButton.setStyleSheet("background-color:white")
        refinementButton.clicked.connect(self.on_refinement)

        CleanButton = QPushButton("Clear all seeds")
        CleanButton.setStyleSheet("background-color:white")
        CleanButton.clicked.connect(self.on_clean)

        NextButton = QPushButton("Save segmentation")
        NextButton.setStyleSheet("background-color:white")
        NextButton.clicked.connect(self.on_save)

        StateLine = QLabel()
        StateLine.setText("Clicks as user input.")
        palette = QPalette()
        palette.setColor(StateLine.foregroundRole(), Qt.blue)
        StateLine.setPalette(palette)

        MethodLine = QLabel()
        MethodLine.setText("Segmentation.")
        mpalette = QPalette()
        mpalette.setColor(MethodLine.foregroundRole(), Qt.blue)
        MethodLine.setPalette(mpalette)

        SaveLine = QLabel()
        SaveLine.setText("Clean or Save.")
        spalette = QPalette()
        spalette.setColor(SaveLine.foregroundRole(), Qt.blue)
        SaveLine.setPalette(spalette)

        hbox = QVBoxLayout()
        hbox.addWidget(StateLine)
        hbox.addWidget(annotationButton)
        hbox.addWidget(MethodLine)
        hbox.addWidget(segmentButton)
        hbox.addWidget(refinementButton)
        hbox.addWidget(SaveLine)
        hbox.addWidget(CleanButton)
        hbox.addWidget(NextButton)
        hbox.addStretch()

        tipsFont = StateLine.font()
        tipsFont.setPointSize(10)
        StateLine.setFixedHeight(30)
        StateLine.setWordWrap(True)
        StateLine.setFont(tipsFont)
        MethodLine.setFixedHeight(30)
        MethodLine.setWordWrap(True)
        MethodLine.setFont(tipsFont)
        SaveLine.setFixedHeight(30)
        SaveLine.setWordWrap(True)
        SaveLine.setFont(tipsFont)

        self.seedLabel = QLabel()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseMoveEvent = self.mouse_drag

        imagebox = QHBoxLayout()
        imagebox.addWidget(self.seedLabel)

        vbox = QHBoxLayout()

        vbox.addLayout(imagebox)
        vbox.addLayout(hbox)

        mainWidget.setLayout(vbox)

        self.window.setCentralWidget(mainWidget)
        self.window.show()

    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def mouse_down(self, event):
        if event.button() == Qt.LeftButton:
            self.seed_type = 2
        elif event.button() == Qt.RightButton:
            self.seed_type = 3
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    def mouse_drag(self, event):
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    @pyqtSlot()
    def on_open(self):
        f = QFileDialog.getOpenFileName()
        if f[0] is not None and f[0] != "":
            f = f[0]
            self.graph_maker.load_image(str(f))
            self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        else:
            pass

    @pyqtSlot()
    def on_save(self):
        f = QFileDialog.getSaveFileName()
        print('Saving')
        if f is not None and f != "":
            f = f[0]
            self.graph_maker.save_image(f)
        else:
            pass

    @pyqtSlot()
    def on_close(self):
        print('Closing')
        self.window.close()

    @pyqtSlot()
    def on_segment(self):
        self.graph_maker.extreme_segmentation()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.extreme_segmentation))))

    @pyqtSlot()
    def on_clean(self):
        self.graph_maker.clear_seeds()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.clear_seeds))))

    @pyqtSlot()
    def on_refinement(self):
        self.graph_maker.refined_seg()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.refined_seg))))
