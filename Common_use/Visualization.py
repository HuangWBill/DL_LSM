# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '0412.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QInputDialog,QFileDialog
from Common_use import main_use_GUI
import os

class Ui_MainWindo(object):
    def setupUi(self, MainWindo):
        MainWindo.setObjectName("MainWindo")
        MainWindo.resize(709, 576)
        MainWindo.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindo)
        self.centralwidget.setObjectName("centralwidget")
        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(30, 20, 251, 31))
        self.input_label.setStyleSheet("font: 14pt \"Times New Roman\";\n"
"border-right-color: rgb(161, 0, 0);")
        self.input_label.setObjectName("input_label")
        self.input_path = QtWidgets.QLabel(self.centralwidget)
        self.input_path.setGeometry(QtCore.QRect(30, 60, 491, 31))
        self.input_path.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.input_path.setText("")
        self.input_path.setObjectName("input_path")
        self.choose_button = QtWidgets.QPushButton(self.centralwidget)
        self.choose_button.setGeometry(QtCore.QRect(560, 60, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.choose_button.setFont(font)
        self.choose_button.setStyleSheet("font: 14pt \"Times New Roman\";\n"
"border-bottom-color: rgb(106, 106, 106);")
        self.choose_button.setObjectName("choose_button")
        self.output_box = QtWidgets.QLabel(self.centralwidget)
        self.output_box.setEnabled(True)
        self.output_box.setGeometry(QtCore.QRect(30, 180, 641, 341))
        self.output_box.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.output_box.setText("")
        self.output_box.setObjectName("output_box")
        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(30, 150, 201, 31))
        self.output_label.setStyleSheet("font: 14pt \"Times New Roman\";")
        self.output_label.setObjectName("output_label")
        self.roller = QtWidgets.QScrollBar(self.centralwidget)
        self.roller.setGeometry(QtCore.QRect(650, 180, 20, 341))
        self.roller.setOrientation(QtCore.Qt.Vertical)
        self.roller.setObjectName("roller")
        MainWindo.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindo)
        self.statusbar.setObjectName("statusbar")
        MainWindo.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindo)
        self.choose_button.clicked.connect(self.openFile)
        QtCore.QMetaObject.connectSlotsByName(MainWindo)

    def openFile(self):
        get_filename_path, ok = QFileDialog.getOpenFileName(self,"选取单个文件",os.getcwd(),"All Files (*);;Text Files (*.txt)")
        if ok:
            self.input_label.setText(str(get_filename_path))
        main_use_GUI.main_txt(str(get_filename_path))

    def retranslateUi(self, MainWindo):
        _translate = QtCore.QCoreApplication.translate
        MainWindo.setWindowTitle(_translate("MainWindo", "landslide susceptibility mapping"))
        self.input_label.setText(_translate("MainWindo", "Please input \'txt\' parameter file :"))
        self.choose_button.setText(_translate("MainWindo", "Choose file"))
        self.output_label.setText(_translate("MainWindo", "Running process output :"))

