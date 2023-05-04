# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(589, 241)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.modelsComboBox = QtWidgets.QComboBox(self.groupBox)
        self.modelsComboBox.setObjectName("modelsComboBox")
        self.gridLayout_2.addWidget(self.modelsComboBox, 0, 1, 1, 1)
        self.trainModelButton = QtWidgets.QPushButton(self.groupBox)
        self.trainModelButton.setObjectName("trainModelButton")
        self.gridLayout_2.addWidget(self.trainModelButton, 1, 0, 1, 1)
        self.testImageButton = QtWidgets.QPushButton(self.groupBox)
        self.testImageButton.setObjectName("testImageButton")
        self.gridLayout_2.addWidget(self.testImageButton, 3, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.loadImageButton = QtWidgets.QPushButton(self.groupBox)
        self.loadImageButton.setObjectName("loadImageButton")
        self.gridLayout_2.addWidget(self.loadImageButton, 2, 0, 1, 1)
        self.loadModelButton = QtWidgets.QPushButton(self.groupBox)
        self.loadModelButton.setObjectName("loadModelButton")
        self.gridLayout_2.addWidget(self.loadModelButton, 0, 0, 1, 1)
        self.ImageClassLabel = QtWidgets.QLabel(self.groupBox)
        self.ImageClassLabel.setText("")
        self.ImageClassLabel.setObjectName("ImageClassLabel")
        self.gridLayout_2.addWidget(self.ImageClassLabel, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        self.ImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImageLabel.setMinimumSize(QtCore.QSize(96, 96))
        self.ImageLabel.setText("")
        self.ImageLabel.setObjectName("ImageLabel")
        self.gridLayout.addWidget(self.ImageLabel, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 589, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.trainModelButton.setText(_translate("MainWindow", "Обучить модель"))
        self.testImageButton.setText(_translate("MainWindow", "Определить класс самолета"))
        self.loadImageButton.setText(_translate("MainWindow", "Загрузить изображение"))
        self.loadModelButton.setText(_translate("MainWindow", "Загрузить модель"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
