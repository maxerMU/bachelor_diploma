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
        MainWindow.resize(1421, 992)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.ImageDetectedClassLabel = QtWidgets.QLabel(self.groupBox_2)
        self.ImageDetectedClassLabel.setText("")
        self.ImageDetectedClassLabel.setObjectName("ImageDetectedClassLabel")
        self.gridLayout_3.addWidget(self.ImageDetectedClassLabel, 5, 2, 1, 1)
        self.loadImageButton = QtWidgets.QPushButton(self.groupBox_2)
        self.loadImageButton.setObjectName("loadImageButton")
        self.gridLayout_3.addWidget(self.loadImageButton, 1, 0, 1, 1)
        self.ImageLabel = QtWidgets.QLabel(self.groupBox_2)
        self.ImageLabel.setMinimumSize(QtCore.QSize(96, 96))
        self.ImageLabel.setText("")
        self.ImageLabel.setObjectName("ImageLabel")
        self.gridLayout_3.addWidget(self.ImageLabel, 1, 2, 1, 1)
        self.modelsComboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.modelsComboBox.setObjectName("modelsComboBox")
        self.gridLayout_3.addWidget(self.modelsComboBox, 0, 2, 1, 1)
        self.loadModelButton = QtWidgets.QPushButton(self.groupBox_2)
        self.loadModelButton.setObjectName("loadModelButton")
        self.gridLayout_3.addWidget(self.loadModelButton, 0, 0, 1, 1)
        self.testImageButton = QtWidgets.QPushButton(self.groupBox_2)
        self.testImageButton.setObjectName("testImageButton")
        self.gridLayout_3.addWidget(self.testImageButton, 5, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 3, 2, 1, 1)
        self.ImageClassLabel = QtWidgets.QLabel(self.groupBox_2)
        self.ImageClassLabel.setText("")
        self.ImageClassLabel.setObjectName("ImageClassLabel")
        self.gridLayout_3.addWidget(self.ImageClassLabel, 2, 2, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.detectPlanesImageLabel = QtWidgets.QLabel(self.groupBox_3)
        self.detectPlanesImageLabel.setMinimumSize(QtCore.QSize(800, 800))
        self.detectPlanesImageLabel.setText("")
        self.detectPlanesImageLabel.setObjectName("detectPlanesImageLabel")
        self.gridLayout_4.addWidget(self.detectPlanesImageLabel, 1, 1, 1, 1)
        self.loadDetectionButton = QtWidgets.QPushButton(self.groupBox_3)
        self.loadDetectionButton.setObjectName("loadDetectionButton")
        self.gridLayout_4.addWidget(self.loadDetectionButton, 0, 0, 1, 1)
        self.detectionModelsComboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.detectionModelsComboBox.setObjectName("detectionModelsComboBox")
        self.gridLayout_4.addWidget(self.detectionModelsComboBox, 0, 1, 1, 1)
        self.detectPlanesButton = QtWidgets.QPushButton(self.groupBox_3)
        self.detectPlanesButton.setObjectName("detectPlanesButton")
        self.gridLayout_4.addWidget(self.detectPlanesButton, 3, 0, 1, 2)
        self.detectionLoadImageButton = QtWidgets.QPushButton(self.groupBox_3)
        self.detectionLoadImageButton.setObjectName("detectionLoadImageButton")
        self.gridLayout_4.addWidget(self.detectionLoadImageButton, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem1, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox_3, 0, 1, 2, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableWidget = QtWidgets.QTableWidget(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(350)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(100)
        self.tableWidget.verticalHeader().setCascadingSectionResizes(True)
        self.verticalLayout.addWidget(self.tableWidget)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Распознавание летательной техники"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Классификация"))
        self.loadImageButton.setText(_translate("MainWindow", "Загрузить изображение"))
        self.loadModelButton.setText(_translate("MainWindow", "Загрузить модель"))
        self.testImageButton.setText(_translate("MainWindow", "Определить класс самолета"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Распознавание"))
        self.loadDetectionButton.setText(_translate("MainWindow", "Загрузить модель"))
        self.detectPlanesButton.setText(_translate("MainWindow", "Распознать самолеты"))
        self.detectionLoadImageButton.setText(_translate("MainWindow", "Загрузить изображение"))
        self.groupBox.setTitle(_translate("MainWindow", "Распознанные самолеты"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Модель самолета"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
