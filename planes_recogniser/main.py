import sys
import os
import re

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
)
from PyQt5 import QtGui
from PyQt5.uic import loadUi

from mainwindow import Ui_MainWindow

from network_controllers import *

# TODO add errors check


class Window(QMainWindow, Ui_MainWindow):
    # TODO check if controller is loaded
    m_networkController = NetworkController(100, 1e-3, True)
    # m_networkController = None
    m_imagePath = ""

    m_algsControllersMap = {
        "Adam": NetworkController,
        "SGD": SGDNetworkController,
        "RMSProps": RPMSPropsNetworkController
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._ConnectSignalsSlots()
        self._SetupTrainAlgsMenu()
        self._SetupModelsMenu()

    def _SetupTrainAlgsMenu(self):
        for alg in self.m_algsControllersMap:
            self.learningAlgComboBox.addItem(alg)
        
        # self.m_algsControllersMap = self.m_algsControllersMap["Adam"](10, 1e-3)

    def _ConnectSignalsSlots(self):
        self.loadModelButton.clicked.connect(self._OnLoadModel)
        self.loadImageButton.clicked.connect(self._OnLoadImage)
        self.trainModelButton.clicked.connect(self._OnTrainModel)
        self.testImageButton.clicked.connect(self._OnTestImage)

    def _SetupModelsMenu(self):
        models = self.m_networkController.GetAllModels()
        for model in models:
            self.modelsComboBox.addItem(model)

    def _OnLoadModel(self):
        modelName = self.modelsComboBox.currentText()
        self.m_networkController.LoadModel(modelName)

    def _OnLoadImage(self):
        self.m_imagePath = QFileDialog.getOpenFileName(
            self, "Open Image", "../planes_dataset/Parsed", "Image Files (*.png *.jpg *.bmp)")[0]

        if len(self.m_imagePath) == 0:
            return

        m = re.search(r"(A\d+)", self.m_imagePath)
        self.ImageLabel.setPixmap(QtGui.QPixmap(self.m_imagePath))
        self.ImageClassLabel.setText(f"Самолет модели {m.group(0)}")

    def _OnTrainModel(self):
        modelName = self.lineEdit.text()
        lr = self.LearningRateSpinBox.value()
        batchSize = self.BatchSizeSpinBox.value()
        epochs = self.epochsSpinBox.value()
        alg = self.learningAlgComboBox.currentText()
        needAug = self.AugRadioButton.isEnabled()

        networkController = self.m_algsControllersMap[alg](batchSize, lr, needAug)

        networkController.TrainNetwork(epochs)

        self.m_networkController.SaveModel(modelName)

    def _OnTestImage(self):
        planeModel = self.m_networkController.GetResult(
            self.m_imagePath).long() + 1
        self.ImageDetectedClassLabel.setText(
            f"Распознан самолет А{planeModel}")


def train(networkController: INetworkController):
    networkController.TrainNetwork(15)
    networkController.SaveModel()


def test(networkController: INetworkController):
    networkController.LoadModel()
    networkController.LogTraining()


if __name__ == "__main__":
    # controller = AugmentationNetworkController(100, 1e-3)
    # train(controller)
    # test(controller)
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
