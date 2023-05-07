import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
)
from PyQt5 import QtGui
from PyQt5.uic import loadUi

from mainwindow import Ui_MainWindow

from network_controllers import NetworkController, AugmentationNetworkController, INetworkController

class Window(QMainWindow, Ui_MainWindow):
    # TODO move to ui
    controller = NetworkController(100, 1e-3)
    imagePath = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.ConnectSignalsSlots()
        self.SetupModelsMenu()
    
    def ConnectSignalsSlots(self):
        self.loadModelButton.clicked.connect(self._OnLoadModel)
        self.loadImageButton.clicked.connect(self._OnLoadImage)
        self.trainModelButton.clicked.connect(self._OnTrainModel)
        self.testImageButton.clicked.connect(self._OnTestImage)
    
    def SetupModelsMenu(self):
        models = self.controller.GetAllModels()
        for model in models:
            self.modelsComboBox.addItem(model)
    
    def _OnLoadModel(self):
        modelName = self.modelsComboBox.currentText()
        self.controller.LoadModel(modelName)

    def _OnLoadImage(self):
        self.imagePath = QFileDialog.getOpenFileName(self, "Open Image", "../planes_dataset/Parsed", "Image Files (*.png *.jpg *.bmp)")[0]

        if len(self.imagePath) == 0:
            return
        
        self.ImageLabel.setPixmap(QtGui.QPixmap(self.imagePath))


    def _OnTrainModel(self):
        modelName = self.lineEdit.text()
        self.controller.TrainNetwork(15)
        self.controller.SaveModel(modelName)
        

    def _OnTestImage(self):
        planeModel = self.controller.GetResult(self.imagePath).long() + 1
        self.ImageClassLabel.setText(f"Самолет А{planeModel}")

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
