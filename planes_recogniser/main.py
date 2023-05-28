import sys
import os
import re
from typing import List

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
)
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.uic import loadUi
from PIL import Image

from mainwindow import Ui_MainWindow

from network_controllers import *
from planes_detector import (
    IDetectionNetworkController, YoloDetectionNetworkController, YoloDetectionNetworkControllerM, BBox
)

# TODO add errors check
# TODO check image open cancel
# TODO add conf tresh in ui

class BBoxWithClass(BBox):
    def __init__(self, bbox: BBox, planeClass):
        super().__init__(bbox.x, bbox.y, bbox.width, bbox.height)
        self.planeClass = planeClass


class Window(QMainWindow, Ui_MainWindow):
    m_networkController = NetworkController(100, 1e-3, True)
    m_planesDetector = YoloDetectionNetworkController()
    m_imageClassificationPath = ""
    m_imageDetectionPath = ""

    m_algsControllersMap = {
        "Adam": NetworkController,
        "SGD": SGDNetworkController,
        "RMSProps": RPMSPropsNetworkController,
        "SortingLoss": SortNetworkController
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
        self.loadImageButton.clicked.connect(self._OnLoadImageClassification)
        self.trainModelButton.clicked.connect(self._OnTrainModel)
        self.testImageButton.clicked.connect(self._OnTestImage)

        self.loadDetectionButton.clicked.connect(self._OnLoadDetector)
        self.detectionLoadImageButton.clicked.connect(self._OnLoadImageDetection)
        self.detectPlanesButton.clicked.connect(self._OnTestDetection)

    def _SetupModelsMenu(self):
        models = self.m_networkController.GetAllModels()
        for model in models:
            self.modelsComboBox.addItem(model)
        
        detectorModels = self.m_planesDetector.GetAllModels()
        for model in detectorModels:
            self.detectionModelsComboBox.addItem(model)

    def _OnLoadModel(self):
        modelName = self.modelsComboBox.currentText()
        self.m_networkController.LoadModel(modelName)
    
    def _OnLoadDetector(self):
        modelName = self.detectionModelsComboBox.currentText()
        self.m_planesDetector.LoadModel(modelName)

    def _OnLoadImageClassification(self):
        self.m_imageClassificationPath = QFileDialog.getOpenFileName(
            self, "Open Image", "../planes_dataset/Parsed", "Image Files (*.png *.jpg *.bmp)")[0]

        if len(self.m_imageClassificationPath) == 0:
            return

        m = re.search(r"(A\d+)", self.m_imageClassificationPath)
        self.ImageLabel.setPixmap(QtGui.QPixmap(self.m_imageClassificationPath))
        self.ImageClassLabel.setText(f"Самолет модели {m.group(0)}")

    def _OnLoadImageDetection(self):
        self.m_imageDetectionPath = QFileDialog.getOpenFileName(
            self, "Open Image", "../planes_dataset/JPEGImages", "Image Files (*.png *.jpg *.bmp)")[0]

        if len(self.m_imageDetectionPath) == 0:
            return

        self.detectPlanesImageLabel.setPixmap(QtGui.QPixmap(self.m_imageDetectionPath))

    def _OnTrainModel(self):
        modelName = self.lineEdit.text()
        lr = self.LearningRateSpinBox.value()
        batchSize = self.BatchSizeSpinBox.value()
        epochs = self.epochsSpinBox.value()
        alg = self.learningAlgComboBox.currentText()
        needAug = self.AugRadioButton.isChecked()

        networkController = self.m_algsControllersMap[alg](batchSize, lr, needAug)

        networkController.TrainNetwork(epochs)

        self.m_networkController.SaveModel(modelName)

    def _OnTestImage(self):
        planeModel = self.m_networkController.GetResult(self.m_imageClassificationPath).long() + 1
        self.ImageDetectedClassLabel.setText(f"Распознан самолет А{planeModel}")
    
    def _OnTestDetection(self):
        bboxes = self.m_planesDetector.GetBBoxes(self.m_imageDetectionPath)
        bboxesWithClasses = []

        TMP_IMAGE_PATH = "tmp.png"
        for bbox in bboxes:
            self._CropImage(self.m_imageDetectionPath, bbox, outputSize=96, outputPath=TMP_IMAGE_PATH)
            planeModel = self.m_networkController.GetResult(TMP_IMAGE_PATH).long() + 1
            bboxesWithClasses.append(BBoxWithClass(bbox, planeModel))


        self._PlotBoxes(bboxesWithClasses)
    
    def _CropImage(self, imagePath, bbox, outputSize=96, outputPath="tmp.png"):
        image: Image.Image = Image.open(imagePath)
        xmin, xmax = bbox.x - bbox.width // 2, bbox.x + bbox.width // 2
        ymin, ymax = bbox.y - bbox.height // 2, bbox.y + bbox.height // 2

        xmin, xmax = self._ResizeInterval(outputSize, xmin, xmax, image.width)
        ymin, ymax = self._ResizeInterval(outputSize, ymin, ymax, image.height)

        cropedPlane = image.crop((xmin, ymin, xmax, ymax))
        cropedPlane.save(outputPath)

    
    def _ResizeInterval(self, outputSize, start, end, maxSize):
        diff = outputSize - (end - start)
        if start - diff / 2 < 0:
            start, end = 0, outputSize
        elif end + diff / 2 > maxSize:
            start, end = maxSize - outputSize, maxSize
        else:
            start = start - diff / 2
            end = start + outputSize
    
        return start, end
    
    def _PlotBoxes(self, bboxes: List[BBoxWithClass]):
        pixmap = QtGui.QPixmap(self.m_imageDetectionPath)
        painter = QtGui.QPainter(pixmap)

        pen = QtGui.QPen(QtCore.Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)

        for box in bboxes:
            painter.drawRect(box.x - box.width // 2, box.y - box.height // 2, box.width, box.height)
            painter.drawText(box.x - box.width // 2, box.y - box.height // 2, f"A{box.planeClass}")
        
        painter.end()
        
        self.detectPlanesImageLabel.setPixmap(pixmap)



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
    if "-no-ui" in sys.argv:
        controller = AugmentationNetworkController(50, 1e-5)
        if "-train" in sys.argv:
            controller.TrainNetwork(20, "augadammodeltmp.pt")
    else:
        app = QApplication(sys.argv)
        win = Window()
        win.show()
        sys.exit(app.exec())
