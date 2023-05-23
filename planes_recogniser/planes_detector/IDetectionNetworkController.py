from abc import ABC, abstractmethod

class BBox:
    x = 0
    y = 0
    width = 0
    height = 0

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

class IDetectionNetworkController(ABC):
    @abstractmethod
    def LoadModel(self, modelPath):
        pass

    @abstractmethod
    def GetAllModels(self):
        pass

    @abstractmethod
    def GetBBoxes(self, imagePath):
        pass