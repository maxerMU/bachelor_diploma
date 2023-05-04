import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

from models.conv_network import PlanesNetwork
from models.simple_net import simplenet
from dataset_handler.dataset_handler import DataSetHandler

from network_controllers import INetworkController

class NetwrokController(INetworkController):
    _TRAINED_MODELS_PATH = "trained_models/"

    def __init__(self, batchSize, learningRate):
        # self.planesNetwork = PlanesNetwork(20)
        self.planesNetwork = simplenet(20)
        self.datasetHandler = DataSetHandler()

        self.batchSize = batchSize
        self.learningRate = learningRate

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.planesNetwork.parameters(), lr=self.learningRate)
        self.datasetLen = self.datasetHandler.TrainSize()
    
    def TrainPrepare(self):
        pass

    def TrainEpoch(self):
        order = np.random.permutation(self.datasetLen)
        for startIndex in range(0, self.datasetLen, self.batchSize):
            self.optimizer.zero_grad()

            xBatch, yBatch = self.datasetHandler.GetTrainingBatch(order[startIndex:startIndex+self.batchSize])

            preds = self.planesNetwork.forward(xBatch)
            lossValue = self.loss(preds, yBatch)

            # print(preds.argmax(dim=1))
            # print(lossValue)
            lossValue.backward()

            self.optimizer.step()
    
    def GetResults(self, xBatch):
        self.planesNetwork.eval()
        results = self.planesNetwork.forward(xBatch).argmax(dim=1)
        self.planesNetwork.train()

        return results
    
    def GetResult(self, imagePath):
        image: Image.Image = Image.open(imagePath)
        transform = transforms.ToTensor()
        tensor: torch.Tensor = transform(image)
        # remove alpha channel
        if (tensor.size(0) == 4):
            tensor = tensor[:-1]
        
        t = torch.stack([tensor])
        return self.GetResults(t)[0]

    
    def SaveModel(self, modelPath):
        torch.save(self.planesNetwork, f"{self._TRAINED_MODELS_PATH}{modelPath}")
    
    def LoadModel(self, modelPath):
        self.planesNetwork = torch.load(f"{self._TRAINED_MODELS_PATH}{modelPath}")
    
    def GetAllModels(self):
        models = []
        for _, _, filenames in os.walk(self._TRAINED_MODELS_PATH):
            for model in filenames:
                models.append(model)
        
        return models



if __name__ == "__main__":
    NetwrokController().TrainNetwork(15, 100, 1e-3)
