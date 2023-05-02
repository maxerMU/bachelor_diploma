import torch
import numpy as np

from models.conv_network import PlanesNetwork
from models.simple_net import simplenet
from dataset_handler.dataset_handler import DataSetHandler

from network_controllers import INetworkController

class NetwrokController(INetworkController):
    _MODEL_PATH="trained_models/simplenet.pt"

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
        return self.planesNetwork.forward(xBatch).argmax(dim=1)
    
    def SaveModel(self):
        torch.save(self.planesNetwork, self._MODEL_PATH)
    
    def LoadModel(self):
        self.planesNetwork = torch.load(self._MODEL_PATH)



if __name__ == "__main__":
    NetwrokController().TrainNetwork(15, 100, 1e-3)
