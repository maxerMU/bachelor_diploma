import torch
import numpy as np

from models.conv_network import PlanesNetwork
from dataset_handler.dataset_handler import DataSetHandler

from network_controllers import INetworkController

class NetwrokController(INetworkController):
    def __init__(self, batchSize, learningRate):
        self.planesNetwork = PlanesNetwork(20)
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


if __name__ == "__main__":
    NetwrokController().TrainNetwork(15, 100, 1e-3)
