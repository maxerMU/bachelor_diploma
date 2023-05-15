import numpy as np
import torch
from network_controllers import INetworkController, NetworkController

class SortNetworkController(NetworkController):
    def __init__(self, batchSize, learningRate, needAug=True):
        super(SortNetworkController, self).__init__(batchSize, learningRate, needAug)
        self.m_order = np.random.permutation(self.m_datasetLen)
    
    class LossIndex():
        def __init__(self, loss: float, index: int) -> None:
            self.loss = loss
            self.index = index

    def TrainPrepare(self):
        super().TrainPrepare()
        self.m_order = np.random.permutation(self.m_datasetLen)
    
    def TrainEpoch(self):
        self.m_planesNetwork.train()
        for startIndex in range(0, self.m_datasetLen, self.m_batchSize):
            self.m_optimizer.zero_grad()

            xBatch, yBatch = self.m_datasetHandler.GetTrainingBatch(self.m_order[startIndex:startIndex+self.m_batchSize], needAug=self.m_needAug)
            xBatch.to(self.m_device)
            yBatch.to(self.m_device)

            preds = self.m_planesNetwork.forward(xBatch)
            lossValue = self.m_loss(preds, yBatch)

            # print(preds.argmax(dim=1))
            # print(lossValue)
            lossValue.backward()

            self.m_optimizer.step()
        
        self._UpdateOrder()

    def _UpdateOrder(self):
        self.m_planesNetwork.eval()

        indexesLosses = []
        for i in self.m_order:
            xBatch, yBatch = self.m_datasetHandler.GetTrainingBatch([i])
            xBatch.to(self.m_device)
            yBatch.to(self.m_device)

            preds = self.m_planesNetwork.forward(xBatch)
            lossValue = self.m_loss(preds, yBatch)

            indexesLosses.append(self.LossIndex(lossValue, i))
        
        indexesLosses.sort(lambda x: x.loss)
        self.m_order = [x.index for x in indexesLosses]

        self.m_planesNetwork.train()
