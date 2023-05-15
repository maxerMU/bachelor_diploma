import torch
from network_controllers import INetworkController, NetworkController

class SGDNetworkController(NetworkController):
    def __init__(self, batchSize, learningRate, needAug=True):
        super(SGDNetworkController, self).__init__(batchSize, learningRate, needAug)
        self.m_optimizer = torch.optim.SGD(self.m_planesNetwork.parameters(), lr=self.m_learningRate)
