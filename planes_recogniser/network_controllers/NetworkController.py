import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os

from models.conv_network import PlanesNetwork
from models.simple_net import simplenet
from dataset_handler.dataset_handler import DataSetHandler

from network_controllers import INetworkController

class NetworkController(INetworkController):
    _TRAINED_MODELS_PATH = "trained_models/"

    def __init__(self, batchSize, learningRate, needAug=True):
        # self.m_planesNetwork = PlanesNetwork(20)
        # self.m_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO
        self.m_device = "cpu"
        print(self.m_device)
        self.m_planesNetwork = simplenet(20).to(self.m_device)
        # print(sum(p.numel() for p in self.m_planesNetwork.parameters() if p.requires_grad))
        self.m_datasetHandler = DataSetHandler()

        self.m_batchSize = batchSize
        self.m_learningRate = learningRate
        self.m_needAug = needAug

        self.m_loss = torch.nn.CrossEntropyLoss()
        # TODO change on ADAM
        self.m_optimizer = torch.optim.Adam(self.m_planesNetwork.parameters(), lr=self.m_learningRate)
        self.m_datasetLen = self.m_datasetHandler.TrainSize()
    
    def TrainPrepare(self):
        pass

    def TrainEpoch(self):
        self.m_planesNetwork.train()
        order = np.random.permutation(self.m_datasetLen)
        for startIndex in range(0, self.m_datasetLen, self.m_batchSize):
            self.m_optimizer.zero_grad()

            xBatch, yBatch = self.m_datasetHandler.GetTrainingBatch(order[startIndex:startIndex+self.m_batchSize], needAug=self.m_needAug)
            xBatch = xBatch.to(self.m_device)
            yBatch = yBatch.to(self.m_device)

            preds = self.m_planesNetwork.forward(xBatch)
            lossValue = self.m_loss(preds, yBatch)

            # print(preds.argmax(dim=1))
            # print(lossValue)
            lossValue.backward()

            self.m_optimizer.step()
    
    def GetResults(self, xBatch):
        self.m_planesNetwork.eval()
        results = self.m_planesNetwork.forward(xBatch).argmax(dim=1)
        self.m_planesNetwork.train()

        return results
    
    def GetResult(self, imagePath):
        image: Image.Image = Image.open(imagePath)
        transform = transforms.ToTensor()
        tensor: torch.Tensor = transform(image)
        # remove alpha channel
        if (tensor.size(0) == 4):
            tensor = tensor[:-1]
        
        # # TODO constant
        # stride = 5
        # i = 0

        # while i < len(tensor[0]) + 96:
        #     j = 0
        #     while j < len(tensor[0][0]) + 96:
        #         sector = tensor[:, i:i+96, j:j+96]

        #         im2display = np.transpose(sector, (1,2,0))
        #         plt.imshow(im2display)
        #         plt.show()
        #         t = torch.stack([sector])
        #         self.GetResults(t)


        #         j += stride
        #     i += stride
        
        t = torch.stack([tensor])
        return self.GetResults(t)[0]

    
    def SaveModel(self, modelPath):
        torch.save(self.m_planesNetwork, f"{self._TRAINED_MODELS_PATH}{modelPath}")
    
    def LoadModel(self, modelPath):
        self.m_planesNetwork = torch.load(f"{self._TRAINED_MODELS_PATH}{modelPath}")
        self.m_planesNetwork.to(self.m_device)
    
    def GetAllModels(self):
        models = []
        for _, _, filenames in os.walk(self._TRAINED_MODELS_PATH):
            for model in filenames:
                models.append(model)
        
        return models



if __name__ == "__main__":
    NetworkController().TrainNetwork(15, 100, 1e-3)
