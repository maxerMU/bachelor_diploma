import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from random import randint, choice

from models.conv_network import PlanesNetwork
from dataset_handler.dataset_handler import DataSetHandler

from network_controllers import INetworkController, NetworkController

class AugmentationNetwrokController(NetworkController.NetwrokController):
    ROTATION_ANGLE_MIN = 8
    ROTATION_ANGLE_MAX = 20

    def __init__(self, batchSize, learningRate):
        super(AugmentationNetwrokController, self).__init__(batchSize, learningRate)

    def TrainPrepare(self):
        pass

    def TrainEpoch(self):
        order = np.random.permutation(self.datasetLen)
        for startIndex in range(0, self.datasetLen, self.batchSize):
            self.optimizer.zero_grad()

            batches = self.datasetHandler.GetTrainingBatch(order[startIndex:startIndex+self.batchSize])
            xBatch: torch.Tensor = batches[0]
            yBatch: torch.Tensor = batches[1]
            xBatchAug, yBatchAug = self._AugmentateBatches(xBatch, yBatch)

            preds = self.planesNetwork.forward(xBatchAug)
            lossValue = self.loss(preds, yBatchAug)

            # print(preds.argmax(dim=1))
            # print(lossValue)
            lossValue.backward()

            self.optimizer.step()
    
    def GetResults(self, xBatch):
        return self.planesNetwork.forward(xBatch).argmax(dim=1)
    
    def _AugmentateBatches(self, xBatch: torch.Tensor, yBatch: torch.Tensor):
        xBatchAug = []
        yBatchAug = []

        for i, tensor in enumerate(xBatch):
            # toImageTransform = transforms.ToPILImage()
            # image: Image.Image = toImageTransform(tensor)
            # angle = randint(self.ROTATION_ANGLE_MIN, self.ROTATION_ANGLE_MAX) * choice([1, -1])
            # image = image.rotate(angle)
            # toTensorTransform = transforms.ToTensor()
            # augTensor = toTensorTransform(image)
            randAugTransform = transforms.RandomRotation(self.ROTATION_ANGLE_MAX)
            augTensor1 = randAugTransform(tensor)
            randAugTransform = transforms.RandomRotation(self.ROTATION_ANGLE_MAX * 3)
            augTensor2 = randAugTransform(tensor)

            xBatchAug += [tensor, augTensor1, augTensor2]
            yBatchAug += [yBatch[i].item()] * 3
        
        order = np.random.permutation(len(xBatchAug))
        xBatchAug = np.array(xBatchAug)[order]
        yBatchAug = np.array(yBatchAug)[order]

        return torch.stack(list(xBatchAug)), torch.Tensor(list(yBatchAug)).to(torch.long)


if __name__ == "__main__":
    NetwrokController().TrainNetwork(15, 100, 1e-3)
