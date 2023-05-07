import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from random import randint, choice

from models.conv_network import PlanesNetwork
from dataset_handler.dataset_handler import DataSetHandler

from network_controllers import INetworkController, NetworkController

class AugmentationNetworkController(NetworkController):
    ROTATION_ANGLE_MIN = 20
    ROTATION_ANGLE_MAX = 50
    # _MODEL_PATH = "trained_models/augconvnet.pt"
    # _MODEL_PATH = "trained_models/augsimplenet1.pt"

    def __init__(self, batchSize, learningRate):
        super(AugmentationNetworkController, self).__init__(batchSize, learningRate)

    def TrainPrepare(self):
        pass

    def TrainEpoch(self):
        order = np.random.permutation(self.m_datasetLen)
        j = 0
        for startIndex in range(0, self.m_datasetLen, self.m_batchSize):
            self.m_optimizer.zero_grad()

            batches = self.m_datasetHandler.GetTrainingBatch(order[startIndex:startIndex+self.m_batchSize])
            xBatch: torch.Tensor = batches[0]
            yBatch: torch.Tensor = batches[1]
            xBatchAug, yBatchAug = self._AugmentateBatches(xBatch, yBatch)

            preds = self.m_planesNetwork.forward(xBatchAug)
            lossValue = self.m_loss(preds, yBatchAug)

            # print(preds.argmax(dim=1))
            # print(lossValue)
            print(float(startIndex) / self.m_datasetLen)
            lossValue.backward()

            self.m_optimizer.step()

            j += 1
            if j % 1 == 0:
                self.LogTraining()
    
    # def GetResults(self, xBatch):
    #     return self.planesNetwork.forward(xBatch).argmax(dim=1)
    
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
            randAugTransform = transforms.RandomRotation([self.ROTATION_ANGLE_MIN, self.ROTATION_ANGLE_MAX])
            augTensor1 = randAugTransform(tensor)
            randAugTransform = transforms.RandomRotation([self.ROTATION_ANGLE_MIN * 3, self.ROTATION_ANGLE_MAX * 3])
            augTensor2 = randAugTransform(tensor)
            augTensor3 = transforms.functional.adjust_brightness(tensor, 2)
            augTensor4 = transforms.functional.gaussian_blur(tensor, kernel_size=(5,9), sigma=(0.1, 5))
            colorJitterTransform = transforms.ColorJitter(brightness=.5, hue=.3)
            augTensor5 = colorJitterTransform(tensor)

            xBatchAug += [tensor, augTensor1, augTensor2, augTensor3, augTensor4, augTensor5]
            yBatchAug += [yBatch[i].item()] * 6
        
        order = np.random.permutation(len(xBatchAug))
        xBatchAug = np.array(xBatchAug)[order]
        yBatchAug = np.array(yBatchAug)[order]

        return torch.stack(list(xBatchAug)), torch.Tensor(list(yBatchAug)).to(torch.long)
