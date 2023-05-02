from os import walk, path, listdir
import fnmatch
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH: str = "../planes_dataset"
TRAIN_TENSORS_PATH: str = "./train_tensors"
TEST_TENSORS_PATH: str = "./test_tensors"

class DataSetHandler:
    def TrainSize(self):
        files = listdir(TRAIN_TENSORS_PATH)
        return len(files) - 1 #file with y results

    def TestSize(self):
        files = listdir(TEST_TENSORS_PATH)
        return len(files) - 1 #file with y results

    def GetTrainingBatch(self, batchIndexes):
        return self._GetBatch(batchIndexes, f"{TRAIN_TENSORS_PATH}/train")

    def GetTestBatch(self, batchIndexes):
        return self._GetBatch(batchIndexes, f"{TEST_TENSORS_PATH}/test")
    
    def UpdateData(self):
        self._UpdateTrainData()
        self._UpdateTestData()

    def _GetBatch(self, batchIndexes, pathPrefix):
        y = []
        with open(f"{pathPrefix}_results.txt") as classesFile:
            classes = classesFile.read().split('\n')
            for i in batchIndexes:
                y.append(int(classes[i]))
        
        x = []
        for i in batchIndexes:
            x.append(torch.load(f"{pathPrefix}_{i}.pt"))

        return torch.stack(x), torch.Tensor(y).to(torch.long)

    def _UpdateTrainData(self):
        xTrain = []
        yTrain = []
        with open(f"{DATASET_PATH}/ImageSets/Main/train.txt") as trainImagesFile:
            for imageNumber in trainImagesFile:
                images, classes = self._FindAllImages(int(imageNumber))
                xTrain = xTrain + images
                yTrain = yTrain + classes
                
        for i in range(len(xTrain)):
            tensor: torch.Tensor = self._ConverToTensor(xTrain[i])
            torch.save(tensor, f"{TRAIN_TENSORS_PATH}/train_{i}.pt")

        with open(f"{TRAIN_TENSORS_PATH}/train_results.txt", "w") as f:
            for i in range(len(yTrain)):
                f.write(f"{yTrain[i]}\n")

    def _UpdateTestData(self):
        xTest = []
        yTest = []
        with open(f"{DATASET_PATH}/ImageSets/Main/test.txt") as testImagesFile:
            for imageNumber in testImagesFile:
                images, classes = self._FindAllImages(int(imageNumber))
                xTest = xTest + images
                yTest = yTest + classes
                
        for i in range(len(xTest)):
            # print(f"{i} -- {xTest[i]}")
            tensor: torch.Tensor = self._ConverToTensor(xTest[i])
            torch.save(tensor, f"{TEST_TENSORS_PATH}/test_{i}.pt")

        with open(f"{TEST_TENSORS_PATH}/test_results.txt", "w") as f:
            for i in range(len(yTest)):
                f.write(f"{yTest[i]}\n")


        # batchSize = 50
        # xTrainTensors = torch.stack(xTrain[:batchSize])
        # del(xTrain[:batchSize])
        # while len(xTrain) > 0:
        #     ram1 = psutil.virtual_memory().percent
        #     xTrainBatch = torch.stack(xTrain[:batchSize])
        #     xTrainTensors = torch.cat([xTrainTensors, xTrainBatch], dim=0)
        #     ram2 = psutil.virtual_memory().percent
        #     del(xTrain[:batchSize])
        #     # gc.collect()
        #     ram3 = psutil.virtual_memory().percent
        #     print(ram1, ram2, ram3)

        # return xTrainTensors, yTrain

    def _FindAllImages(self, imageNumber: int):
        images = []
        classes = []
        for root, dirnames, filenames in walk(f"{DATASET_PATH}/Parsed/"):
            for filename in fnmatch.filter(filenames, f"{imageNumber}_*.png"):
                planeImage = path.join(root, filename)
                images.append(planeImage)
                classNameIndex = path.dirname(planeImage).rfind("A")
                classNumber = int(path.dirname(planeImage)[classNameIndex+1:]) - 1
                classes.append(classNumber)

        return (images, classes)

    def _ConverToTensor(self, imagePath: str) -> torch.Tensor:
        image: Image.Image = Image.open(imagePath)
        transform = transforms.ToTensor()
        tensor: torch.Tensor = transform(image)
        # remove alpha channel
        if (tensor.size(0) == 4):
            tensor = tensor[:-1]

        return tensor
 

if __name__ == "__main__":
    DataSetHandler().UpdateData()
    print(DataSetHandler().TrainSize())
    print(DataSetHandler().TestSize())
    xTrain, yTrain = DataSetHandler().GetTrainingBatch(range(100))
    print(xTrain[0], yTrain[0])
    print(xTrain.size(0), xTrain.size(1), xTrain.size(2), xTrain.size(3))

    im2display = np.transpose(xTrain[0], (1,2,0))
    plt.imshow(im2display)
    plt.show()
