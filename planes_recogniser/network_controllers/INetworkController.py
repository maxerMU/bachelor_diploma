from abc import ABC, abstractmethod
import numpy as np
from dataset_handler.dataset_handler import DataSetHandler

class INetworkController(ABC):
    @abstractmethod
    def TrainPrepare(self):
        pass

    @abstractmethod
    def TrainEpoch(self):
        pass
    
    @abstractmethod
    def GetResults(self, xBatch):
        pass

    @abstractmethod
    def SaveModel(self):
        pass

    @abstractmethod
    def LoadModel(self):
        pass

    def TrainNetwork(self, epochs: int):
        self.TrainPrepare()

        for epoch in range(epochs):
            self.TrainEpoch()
            return
            self.LogTraining()
    
    def LogTraining(self):
        datasetHandler = DataSetHandler()
        #TODO
        xBatchTest, yBatchTest = self.datasetHandler.GetTestBatch(np.random.permutation(range(datasetHandler.TestSize()))[:500])
        predsTest = self.GetResults(xBatchTest)

        xBatchTrain, yBatchTrain = self.datasetHandler.GetTrainingBatch(np.random.permutation(range(datasetHandler.TrainSize()))[:500])
        predsTrain = self.GetResults(xBatchTrain)

        print("=====================================")
        testMisses = [0] * 20
        testClasses = [0] * 20
        recognizedTest = 0
        for i in range(len(predsTest)):
            testClasses[yBatchTest[i]] += 1
            if predsTest[i] == yBatchTest[i]:
                recognizedTest += 1
            else:
                testMisses[yBatchTest[i]] += 1


        trainMisses = [0] * 20
        trainClasses = [0] * 20
        recognizedTrain = 0
        for i in range(len(predsTrain)):
            trainClasses[yBatchTest[i]] += 1
            if predsTrain[i] == yBatchTrain[i]:
                recognizedTrain += 1
            else:
                trainMisses[yBatchTest[i]] += 1


        print("".join(map(lambda x: "{:6}".format(x), list(range(1,21)))))
        print("".join(map(lambda x: "{:6}".format(x), testClasses)))
        print("".join(map(lambda x: "{:6}".format(x), testMisses)))
        print("".join(map(lambda x: "{:6}".format(x), trainClasses)))
        print("".join(map(lambda x: "{:6}".format(x), trainMisses)))

        print(float(recognizedTest) / len(predsTest))
        print(float(recognizedTrain) / len(predsTrain))
        print("=====================================")

