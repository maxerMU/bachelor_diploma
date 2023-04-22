from abc import ABC, abstractmethod
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

    def TrainNetwork(self, epochs: int):
        self.TrainPrepare()

        for epoch in range(epochs):
            self.TrainEpoch()
            self.LogTraining()
    
    def LogTraining(self):
        datasetHandler = DataSetHandler()
        #TODO
        xBatchTest, yBatchTest = self.datasetHandler.GetTestBatch(range(3000))
        predsTest = self.GetResults(xBatchTest)

        xBatchTrain, yBatchTrain = self.datasetHandler.GetTrainingBatch(range(3000))
        predsTrain = self.GetResults(xBatchTrain)

        print("=====================================")
        testMisses = [0] * 20
        recognizedTest = 0
        for i in range(len(predsTest)):
            if predsTest[i] == yBatchTest[i]:
                recognizedTest += 1
            else:
                testMisses[yBatchTest[i]] += 1


        trainMisses = [0] * 20
        recognizedTrain = 0
        for i in range(len(predsTrain)):
            if predsTrain[i] == yBatchTrain[i]:
                recognizedTrain += 1
            else:
                trainMisses[yBatchTest[i]] += 1


        print("".join(map(lambda x: "{:6}".format(x), list(range(1,21)))))
        print("".join(map(lambda x: "{:6}".format(x), testMisses)))
        print("".join(map(lambda x: "{:6}".format(x), trainMisses)))

        print(float(recognizedTest) / len(predsTest))
        print(float(recognizedTrain) / len(predsTrain))
        print("=====================================")

