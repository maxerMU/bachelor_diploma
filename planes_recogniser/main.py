from network_controllers import AugmentationNetwrokController, INetworkController

def train(networkController: INetworkController):
    networkController.TrainNetwork(15)
    networkController.SaveModel()

if __name__ == "__main__":
    controller = AugmentationNetwrokController(100, 1e-3)
    train(controller)