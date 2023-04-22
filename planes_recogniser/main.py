from network_controllers import AugmentationNetwrokController, INetworkController

def train(networkController: INetworkController):
    networkController.TrainNetwork(15)

if __name__ == "__main__":
    controller = AugmentationNetwrokController(50, 1e-3)
    train(controller)