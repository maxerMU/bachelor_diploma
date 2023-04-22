import torch

class PlanesNetwork(torch.nn.Module):
    def __init__(self, numClasses: int):
        super(PlanesNetwork, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, padding=2)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(in_channels=9, out_channels=16, kernel_size=5, padding=0)
        self.act2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, padding=0)
        self.act3 = torch.nn.ReLU()
        self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(in_features=9*9*20, out_features=512)
        self.act3 = torch.nn.Tanh()

        self.dropout = torch.nn.Dropout(p=0.6)

        self.fc2 = torch.nn.Linear(in_features=512, out_features=84)
        self.act4 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(in_features=84, out_features=numClasses)
    
    def forward(self, image):
        image = self.conv1(image)
        image = self.act1(image)
        image = self.pool1(image)

        image = self.conv2(image)
        image = self.act2(image)
        image = self.pool2(image)

        image = self.conv3(image)
        image = self.act3(image)
        image = self.pool3(image)

        image = image.view(image.size(0), image.size(1) * image.size(2) * image.size(3))

        image = self.fc1(image)
        image = self.act3(image)

        image = self.dropout(image)

        image = self.fc2(image)
        image = self.act4(image)
        image = self.fc3(image)

        return image