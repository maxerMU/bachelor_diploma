import torch

class ConvRelu(torch.nn.Module):
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int, stride: int, padding: int):
        super(ConvRelu, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=kernelSize,
                            stride=stride, padding=padding),
            torch.nn.ReLU()
        )
    
    def forward(self, tensor):
        return self.conv(tensor)

class ReduceConvBlock(torch.nn.Module):
    def __init__(self, inChannels: int, outChannels1: int, outChannels2: int, kernelSize: int, padding: int):
        super(ReduceConvBlock, self).__init__()
        self.redConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=inChannels, out_channels=outChannels1, kernel_size=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=outChannels1, out_channels=outChannels2, kernel_size=kernelSize, stride=1, padding=padding),
            torch.nn.ReLU()
        )

    def forward(self, tensor):
        tensor = self.redConv(tensor)

        return tensor

class InceptionBlock(torch.nn.Module):
    def __init__(self, inChannels: int, out1: int, out3r: int, out3: int, out5r: int, out5: int, poolOut: int):
        super(InceptionBlock, self).__init__()
        self.conv1 = ConvRelu(inChannels, out1, 1, 1, 0)
        self.conv3 = ReduceConvBlock(inChannels, out3r, out3, 3, 1)
        self.conv5 = ReduceConvBlock(inChannels, out5r, out5, 5, 2)

        self.poolConv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=1, stride=1),
            torch.nn.Conv2d(in_channels=inChannels, out_channels=poolOut, kernel_size=1, stride=1),
            torch.nn.ReLU()
        )
    
    def forward(self, tensor):
        conv1 = self.conv1(tensor)
        conv3 = self.conv3(tensor)
        conv5 = self.conv5(tensor)
        pool = self.poolConv(tensor)

        return torch.cat([conv1, conv3, conv5, pool], dim=1)

class ClassifyBlock(torch.nn.Module):
    def __init__(self, inChannels: int, numClasses: int):
        super(ClassifyBlock, self).__init__()

        self.avgPool = torch.nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv = torch.nn.Conv2d(in_channels=inChannels, out_channels=128, kernel_size=1, stride=1)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features= 4 * 4 * 128, out_features=1024)
        self.dropout = torch.nn.Dropout(p=0.7)
        self.classes = torch.nn.Linear(in_features=1024, out_features=numClasses)

    def forward(self, tensor: torch.Tensor):
        tensor = self.avgPool(tensor)
        tensor = self.conv(tensor)
        tensor = self.relu(tensor)

        tensor = tensor.view(tensor.size(0), tensor.size(1) * tensor.size(2) * tensor.size(3))
        
        tensor = self.fc(tensor)
        tensor = self.dropout(tensor)
        tensor = self.classes(tensor)

        return tensor


class PlanesNetwork(torch.nn.Module):
    def __init__(self, numClasses: int):
        super(PlanesNetwork, self).__init__()

        self.conv1 = ConvRelu(3, 64, 7, 2, 3)
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = torch.nn.Sequential(
            ConvRelu(64, 64, 1, 1, 0),
            ConvRelu(64, 192, 3, 1, 1)
        )

        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.classifier1 = ClassifyBlock(512, numClasses)
        self.classifier2 = ClassifyBlock(528, numClasses)
        self.avgPool = torch.nn.AdaptiveAvgPool2d(output_size=7)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(in_features=7*7*1024, out_features=numClasses)
        )

    def forward(self, tensor: torch.Tensor):
        tensor = self.conv1(tensor)
        tensor = self.maxPool1(tensor)
        tensor = self.conv2(tensor)
        tensor = self.maxPool1(tensor)
        tensor = self.inception_3a(tensor)
        tensor = self.inception_3b(tensor)
        tensor = self.maxPool1(tensor)
        tensor = self.inception_4a(tensor)

        res1 = self.classifier1(tensor)

        tensor = self.inception_4b(tensor)
        tensor = self.inception_4c(tensor)
        tensor = self.inception_4d(tensor)

        res2 = self.classifier2(tensor)

        tensor = self.inception_4e(tensor)
        tensor = self.maxPool1(tensor)
        tensor = self.inception_5a(tensor)
        tensor = self.inception_5b(tensor)
        tensor = self.avgPool(tensor)

        tensor = tensor.view(tensor.size(0), tensor.size(1) * tensor.size(2) * tensor.size(3))
        res = self.classifier(tensor)

        if self.training:
            return [res, res1, res2]
        else:
            return res