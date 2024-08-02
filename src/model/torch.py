from torch import nn
from torchvision import models


class ResNet18Pre(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Pre, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet18ForOneTunnel(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18ForOneTunnel, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
