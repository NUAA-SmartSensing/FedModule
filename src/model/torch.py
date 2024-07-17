from torch import nn
from torchvision import models


class ResNet18Pre(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Pre, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
