from dataset.CIFAR10 import CIFAR10
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 3, 32, 32)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor
    
# Define the CNN model
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)  # The image size will be reduced to 3x3 after pooling
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))  # 28x28x1 => 14x14x32
        x = self.pool(F.relu(self.conv2(x)))  # 14x14x32 => 7x7x64
        x = self.pool(F.relu(self.conv3(x)))  # 7x7x64 => 3x3x64
        # Flatten image input
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        # Add dropouts to avoid overfitting
        # x = F.dropout(x), p=0.5)
        x = self.fc2(x)
        return x
    
class CIFARCNN(nn.Module):
    def __init__(self):
        super(CIFARCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # The image size will be 4x4 after pooling layers
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32x3 => 16x16x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x16x32 => 8x8x64
        x = self.pool(F.relu(self.conv3(x)))  # 8x8x64 => 4x4x64
        x = x.view(-1, 64 * 4 * 4)            # Flatten the vector
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x