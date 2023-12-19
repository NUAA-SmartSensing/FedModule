import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_L1(nn.Module):
    def __init__(self, l1_lambda):
        super(CNN_L1, self).__init__()
        self.mask_list = {}
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)
        self.l1_lambda = l1_lambda  # L1正则化系数

        # 添加权重掩码
        self.mask_list["conv1"] = nn.Parameter(torch.ones_like(self.conv1.weight), requires_grad=False)
        self.mask_list["conv2"] = nn.Parameter(torch.ones_like(self.conv2.weight), requires_grad=False)
        self.mask_list["fc1"] = nn.Parameter(torch.ones_like(self.fc1.weight), requires_grad=False)
        self.mask_list["fc2"] = nn.Parameter(torch.ones_like(self.fc2.weight), requires_grad=False)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor * self.mask_list["conv1"]))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor * self.mask_list["conv2"]))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor * self.mask_list["fc1"]))
        tensor = self.fc2(tensor * self.mask_list["fc2"])
        return tensor

    def l1_regularization_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

    # 剪枝算法


