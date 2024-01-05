import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_pruning(nn.Module):
    def __init__(self):
        super(CNN_pruning, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

        # 添加权重掩码
        self.mask_conv1 = torch.ones_like(self.conv1.weight, dtype=torch.bool)
        self.mask_conv2 = torch.ones_like(self.conv2.weight, dtype=torch.bool)
        self.mask_fc1 = torch.ones_like(self.fc1.weight, dtype=torch.bool)
        self.mask_fc2 = torch.ones_like(self.fc2.weight, dtype=torch.bool)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # conv1
        masked_weight = self.conv1.weight * self.mask_conv1
        x = F.relu(F.conv2d(x, masked_weight, stride=self.conv1.stride, padding=self.conv1.padding))
        x = self.pool1(x)
        # conv2
        masked_weight = self.conv2.weight * self.mask_conv2
        x = F.relu(F.conv2d(x, masked_weight, stride=self.conv2.stride, padding=self.conv2.padding))
        x = self.pool2(x)

        x = x.view(-1, 7 * 7 * 64)
        # fc1
        masked_weight = self.fc1.weight * self.mask_fc1
        x = F.linear(x, masked_weight)
        x = F.relu(x)
        # fc2
        masked_weight = self.fc2.weight * self.mask_fc2
        x = F.linear(x, masked_weight)
        return x

    # 剪枝算法
    def pruning_by_ratio(self, ratio):
        if ratio == 0:
            return

        # 逐层修改mask
        # conv1
        weights_val = self.conv1.weight[self.mask_conv1 == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[int(ratio * self.conv1.weight.nelement())]
        self.mask_conv1 *= (torch.abs(self.conv1.weight) >= thr)

        # conv2
        weights_val = self.conv2.weight[self.mask_conv2 == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[int(ratio * self.conv2.weight.nelement())]
        self.mask_conv2 *= (torch.abs(self.conv2.weight) >= thr)

        # fc1
        weights_val = self.fc1.weight[self.mask_fc1 == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[int(ratio * self.fc1.weight.nelement())]
        self.mask_fc1 *= (torch.abs(self.fc1.weight) >= thr)

        # fc2
        weights_val = self.fc2.weight[self.mask_fc2 == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[int(ratio * self.fc2.weight.nelement())]
        self.mask_fc2 *= (torch.abs(self.fc2.weight) >= thr)

    def print_prune(self):
        print("conv1 : remaining/all : {}/{}".format(torch.sum(self.mask_conv1).int().item(),
                                                     self.conv1.weight.nelement()))
        print("conv2 : remaining/all : {}/{}".format(torch.sum(self.mask_conv2).int().item(),
                                                     self.conv2.weight.nelement()))
        print("fc1 : remaining/all : {}/{}".format(torch.sum(self.mask_fc1).int().item(),
                                                   self.fc1.weight.nelement()))
        print("fc2 : remaining/all : {}/{}".format(torch.sum(self.mask_fc2).int().item(),
                                                   self.fc2.weight.nelement()))

    def to(self, device):
        self.mask_conv1 = self.mask_conv1.to(device)
        self.mask_conv2 = self.mask_conv2.to(device)
        self.mask_fc1 = self.mask_fc1.to(device)
        self.mask_fc2 = self.mask_fc2.to(device)
        return super().to(device)
