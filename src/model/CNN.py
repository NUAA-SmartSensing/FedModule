import copy

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

    def train_one_epoch(self, epoch, dev, train_dl, model, loss_func, opti):
        # 设置迭代次数
        data_sum = 0
        for epoch in range(epoch):
            for data, label in train_dl:
                data, label = data.to(dev), label.to(dev)
                # 模型上传入数据
                preds = model(data)
                # 计算损失函数
                loss = loss_func(preds, label)
                data_sum += label.size(0)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights
