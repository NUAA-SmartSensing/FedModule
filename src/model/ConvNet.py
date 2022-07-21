import copy

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_one_epoch(self, epoch, dev, train_dl, model, loss_func, opti, mu):
        if mu != 0:
            global_model = copy.deepcopy(model)
        data_sum = 0
        for t in range(epoch):
            # 设置迭代次数
            running_loss = 0.0
            for i, data in enumerate(train_dl, 0):
                # get the inputs
                inputs, labels = data
                # warp them in Variable
                inputs, labels = inputs.to(dev), labels.to(dev)

                # zero the parameter gradients
                opti.zero_grad()

                # forward
                outputs = model(inputs)
                # loss
                loss = loss_func(outputs, labels)
                data_sum += labels.size(0)
                # 正则项
                if mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (mu / 2) * proximal_term
                # backward
                loss.backward()
                # update weights
                opti.step()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights


if __name__ == '__main__':
    import torch
    net = ConvNet()
    net2 = ConvNet()
    w = net.state_dict()
    w2 = net2.state_dict()
    total_diff = 0
    for key, var in w.items():
        total_diff += torch.sum((w[key] - w2[key]) ** 2)
    a = total_diff.tolist()
    b = type(a)
    print(b)
