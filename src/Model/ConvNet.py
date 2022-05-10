import copy

import torch.nn as nn
import torch.nn.functional as F

import DataSet.CIFAR10


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

    def train_one_epoch(self, epoch, dev, train_dl, model, loss_func, opti):
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
                # backward
                loss.backward()
                # update weights
                opti.step()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return weights

if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
    trainset2 = DataSet.CIFAR10.CIFAR10(50).get_train_dataset()[0]
    trainloader2 = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset2, batch_size=4,
                                              shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root='../data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = ConvNet()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # 多批次循环

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入
            inputs, labels = data

            # 梯度置0
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000批次打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')