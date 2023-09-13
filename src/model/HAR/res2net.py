import torch
import torch.nn as nn

'''Res2Net: 感受野堆叠'''


class Res2NetBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.k1 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)),
            nn.BatchNorm2d(channel)
        )
        self.k2 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)),
            nn.BatchNorm2d(channel)
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)),
            nn.BatchNorm2d(channel)
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)),
            nn.BatchNorm2d(channel)
        )
        self.merge = nn.Conv2d(channel * 4, channel, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x1 = x
        y1 = self.k1(x1)

        x2 = x + y1
        y2 = self.k2(self.relu(x2))

        x3 = x + y2
        y3 = self.k3(self.relu(x3))

        x4 = x + y3
        y4 = self.k4(self.relu(x4))

        out = self.merge(torch.cat([y1, y2, y3, y4], dim=1))  # 按channel维度cat后进行合并

        return out


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (3, 1), (stride, 1), (1, 0)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            Res2NetBlock(channel=outchannel),
            nn.BatchNorm2d(outchannel)
        )
        self.short = nn.Sequential()
        if (inchannel != outchannel or stride != 1):
            self.short = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, (3, 1), (stride, 1), (1, 0)),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        out = self.block(x) + self.short(x)
        return nn.ReLU()(out)


class Res2Net(nn.Module):
    def __init__(self, train_shape, category):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer1 = self.make_layers(1, 64, 2, 1)
        self.layer2 = self.make_layers(64, 128, 2, 1)
        self.layer3 = self.make_layers(128, 256, 2, 1)
        self.layer4 = self.make_layers(256, 512, 2, 1)
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(512 * train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layers(self, inchannel, outchannel, stride, blocks):
        layer = [Block(inchannel, outchannel, stride)]
        for i in range(1, blocks):
            layer.append(Block(outchannel, outchannel, 1))
        return nn.Sequential(*layer)
