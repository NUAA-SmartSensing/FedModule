import torch.nn as nn

'''Resdual Neural Network'''


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (3, 1), (stride, 1), (1, 0)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 1, 1, 0),
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


class ResNet(nn.Module):
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
