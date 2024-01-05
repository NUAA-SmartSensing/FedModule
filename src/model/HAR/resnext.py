import torch
import torch.nn as nn

'''ResNext: 分组卷积'''


class ResNextBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, groups):
        super().__init__()
        if inchannel != 1:
            assert inchannel % groups == 0  # inchannel必须整除groups
            group_channel = inchannel // groups
        else:
            group_channel = outchannel
        layer_lists = []
        for i in range(groups):
            layer_lists.append(nn.Sequential(
                nn.Conv2d(inchannel, group_channel, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0)),
                nn.BatchNorm2d(group_channel),
                nn.ReLU(),
                nn.Conv2d(group_channel, outchannel, 1, 1, 0),
                nn.BatchNorm2d(outchannel)
            ))

        self.group_layer_lists = nn.ModuleList(layer_lists)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        out = torch.cat([group_layer(x).unsqueeze(0) for group_layer in self.group_layer_lists], dim=0)
        out = self.relu(torch.sum(out, dim=0))
        return out


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, groups):
        super().__init__()
        self.block = ResNextBlock(inchannel=inchannel, outchannel=outchannel, kernel_size=kernel_size, stride=stride,
                                  groups=groups)
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


class ResNext(nn.Module):
    def __init__(self, train_shape, category, groups=4):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
            groups: 每个ResNextBlock里有多少并行组
        '''
        self.groups = groups
        self.layer1 = self.make_layers(1, 64, 3, 2, 1)
        self.layer2 = self.make_layers(64, 128, 3, 2, 1)
        self.layer3 = self.make_layers(128, 256, 3, 2, 1)
        self.layer4 = self.make_layers(256, 512, 3, 2, 1)
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

    def make_layers(self, inchannel, outchannel, kernel_size, stride, blocks):
        layer = [Block(inchannel, outchannel, kernel_size, stride, self.groups)]
        for i in range(1, blocks):
            layer.append(Block(outchannel, outchannel, kernel_size, 1, self.groups))
        return nn.Sequential(*layer)
