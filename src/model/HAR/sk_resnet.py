import torch
import torch.nn as nn

'''Selective Kernel Resdual Network: 选择核残差网络'''


class SKBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        # SK 选择核思想 这里默认选了(3,1), (5,1), (7,1), (9,1) 四种并行进行选择 每一个尺度是独立的Conv2D
        self.kernel1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (3, 1), (stride, 1), (1, 0)),
            nn.BatchNorm2d(outchannel)
        )
        self.kernel2 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (5, 1), (stride, 1), (2, 0)),
            nn.BatchNorm2d(outchannel)
        )
        self.kernel3 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (7, 1), (stride, 1), (3, 0)),
            nn.BatchNorm2d(outchannel)
        )
        self.kernel4 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(outchannel)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.att_fc = nn.Sequential(
            nn.Linear(outchannel, outchannel * 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        y1 = self.kernel1(x).unsqueeze(1)  # [b, 1, outchannel, h, modal]
        y2 = self.kernel2(x).unsqueeze(1)
        y3 = self.kernel3(x).unsqueeze(1)
        y4 = self.kernel4(x).unsqueeze(1)
        y_total = torch.cat([y1, y2, y3, y4], dim=1)  # [b, 4, outchannel, h, modal]

        attn = self.att_fc(self.gap(torch.sum(y_total, dim=1)).squeeze(-1).squeeze(-1))  # [b, outchanel * 4]
        attn = attn.reshape(x.size(0), 4, -1).unsqueeze(-1).unsqueeze(-1)  # [b, 4, outchanel, 1, 1]

        attn_y = y_total * attn  # [b, 4, outchannel, h, modal] * [b, 4, outchanel, 1, 1] = [b, 4, outchannel, h, modal]
        out = torch.sum(attn_y, dim=1)  # [b, outchannel, h, modal]
        return out


class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            SKBlock(inchannel=inchannel, outchannel=outchannel, stride=stride),
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


class SKResNet(nn.Module):
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
