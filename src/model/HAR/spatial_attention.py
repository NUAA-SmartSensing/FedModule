import torch
import torch.nn as nn

'''Spatial Attention Neural Network: 空间注意力'''


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_fc = nn.Sequential(
            nn.Conv2d(1, 1, (3, 1), (1, 1), (1, 0)),  # 传感数据特殊性，固定模态轴，只在时序轴上进行空间注意力
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        att = torch.mean(x, dim=1, keepdim=True)  # [b, c, series, modal] -> [b, 1, series, modal]
        att = self.att_fc(att)  # [b, 1, series, modal]
        out = x * att
        return out


class SpatialAttentionNeuralNetwork(nn.Module):
    def __init__(self, train_shape, category):
        super(SpatialAttentionNeuralNetwork, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (3, 1), (2, 1), (1, 0)),
            SpatialAttentionModule(),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
            SpatialAttentionModule(),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),
            SpatialAttentionModule(),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)),
            SpatialAttentionModule(),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(512 * train_shape[-1], category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
