import torch.nn as nn

'''Depthwise Convolutional Neural Network: 深度可分离卷积'''
'''即 MobileNet V1，引入通道注意力Channel Attention机制后即为 MobileNet V3'''


class DepthwiseConv(nn.Module):
    def __init__(self, train_shape, category, kernel_size=3):
        super(DepthwiseConv, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
            kernel_size: 时序维度 卷积核尺寸
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 1, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=1),
            nn.Conv2d(1, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=64),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=128),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=256),
            nn.Conv2d(256, 512, 1, 1, 0),
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
