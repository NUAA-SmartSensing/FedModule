import torch.nn as nn

'''ShuffleNet: 通道混合轻量级网络'''
'''即 MobileNet（DepthWise Net）基础上引入通道混合机制（Channel Shuffle）'''


class ChannelShuffleModule(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        '''
            channels: 张量通道数
            groups: 通道组数【将channels分为groups组去shuffle】
        '''
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.channel_per_group = self.channels // self.groups

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        batch, _, series, modal = x.size()
        x = x.reshape(batch, self.groups, self.channel_per_group, series, modal)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch, self.channels, series, modal)
        return x


class ShuffleNet(nn.Module):
    def __init__(self, train_shape, category, kernel_size=3):
        super(ShuffleNet, self).__init__()
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
            ChannelShuffleModule(channels=64, groups=8),

            nn.Conv2d(64, 64, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=64),
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ChannelShuffleModule(channels=128, groups=8),

            nn.Conv2d(128, 128, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=128),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ChannelShuffleModule(channels=256, groups=16),

            nn.Conv2d(256, 256, (kernel_size, 1), (2, 1), (kernel_size // 2, 0), groups=256),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ChannelShuffleModule(channels=512, groups=16)
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
