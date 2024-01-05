import torch.nn as nn

'''Channel Attention Neural Network: 通道注意力'''


class ChannelAttentionModule(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.att_fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // 4),
            nn.ReLU(),
            nn.Linear(inchannel // 4, inchannel),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        # 传感数据特殊性，固定模态轴，只在时序轴上做GAP或者GMP
        att = nn.AdaptiveAvgPool2d((1, x.size(-1)))(x)  # [b, c, series, modal] -> [b, c, 1, modal]
        att = att.permute(0, 3, 1, 2).squeeze(-1)  # [b, c, 1, modal] -> [b, modal, c]
        att = self.att_fc(att)  # [b, modal, c]
        att = att.permute(0, 2, 1).unsqueeze(-2)  # [b, modal, c] -> [b, c, modal] -> [b, c, 1, modal]
        out = x * att
        return out


class ChannelAttentionNeuralNetwork(nn.Module):
    def __init__(self, train_shape, category):
        super(ChannelAttentionNeuralNetwork, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(128),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(256),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)),
            ChannelAttentionModule(512),
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
