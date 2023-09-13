import torch.nn as nn

'''Dilated Convolutional Neural Network: 空洞卷积'''


class DilatedConv(nn.Module):
    def __init__(self, train_shape, category, kernel_size=3, dilations=[1, 2, 3]):
        super(DilatedConv, self).__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
            kernel_size: 时序维度 卷积核尺寸
            dilations: 空洞率【为防止网格效应，串联堆叠空洞率应当只有公因子 1 】
        '''
        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, (kernel_size, 1), (2, 1), ((dilations[0] * (kernel_size - 1) + 1) // 2, 0),
                      dilation=dilations[0]),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (kernel_size, 1), (2, 1), ((dilations[1] * (kernel_size - 1) + 1) // 2, 0),
                      dilation=dilations[1]),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (kernel_size, 1), (2, 1), ((dilations[2] * (kernel_size - 1) + 1) // 2, 0),
                      dilation=dilations[2]),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, (kernel_size, 1), (2, 1), (kernel_size // 2, 0)),  # 感受野足够便不需要继续空洞Conv
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
