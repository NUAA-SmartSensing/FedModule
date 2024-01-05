import torch.nn as nn

'''Long Short Term Memory Network'''


class LSTM(nn.Module):
    def __init__(self, train_shape, category):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
        '''
        self.lstm = nn.LSTM(train_shape[-1], 512, 2, batch_first=True)
        self.fc = nn.Linear(512, category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x, _ = self.lstm(x.squeeze(1))
        x = x[:, -1, :]
        x = self.fc(x)
        return x
