import numpy as np
import torch

from loss.AbstractLoss import AbstractLoss


class FedLC(AbstractLoss):
    def __init__(self, belong_object, config):
        super().__init__(belong_object, config)
        self.dev = belong_object.dev
        dataset = self.belong_obj.fl_train_ds.dataset
        self.z = torch.from_numpy(np.bincount(dataset.targets))
        self.tau = config.get('tau', 0.6)

    def forward(self, x, y, reduction="mean"):
        # reduction = "mean" or "sum"
        # input是模型输出的结果，与target求loss
        # target的长度和input第一维的长度一致
        # target的元素值为目标class
        # reduction默认为mean，即对loss求均值
        # 还有另一种为sum，对loss求和

        # 这里对input所有元素求exp
        z = torch.cat((self.z, torch.zeros(x.shape[1] - self.z.shape[0], dtype=torch.int64)), 0)
        z = z.to(self.dev)
        x = x.to(self.dev)
        x = x - self.tau * z ** -0.25
        exp = torch.exp(x)
        # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
        tmp1 = exp.gather(1, y.unsqueeze(-1)).squeeze()
        # 在exp第一维求和，这是softmax的分母
        tmp2 = exp.sum(1)
        # softmax公式：ei / sum(ej)
        softmax = tmp1 / tmp2
        # cross-entropy公式： -yi * log(pi)
        # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
        # 公式中的pi就是softmax的结果
        log = -torch.log(softmax)
        # 官方实现中，reduction有mean/sum及none
        # 只是对交叉熵后处理的差别
        if reduction == "mean":
            return log.mean()
        else:
            return log.sum()
