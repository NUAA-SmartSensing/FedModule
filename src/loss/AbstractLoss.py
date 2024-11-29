from torch import nn


class AbstractLoss(nn.Module):
    def __init__(self, belong_obj, config: dict):
        super().__init__()
        self.belong_obj = belong_obj
        self.config = config
