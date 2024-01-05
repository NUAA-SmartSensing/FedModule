import copy
import time

import torch

from client import TestClient


class PerAvg_Client(TestClient.TestClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        TestClient.TestClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev)
        self.beta = 0.5

    def train_one_epoch(self):
        # 设置迭代次数
        data_sum = 0
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                temp_model = copy.deepcopy(self.model.state_dict())

                # step 1
                output = self.model(data)
                # 计算损失函数
                loss = self.loss_func(output, label)
                data_sum += label.size(0)
                # 将梯度归零，初始化梯度
                self.opti.zero_grad()
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                self.opti.step()

                # step2
                self.opti.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, label)
                # 反向传播
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.state_dict(), temp_model):
                    old_param.data = new_param.data.clone()

                self.opti.step(beta=self.beta)

        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(self.model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights
