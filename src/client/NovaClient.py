import copy

from client.NormalClient import NormalClient


class NovaClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev)
        self.tau = 0

    def train_one_epoch(self):
        self.tau = 0
        global_model = copy.deepcopy(self.model.state_dict())
        # 设置迭代次数
        data_sum = len(self.train_dl)
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                self.tau += 1
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = self.model(data)
                # 计算损失函数
                loss = self.loss_func(preds, label)
                # 正则项
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                self.opti.step()
                # 将梯度归零，初始化梯度
                self.opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(self.model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k] - global_model[k]
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum, "time_stamp": self.time_stamp, "tau": self.tau}
        self.message_queue.put_into_uplink(update_dict)
