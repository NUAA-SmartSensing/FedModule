import copy
import time

from torch.utils.data import DataLoader

from client import Client
from loss.LossFactory import LossFactory
from utils import ModuleFindTool


class NormalClient(Client.Client):
    def __init__(self, c_id, stop_event, delay, train_ds, config, dev):
        Client.Client.__init__(self, c_id, stop_event, delay, train_ds, dev)
        self.queue_manager = self.global_var["queue_manager"]
        self.batch_size = config["batch_size"]
        self.epoch = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.mu = config["mu"]
        self.config = config

        # 本地模型
        model_class = ModuleFindTool.find_class_by_path(config["model"]["path"])
        self.model = model_class(**config["model"]["params"])
        self.model = self.model.to(self.dev)

        # 优化器
        opti_class = ModuleFindTool.find_class_by_path(self.optimizer_config["path"])
        self.opti = opti_class(self.model.parameters(), **self.optimizer_config["params"])

        # loss函数
        self.loss_func = LossFactory(config["loss"], self).create_loss()

        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def run(self):
        while not self.stop_event.is_set():
            if self.received_weights:
                # 更新模型参数
                self.model.load_state_dict(self.weights_buffer, strict=True)
                self.received_weights = False
            if self.received_time_stamp:
                self.time_stamp = self.time_stamp_buffer
                self.received_time_stamp = False
            if self.event_is_set:
                self.event_is_set = False

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.client_thread_lock.acquire()
                # 该client进行训练
                data_sum, weights = self.train()

                # client传回server的信息具有延迟
                self.print_lock.acquire()
                print("Client", self.client_id, "trained")
                self.print_lock.release()
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                self.upload(data_sum, weights)
                self.event.clear()
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()

    def train(self):
        return self.train_one_epoch()

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp}
        self.queue_manager.put(update_dict)

    def train_one_epoch(self):
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        # 设置迭代次数
        data_sum = 0
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = self.model(data)
                # 计算损失函数
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
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
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights
