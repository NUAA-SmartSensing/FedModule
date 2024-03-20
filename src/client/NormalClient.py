import copy
import time
import torch
from torch.utils.data import DataLoader

from client.Client import Client
from client.mixin.Gradient import GradientMixin
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DataReader import FLDataset
from utils.Tools import to_cpu


class NormalClient(Client):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        Client.__init__(self, c_id, stop_event, selected_event, delay, index_list, dev)
        self.fl_train_ds = None
        self.opti = None
        self.loss_func = None
        self.train_dl = None
        self.batch_size = config["batch_size"]
        self.epoch = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.mu = config["mu"]
        self.config = config

    def run(self):
        self.init_client()
        while not self.stop_event.is_set():
            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.event.clear()
                self.message_queue.set_training_status(self.client_id, True)
                self.wait_notify()
                self.local_task()
                self.message_queue.set_training_status(self.client_id, False)
            # 该client等待被选中
            else:
                self.event.wait()

    def local_task(self):
        # 该client进行训练
        data_sum, weights = self.train()

        # client传回server的信息具有延迟
        print("Client", self.client_id, "trained")
        time.sleep(self.delay)

        # 返回其ID、模型参数和时间戳
        self.upload(data_sum, weights)

    def train(self):
        data_sum, weights = self.train_one_epoch()
        return data_sum, to_cpu(weights)

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp}
        self.message_queue.put_into_uplink(update_dict)

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
        weights = self.model.state_dict()
        torch.cuda.empty_cache()
        return data_sum, weights

    def wait_notify(self):
        if self.message_queue.get_from_downlink(self.client_id, 'received_weights'):
            if self.training_params is None:
                self.training_params = self.message_queue.get_training_params()
            self.message_queue.put_into_downlink(self.client_id, 'received_weights', False)
            weights_buffer = self.message_queue.get_from_downlink(self.client_id, 'weights_buffer')
            state_dict = self.model.state_dict()
            for k in weights_buffer:
                if self.training_params[k]:
                    state_dict[k] = weights_buffer[k]
            self.model.load_state_dict(state_dict)
        if self.message_queue.get_from_downlink(self.client_id, 'received_time_stamp'):
            self.message_queue.put_into_downlink(self.client_id, 'received_time_stamp', False)
            self.time_stamp = self.message_queue.get_from_downlink(self.client_id, 'time_stamp_buffer')
            self.schedule_t = self.message_queue.get_from_downlink(self.client_id, 'schedule_time_stamp_buffer')

    def init_client(self):
        config = self.config
        self.train_ds = self.message_queue.get_train_dataset()

        self.transform, self.target_transform = self._get_transform(config)
        self.fl_train_ds = FLDataset(self.train_ds, list(self.index_list), self.transform, self.target_transform)

        self.model = self._get_model(config)
        self.model = self.model.to(self.dev)

        # 优化器
        opti_class = ModuleFindTool.find_class_by_path(self.optimizer_config["path"])
        self.opti = opti_class(self.model.parameters(), **self.optimizer_config["params"])

        # loss函数
        self.loss_func = LossFactory(config["loss"], self).create_loss()

        self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    @staticmethod
    def _get_transform(config):
        transform, target_transform = None, None
        if "transform" in config:
            transform_func = ModuleFindTool.find_class_by_path(config["transform"]["path"])
            transform = transform_func(**config["transform"]["params"])
        if "target_transform" in config:
            target_transform_func = ModuleFindTool.find_class_by_path(config["target_transform"]["path"])
            target_transform = target_transform_func(**config["target_transform"]["params"])
        return transform, target_transform

    @staticmethod
    def _get_model(config):
        # 本地模型
        model_class = ModuleFindTool.find_class_by_path(config["model"]["path"])
        for k, v in config["model"]["params"].items():
            if isinstance(v, str):
                config["model"]["params"][k] = eval(v)
        return model_class(**config["model"]["params"])


class NormalClientWithGrad(NormalClient, GradientMixin):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        GradientMixin.__init__(self)

    def train(self):
        self._save_global_model(self.model.state_dict())
        return super().train()

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "data_sum": data_sum,
                       "time_stamp": self.time_stamp, "weights": self._to_gradient()}
        self.message_queue.put_into_uplink(update_dict)
