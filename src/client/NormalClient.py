import time

from torch.utils.data import DataLoader

from client import Client
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.ModelTraining import train_one_epoch


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
                data_sum, weights = self.train_one_epoch()

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

    def train_one_epoch(self):
        return train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func, self.opti, self.mu)

    def upload(self, data_sum, weights):
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp}
        self.queue_manager.put(update_dict)
