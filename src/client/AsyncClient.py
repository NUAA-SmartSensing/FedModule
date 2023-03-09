import copy
import time

from torch.utils.data import DataLoader

from client import Client
from utils import ModuleFindTool


class AsyncClient(Client.Client):
    def __init__(self, c_id, queue, stop_event, delay, train_ds, client_config, dev):
        Client.Client.__init__(self, c_id, stop_event, delay, train_ds, dev)
        self.queue = queue
        self.batch_size = client_config["batch_size"]
        self.epoch = client_config["epochs"]
        self.model_name = client_config["model_name"]
        self.optimizer_config = client_config["optimizer"]
        self.mu = client_config["mu"]
        self.config = client_config

        # 本地模型
        model_class = ModuleFindTool.find_class_by_path(f'model.{client_config["model_file"]}.{client_config["model_name"]}')
        self.model = model_class()
        self.model = self.model.to(self.dev)

        # 优化器
        opti_class = ModuleFindTool.find_class_by_path(f'torch.optim.{self.optimizer_config["name"]}')
        self.opti = opti_class(self.model.parameters(), lr=self.optimizer_config["lr"], weight_decay=self.optimizer_config["weight_decay"])

        # loss函数
        if isinstance(client_config["loss"], str):
            self.loss_func = ModuleFindTool.find_class_by_path(f'torch.nn.functional.{client_config["loss"]}')
        else:
            loss_func_class = ModuleFindTool.find_class_by_path(f'loss.{client_config["loss"]["loss_file"]}.{client_config["loss"]["loss_name"]}')
            self.loss_func = loss_func_class(client_config["loss"], self)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

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
                r_weights = copy.deepcopy(self.model.state_dict())
                data_sum, weights = self.train_one_epoch(r_weights)

                # client传回server的信息具有延迟
                print("Client", self.client_id, "trained")
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum, "time_stamp": self.time_stamp}
                self.queue.put(update_dict)
                self.event.clear()
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()

    def train_one_epoch(self, r_weights):
        return self.model.train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func, self.opti, self.mu)
