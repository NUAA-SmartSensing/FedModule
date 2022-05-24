import time
import copy
from model import CNN, ConvNet
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from client import Client


class AsyncClient(Client.Client):
    def __init__(self, c_id, queue, stop_event, delay, train_ds, client_config):
        Client.Client.__init__(self, c_id, queue, stop_event, delay, train_ds)
        self.batch_size = client_config["batch_size"]
        self.epoch = client_config["epochs"]
        self.model_name = client_config["model_name"]
        if self.model_name == "CNN":
            self.model = CNN.CNN()
            self.model = self.model.to(self.dev)
            self.opti = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.005)
            self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        elif self.model_name == "ConvNet":
            self.model = ConvNet.ConvNet()
            self.model = self.model.to(self.dev)
            self.opti = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            all_range = list(range(len(self.train_ds)))
            indices = all_range[self.client_id::50]
            self.train_dl = DataLoader(self.train_ds, batch_size=4, num_workers=2, sampler=sampler.SubsetRandomSampler(indices))
        else:
            self.model = CNN.CNN()
            self.model = self.model.to(self.dev)
            self.opti = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.005)
            self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.loss_func = F.cross_entropy

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
        return self.model.train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func, self.opti)
