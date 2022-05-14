import threading
import time
import copy
import torch.nn as nn
from Model import CNN, ConvNet
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import collections


class AsyncClient(threading.Thread):
    def __init__(self, c_id, queue, batch_size, stop_event, delay, train_ds, epoch, loss_func):
        threading.Thread.__init__(self)
        self.client_id = c_id
        self.queue = queue
        self.event = threading.Event()
        self.event.clear()
        self.batch_size = batch_size
        self.stop_event = stop_event
        self.delay = delay
        self.train_ds = train_ds
        self.epoch = epoch
        self.time_stamp = 0
        self.client_thread_lock = threading.Lock()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "CNN"
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
        # self.opti = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_func = loss_func

        self.weights_buffer = collections.OrderedDict()
        self.time_stamp_buffer = 0
        self.received_weights = False
        self.received_time_stamp = False
        self.event_is_set = False

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
                self.queue.put((self.client_id, weights, data_sum, self.time_stamp))
                self.event.clear()
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()

    def set_client_id(self, new_id):
        self.client_thread_lock.acquire()
        self.client_id = new_id
        self.client_thread_lock.release()

    def get_client_id(self):
        # self.client_thread_lock.acquire()
        c_id = copy.deepcopy(self.client_id)
        # self.client_thread_lock.release()
        return c_id

    def set_client_weight(self, weights):
        # self.client_thread_lock.acquire()
        self.weights_buffer = weights
        self.received_weights = True
        # self.client.set_client_weights(weights)
        # self.client_thread_lock.release()

    def get_client_weight(self):
        # self.client_thread_lock.acquire()
        client_weights = copy.deepcopy(self.model.state_dict())
        # self.client_thread_lock.release()
        return client_weights

    def set_event(self):
        # self.client_thread_lock.acquire()
        self.event_is_set = True
        self.event.set()
        # self.client_thread_lock.release()

    def get_event(self):
        # self.client_thread_lock.acquire()
        event_is_set = self.event.is_set()
        # self.client_thread_lock.release()
        return event_is_set

    def set_time_stamp(self, current_time):
        # self.client_thread_lock.acquire()
        self.time_stamp_buffer = current_time
        self.received_time_stamp = True
        # self.time_stamp = current_time
        # self.client_thread_lock.release()

    def get_time_stamp(self):
        # self.client_thread_lock.acquire()
        t_s = copy.deepcopy(self.time_stamp)
        # self.client_thread_lock.release()
        return t_s

    def set_delay(self, new_delay):
        self.client_thread_lock.acquire()
        self.delay = new_delay
        self.client_thread_lock.release()

    def get_delay(self):
        # self.client_thread_lock.acquire()
        delay = copy.deepcopy(self.delay)
        # self.client_thread_lock.release()
        return delay

    def train_one_epoch(self, r_weights):
        return self.model.train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func, self.opti)
