import collections
import copy
import threading
from abc import abstractmethod

from utils.GlobalVarGetter import GlobalVarGetter


class Client(threading.Thread):
    def __init__(self, c_id, stop_event, delay, train_ds, dev):
        threading.Thread.__init__(self)
        self.model = None
        self.client_id = c_id
        self.event = threading.Event()
        self.event.clear()
        self.stop_event = stop_event
        self.delay = delay
        self.train_ds = train_ds
        self.client_thread_lock = threading.Lock()
        self.dev = dev
        self.global_var = GlobalVarGetter().get()
        self.print_lock = self.global_var['print_lock']

        self.weights_buffer = collections.OrderedDict()
        self.time_stamp = 0
        self.time_stamp_buffer = 0
        self.received_weights = False
        self.received_time_stamp = False
        self.params = {}
        self.event_is_set = False
        self.schedule_t = None

    @abstractmethod
    def run(self):
        pass

    def set_client_id(self, new_id):
        self.client_thread_lock.acquire()
        self.client_id = new_id
        self.client_thread_lock.release()

    def get_client_id(self):
        c_id = copy.deepcopy(self.client_id)
        return c_id

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params

    def set_client_weight(self, weights):
        self.weights_buffer = weights
        self.received_weights = True

    def get_client_weight(self):
        client_weights = copy.deepcopy(self.model.state_dict())
        return client_weights

    def set_event(self):
        self.event_is_set = True
        self.event.set()

    def get_event(self):
        event_is_set = self.event.is_set()
        return event_is_set

    def set_time_stamp(self, current_time):
        self.time_stamp_buffer = current_time
        self.received_time_stamp = True

    def get_time_stamp(self):
        t_s = copy.deepcopy(self.time_stamp)
        return t_s

    def set_delay(self, new_delay):
        self.client_thread_lock.acquire()
        self.delay = new_delay
        self.client_thread_lock.release()

    def get_delay(self):
        delay = copy.deepcopy(self.delay)
        return delay

    def getDataset(self):
        return self.train_ds

    def set_schedule_time_stamp(self, schedule_t):
        self.schedule_t = schedule_t

    def train_one_epoch(self, epoch, dev, train_dl, model, loss_func, opti, mu):
        if mu != 0:
            global_model = copy.deepcopy(model)
        # 设置迭代次数
        data_sum = 0
        for epoch in range(epoch):
            for data, label in train_dl:
                data, label = data.to(dev), label.to(dev)
                # 模型上传入数据
                preds = model(data)
                # 计算损失函数
                loss = loss_func(preds, label)
                data_sum += label.size(0)
                # 正则项
                if mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (mu / 2) * proximal_term
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights

