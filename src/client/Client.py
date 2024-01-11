from abc import abstractmethod
from threading import Thread

from torch.multiprocessing import Process

from utils.ProcessManager import MessageQueueFactory, mode_is_process


class Client(Process if mode_is_process() else Thread):
    def __init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, dev):
        super().__init__()
        self.model = None
        self.init_lock = init_lock
        self.client_id = c_id
        self.event = selected_event # 每轮被选中事件
        self.event.clear()
        self.stop_event = stop_event # 共享stop_event，终止学习
        self.delay = delay # stale
        self.train_ds = train_ds # 所有的数据 配合index_list使用
        self.dev = dev
        self.time_stamp = 0
        self.params = {}
        self.event_is_set = False
        self.schedule_t = None
        self.index_list = index_list # 对应的数据样本标签
        self.message_queue = MessageQueueFactory.create_message_queue() # 上下行传播传播模型参数
        self.training_params = None

        self.train_ds = None
        self.transform = None
        self.target_transform = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def train_one_epoch(self):
        pass

    @abstractmethod
    def wait_notify(self):
        pass
