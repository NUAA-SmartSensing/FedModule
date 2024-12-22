from abc import ABC

import torch.utils.data
from torch.utils.data import DataLoader

from core.Component import Component
from core.MessageQueue import MessageQueueFactory
from loss.LossFactory import LossFactory
from update.UpdateCaller import UpdateCaller
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.Tools import random_seed_set


class BaseUpdater(Component, ABC):
    def __init__(self, server_thread_lock, config):
        super().__init__()
        self.queue_manager = None
        self.server_thread_lock = server_thread_lock
        self.config = config

        self.T = self.global_var['T']
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_t = self.global_var['current_t']
        self.schedule_t = self.global_var['schedule_t']
        self.model = self.global_var['global_model']

        self.message_queue = MessageQueueFactory.create_message_queue()
        self.test_ds = self.message_queue.get_test_dataset()
        self.test_index_list = self.global_var['test_index_list']
        self.fl_test_ds = FLDataset(self.test_ds, self.test_index_list)
        self.test_dl = DataLoader(self.fl_test_ds, batch_size=128, drop_last=True)

        # loss function
        self.loss_func = LossFactory.create_loss(config['loss'])

        # aggregation method
        update_class = ModuleFindTool.find_class_by_path(self.config['update']['path'])
        self.update_method = update_class(self.config['update']['params'])
        self.update_caller = UpdateCaller(self)

        self.message_queue = MessageQueueFactory.create_message_queue()
        self.optimizer = None
        # server_opt
        if "optimizer" in self.config:
            self.optimizer = ModuleFindTool.find_class_by_path(self.config['optimizer']['path'])(
                self.model.parameters(), **self.config["optimizer"]["params"])

    def init(self) -> None:
        random_seed_set(self.global_var['global_config']['seed'])
        self.queue_manager = self.global_var['queue_manager']
