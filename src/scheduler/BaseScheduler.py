from abc import ABC

from core.Component import Component
from core.MessageQueue import MessageQueueFactory
from schedule.ScheduleCaller import ScheduleCaller
from utils import ModuleFindTool
from utils.Tools import random_seed_set


class BaseScheduler(Component, ABC):
    def __init__(self, server_thread_lock, config):
        super().__init__()
        self.queue_manager = None
        self.selected_event_list = None
        self.server_thread_lock = server_thread_lock
        self.config = config

        self.T = self.global_var['T']
        self.current_t = self.global_var['current_t']
        self.schedule_t = self.global_var['schedule_t']
        model = self.global_var['global_model']
        self.server_weights = model.state_dict()
        self.download_dict = {}

        schedule_class = ModuleFindTool.find_class_by_path(config["schedule"]["path"])
        self.schedule_method = schedule_class(config["schedule"]["params"])
        self.schedule_caller = ScheduleCaller(self)

        self.message_queue = MessageQueueFactory.create_message_queue()

    def init(self) -> None:
        random_seed_set(self.global_var['global_config']['seed'])
        self.selected_event_list = self.global_var['selected_event_list']
        self.queue_manager = self.global_var['queue_manager']

    def customize_download(self):
        pass

    def download_item(self, target, k, v):
        if target not in self.download_dict:
            self.download_dict[target] = {}
        self.download_dict[target][k] = v

    def notify_client(self):
        self.customize_download()
        for target, download_dict in self.download_dict.items():
            for k, v in download_dict.items():
                self.message_queue.put_into_downlink(target, k, v)
