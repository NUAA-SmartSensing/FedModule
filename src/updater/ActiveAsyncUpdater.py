import copy

from core.MessageQueue import MessageQueueFactory
from updater.AsyncUpdater import AsyncUpdater


class ActiveAsyncUpdater(AsyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        super().__init__(server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)
        self.message_queue = MessageQueueFactory.create_message_queue()

    def server_update(self, epoch, update_list):
        super().server_update(epoch, update_list)
        self.message_queue.set_latest_model(copy.deepcopy(self.server_network.state_dict()), epoch)
