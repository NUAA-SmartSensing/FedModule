import copy

from core.MessageQueue import MessageQueueFactory
from core.handlers.Handler import Handler
from core.handlers.ModelTestHandler import ServerTestHandler
from updater.AsyncUpdater import AsyncUpdater
from utils.Tools import to_cpu


class ActiveAsyncUpdater(AsyncUpdater):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        super().__init__(server_thread_lock, config, mutex_sem, empty_sem, full_sem)

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_before(ActiveModelSender(), ServerTestHandler)


class ActiveModelSender(Handler):
    def __init__(self):
        super().__init__()
        self.message_queue = MessageQueueFactory.create_message_queue()

    def _handle(self, request):
        updater = request.get("updater")
        self.message_queue.put_into_downlink("all", "weights", to_cpu(updater.model.state_dict()))
        return request
