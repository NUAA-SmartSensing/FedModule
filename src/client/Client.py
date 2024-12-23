from abc import ABC

from core.Component import ComponentForClient
from core.MessageQueue import MessageQueueFactory


class Client(ComponentForClient, ABC):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, dev):
        super().__init__()
        self.model = None
        self.client_id = c_id
        self.event = selected_event
        self.stop_event = stop_event
        self.delay = delay
        self.dev = dev
        self.time_stamp = 0
        self.params = {}
        self.event_is_set = False
        self.schedule_t = None
        self.index_list = index_list
        self.message_queue = MessageQueueFactory.create_message_queue()
        self.training_params = None

        self.train_ds = None
        self.transform = None
        self.target_transform = None
