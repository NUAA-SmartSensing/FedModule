import collections
from abc import abstractmethod
from multiprocessing import Process, Event


class Client(Process):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, dev):
        Process.__init__(self)
        self.model = None
        self.client_id = c_id
        self.event = selected_event
        self.event.clear()
        self.stop_event = stop_event
        self.delay = delay
        self.train_ds = train_ds
        self.dev = dev

        self.weights_buffer = collections.OrderedDict()
        self.time_stamp = 0
        self.time_stamp_buffer = 0
        self.received_weights = False
        self.received_time_stamp = False
        self.params = {}
        self.event_is_set = False
        self.schedule_t = None
        self.index_list = index_list

    @abstractmethod
    def run(self):
        pass

    def train_one_epoch(self):
        pass

