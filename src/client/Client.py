import threading
import torch
import collections
import copy
from abc import abstractmethod


class Client(threading.Thread):
    def __init__(self, c_id, stop_event, delay, train_ds, dev):
        threading.Thread.__init__(self)
        self.client_id = c_id
        self.event = threading.Event()
        self.event.clear()
        self.stop_event = stop_event
        self.delay = delay
        self.train_ds = train_ds
        self.client_thread_lock = threading.Lock()
        self.dev = dev
        self.weights_buffer = collections.OrderedDict()
        self.time_stamp = 0
        self.time_stamp_buffer = 0
        self.received_weights = False
        self.received_time_stamp = False
        self.params = {}
        self.event_is_set = False

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
