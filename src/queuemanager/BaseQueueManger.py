import copy
import threading
from abc import abstractmethod

from queuemanager.AbstractQueueManager import AbstractQueueManager
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Queue import Queue


class BaseQueueManager(AbstractQueueManager):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.global_var = GlobalVarGetter.get()
        self.pre_queue = None
        self.lock = threading.Lock()
        self.queue = Queue()
        self.receiver = None
        self.checker = None

    # Automatically triggered when new data is uploaded
    @abstractmethod
    def put(self, update, *args, **kwargs):
        pass

    @abstractmethod
    def receive(self, *args, **kwargs):
        pass

    @abstractmethod
    def check(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        self.lock.acquire()
        self.queue = self.pre_queue
        self.lock.release()

    @abstractmethod
    def size(self, *args, **kwargs):
        pass

    @abstractmethod
    def empty(self, *args, **kwargs):
        pass

    def store(self):
        self.pre_queue = copy.deepcopy(self.queue)

    def stop(self):
        if not self.queue.empty():
            print("\nUn-used client weights:", self.queue.qsize())
            for q in range(self.queue.qsize()):
                self.queue.get()
        self.queue.close()

    def get_queue(self):
        return self.queue
