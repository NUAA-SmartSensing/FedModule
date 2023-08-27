from checker.CheckerCaller import CheckerCaller
from queuemanager.BaseQueueManger import BaseQueueManager
from receiver.ReceiverCaller import ReceiverCaller
from utils import ModuleFindTool


class SingleQueueManager(BaseQueueManager):
    def __init__(self, config):
        BaseQueueManager.__init__(self, config)
        # for clients
        checker_class = ModuleFindTool.find_class_by_path(config["checker"]["path"])
        self.checker = checker_class(config["checker"])
        self.checker_caller = CheckerCaller(self)

        # for server
        receiver_class = ModuleFindTool.find_class_by_path(config["receiver"]["path"])
        self.receiver = receiver_class(config["receiver"])
        self.receiver_caller = ReceiverCaller(self)

    # Automatically triggered when new data is uploaded
    def put(self, update, *args, **kwargs):
        self.lock.acquire()
        if self.checker_caller.check(update):
            self.queue.put(update)
        self.lock.release()

    def set(self, *args, **kwargs):
        self.lock.acquire()
        self.queue = self.pre_queue
        self.lock.release()

    def receive(self, nums, *args, **kwargs):
        self.receiver_caller.receive(self.queue, nums, *args, **kwargs)

    def check(self, update, *args, **kwargs):
        self.checker_caller.check(update, *args, **kwargs)

    def get(self, *args, **kwargs):
        return self.queue.get()

    def size(self, *args, **kwargs):
        return self.queue.qsize()

    def empty(self, *args, **kwargs):
        return self.queue.empty()

    def get_queue(self):
        return self.queue
