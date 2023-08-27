from checker.CheckerCaller import CheckerCaller
from queuemanager.BaseQueueManger import BaseQueueManager
from receiver.ReceiverCaller import ReceiverCaller
from utils import ModuleFindTool
from utils.Queue import Queue


class QueueListManager(BaseQueueManager):
    def __init__(self, config):
        BaseQueueManager.__init__(self, config)
        self.queue = [Queue() for _ in range(self.global_var["global_config"]["client_num"])]
        self.group_ready_num = 0

        # for clients
        checker_class = ModuleFindTool.find_class_by_path(config["checker"]["path"])
        self.checker = checker_class(config["checker"]["params"])
        self.checker_caller = CheckerCaller(self)

        # for server
        receiver_class = ModuleFindTool.find_class_by_path(config["receiver"]["path"])
        self.receiver = receiver_class(config["receiver"]["params"])
        self.receiver_caller = ReceiverCaller(self)

    # Automatically triggered when new data is uploaded
    def put(self, update, *args, **kwargs):
        self.lock.acquire()
        if self.checker_caller.check(update):
            self.queue[update["group_id"]].put(update)
        self.lock.release()

    def receive(self, nums, *args, **kwargs):
        self.group_ready_num = self.receiver_caller.receive(self.queue, nums, *args, **kwargs)

    def check(self, update, *args, **kwargs):
        self.checker_caller.check(update, *args, **kwargs)

    def get(self, *args, **kwargs):
        group_id, _, _ = self.__get_group_id(args, kwargs)
        return self.queue[group_id].get()

    def size(self, *args, **kwargs):
        group_id, _, _ = self.__get_group_id(args, kwargs)
        return self.queue[group_id].qsize()

    def empty(self, *args, **kwargs):
        group_id, _, _ = self.__get_group_id(args, kwargs)
        return self.queue[group_id].empty()

    def stop(self):
        total = 0
        for q in self.queue:
            if not q.empty():
                total = total + q.qsize()
                for i in range(q.qsize()):
                    q.get()
            q.close()
        print("\nUn-used client weights:", total)

    def __get_group_id(self, args, kwargs):
        if len(args) > 0:
            group_id = args[0]
            args = args[1:]
        else:
            group_id = kwargs['id']
        return group_id, args, kwargs
