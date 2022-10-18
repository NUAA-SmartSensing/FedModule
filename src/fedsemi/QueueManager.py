import threading

from utils import ModuleFindTool


class QueueManager:
    def __init__(self, queue_list, current_time, config):
        self.queue_list = queue_list
        self.config = config
        checker_class = ModuleFindTool.find_class_by_path(f'fedsemi.checker.{config["checker_file"]}',
                                                          config["checker_name"])
        self.checker = checker_class(current_time, config["params"])
        self.lock = threading.Lock()

    def put(self, update):
        self.lock.acquire()
        if self.checker.check(update):
            self.queue_list[update["group_id"]].put(update)
        self.lock.release()

    def set_queue_list(self, queue_list):
        self.lock.acquire()
        self.queue_list = queue_list
        self.lock.release()
