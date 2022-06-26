from utils import ModuleFindTool


class QueueManager:
    def __init__(self, queue, current_time, config):
        self.queue = queue
        checker_class = ModuleFindTool.find_class_by_string("check", config["checker_file"], config["checker_name"])
        self.checker = checker_class(current_time)

    def put(self, update, config):
        if self.checker.check(update, config):
            self.queue.put(update)
