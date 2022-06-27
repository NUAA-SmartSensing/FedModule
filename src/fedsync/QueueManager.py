from utils import ModuleFindTool


class QueueManager:
    def __init__(self, queue, current_time, config):
        self.queue = queue
        self.config = config
        checker_class = ModuleFindTool.find_class_by_string("checker", config["checker_file"], config["checker_name"])
        self.checker = checker_class(current_time, config)

    def put(self, update):
        if self.checker.check(update):
            self.queue.put(update)
