from checker.AllChecker import AllChecker


class CheckerCaller:
    def __init__(self, queue_manager):
        self.queue_manager = queue_manager

    def check(self, update, *args, **kwargs):
        if isinstance(self.queue_manager.checker, AllChecker):
            return self.queue_manager.checker.check()
        else:
            return self.queue_manager.checker.check(update)
