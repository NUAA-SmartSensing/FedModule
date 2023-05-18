class SyncChecker:
    def __init__(self, current_time, config):
        self.current_time = current_time
        self.config = config

    def check(self, update):
        if update["time_stamp"] == self.current_time.get_time():
            return True
        else:
            return False
