from utils.GlobalVarGetter import GlobalVarGetter


class SyncChecker:
    def __init__(self, config):
        self.current_t = GlobalVarGetter.get()['current_t']
        self.config = config

    def check(self, update):
        if update["time_stamp"] == self.current_t.get_time():
            return True
        else:
            return False
