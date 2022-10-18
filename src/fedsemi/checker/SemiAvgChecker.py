
class SemiAvgChecker:
    def __init__(self, current_time, config):
        self.current_time = current_time
        self.config = config

    def check(self, update):
        # 上传即接受
        return True
