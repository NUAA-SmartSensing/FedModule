class AvgCheck:
    def __init__(self, current_time):
        self.current_time = current_time

    def check(self, update, config):
        if update["time_stamp"] == self.current_time.get_time():
            return True
        else:
            return False
