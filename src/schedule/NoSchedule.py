from schedule.AbstractSchedule import AbstractSchedule


class NoSchedule(AbstractSchedule):
    def __init__(self, config):
        self.config = config
        self.init = False

    def schedule(self, client_list):
        if self.init:
            return []
        else:
            self.init = True
            return client_list
