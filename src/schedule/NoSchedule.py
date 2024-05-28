from schedule.AbstractSchedule import AbstractSchedule


class NoSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.init = False

    def schedule(self, client_list):
        if self.init:
            return []
        else:
            self.init = True
            return client_list
