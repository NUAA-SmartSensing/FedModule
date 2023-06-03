from schedule.AbstractSchedule import AbstractSchedule


class FullSchedule(AbstractSchedule):
    def __init__(self, config):
        self.config = config

    def schedule(self, client_list):
        return client_list
