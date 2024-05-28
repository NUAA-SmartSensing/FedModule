from abc import abstractmethod


class AbstractSchedule:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def schedule(self, *args, **kwargs):
        pass
