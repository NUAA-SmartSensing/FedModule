from abc import abstractmethod


class AbstractReceiver:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def receive(self, *args, **kwargs):
        pass
