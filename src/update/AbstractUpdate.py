from abc import abstractmethod


class AbstractUpdate:
    @abstractmethod
    def update_server_weights(self, *args, **kwargs):
        pass
