from abc import abstractmethod


class AbstractGroup:
    @abstractmethod
    def group(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_update(self, *args, **kwargs):
        pass
