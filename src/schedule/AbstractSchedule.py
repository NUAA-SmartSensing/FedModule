from abc import abstractmethod


class AbstractSchedule:
    @abstractmethod
    def schedule(self, *args, **kwargs):
        pass
