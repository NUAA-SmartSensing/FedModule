from abc import abstractmethod


class AbstractQueueManager:
    @abstractmethod
    def put(self, update, *args, **kwargs):
        pass

    @abstractmethod
    def set(self, *args, **kwargs):
        pass

    @abstractmethod
    def receive(self, *args, **kwargs):
        pass

    @abstractmethod
    def check(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    @abstractmethod
    def size(self, *args, **kwargs):
        pass

    @abstractmethod
    def empty(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_queue(self):
        pass

