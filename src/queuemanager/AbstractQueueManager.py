from abc import abstractmethod
from functools import wraps


def counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return func(*args, **kwargs)

    wrapper.count = 0
    return wrapper


class Meta(type):
    def __new__(cls, name, bases, attrs):
        if 'get' in attrs:
            attrs['get'] = counter(attrs['get'])
        return super().__new__(cls, name, bases, attrs)


class AbstractQueueManager(metaclass=Meta):

    # Automatically triggered when new data is uploaded
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

