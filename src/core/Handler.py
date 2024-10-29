import warnings
from abc import ABC, abstractmethod


class Handler(ABC):
    """抽象处理程序"""
    _next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    def insert_next(self, handler):
        if self._next_handler is None:
            warnings.warn("The next handler is None, the handler will be set as the next handler.")
            self._next_handler = handler
        else:
            handler._next_handler = self._next_handler
            self._next_handler = handler

    @abstractmethod
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

