import warnings
from abc import abstractmethod
from typing import Optional


class HandlerChain:
    def __init__(self, chain=None):
        self._head = chain

    def add_handler_after(self, handler, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            while it is not None:
                if isinstance(it, target_cls):
                    it.insert_next(handler)
                    break

    def add_handler_before(self, handler, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            if isinstance(it, target_cls):
                handler.insert_next(it)
                self._head = handler
            else:
                while it.next_handler is not None:
                    if isinstance(it.next_handler, target_cls):
                        handler.insert_next(it.next_handler)
                        it.insert_next(handler)
                        break
                    it = it.next_handler

    def exchange_handler(self, handler, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            if isinstance(it, target_cls):
                handler.insert_next(it.next_handler)
                self._head = handler
            else:
                while it.next_handler is not None:
                    if isinstance(it.next_handler, target_cls):
                        handler.insert_next(it.next_handler.next_handler)
                        it.insert_next(handler)
                        break
                    it = it.next_handler

    def handle(self, request):
        return self._head.handle(request)

    def set_chain(self, handler) -> "Handler":
        self._head = handler
        return handler


class Handler:
    """抽象处理程序"""

    def __init__(self, handler: Optional["Handler"] = None):
        self.next_handler = handler
        self.has_run = False

    def run_once(self, request):
        pass

    def set_next(self, handler: "Handler") -> "Handler":
        self.next_handler = handler
        return handler

    def next(self) -> "Handler":
        return self.next_handler

    def insert_next(self, handler):
        if self.next_handler is None:
            warnings.warn("The next handler is None, the handler will be set as the next handler.")
            self.next_handler = handler
        else:
            handler.next_handler = self.next_handler
            self.next_handler = handler

    @abstractmethod
    def _handle(self, request):
        pass

    def handle(self, request):
        if not self.has_run:
            self.run_once(request)
            self.has_run = True
        response = self._handle(request)
        if self.next_handler:
            return self.next_handler.handle(response)
        return response

    def __call__(self, *args, **kwargs):
        return self.handle(*args, **kwargs)
