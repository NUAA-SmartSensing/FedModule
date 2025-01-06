import warnings
from abc import abstractmethod, ABC
from typing import Optional


def _function_wrapper(handler):
    if callable(handler):
        new_handler = FunctionHandler(handler)
    else:
        new_handler = handler
    return new_handler


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


class FunctionHandler(Handler):
    def __init__(self, handler: callable):
        super().__init__()
        self.handle_func = handler

    def _handle(self, request):
        return self.handle_func(request)


class Filter(Handler, ABC):
    def __init__(self, handler=None):
        super().__init__(handler)
        self.real_next = self.next_handler

    def set_next(self, handler: "Handler") -> "Handler":
        self.real_next = handler
        return super().set_next(handler)

    def insert_next(self, handler):
        self.real_next = handler
        super().insert_next(handler)

    def handle(self, request):
        if self._handle(request):
            self.next_handler = self.real_next
            return request
        else:
            self.next_handler = None
            return request


class HandlerChain(Handler):
    def __init__(self, chain=None):
        super().__init__()
        self._head = chain

    def add_handler_after(self, handler, target_cls):
        handler = _function_wrapper(handler)

        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            while it is not None:
                if isinstance(it, target_cls):
                    it.insert_next(handler)
                    break
                it = it.next_handler

    def add_handler_before(self, handler, target_cls):
        handler = _function_wrapper(handler)

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
        handler = _function_wrapper(handler)
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            if isinstance(it, target_cls):
                handler.set_next(it.next_handler)
                self._head = handler
            else:
                while it.next_handler is not None:
                    if isinstance(it.next_handler, target_cls):
                        handler.set_next(it.next_handler.next_handler)
                        it.set_next(handler)
                        break
                    it = it.next_handler

    def remove_handler(self, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
        else:
            it = self._head
            if isinstance(it, target_cls):
                self._head = it.next_handler
            else:
                while it.next_handler is not None:
                    if isinstance(it.next_handler, target_cls):
                        it.set_next(it.next_handler.next_handler)
                        break
                    it = it.next_handler

    def add_handler(self, handler):
        handler = _function_wrapper(handler)
        if self._head is None:
            self._head = handler
        else:
            it = self._head
            while it.next_handler is not None:
                it = it.next_handler
            it.next_handler = handler

    def _handle(self, request):
        if self._head is None:
            return request
        return self._head.handle(request)

    def set_chain(self, handler) -> "Handler":
        self._head = handler
        return handler


class TreeFilter(Handler, ABC):
    def __init__(self, handler=None):
        super().__init__(handler)
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def handle(self, request):
        res = self._handle(request)
        return self.children[res].handle(request)

