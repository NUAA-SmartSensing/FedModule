from abc import abstractmethod

from utils.GlobalVarGetter import GlobalVarGetter


class Component:
    def __init__(self):
        self.finals = []
        self.handler_chain = None
        self.global_var = GlobalVarGetter.get()
        self.create_handler_chain()

    def run(self) -> None:
        self.init()
        self._run_iteration()
        self._final_callback()
        self.finish()

    def init(self) -> None:
        pass

    def finish(self) -> None:
        pass

    @abstractmethod
    def create_handler_chain(self):
        pass

    @abstractmethod
    def _run_iteration(self) -> None:
        pass

    def _final_callback(self) -> None:
        for func, params in self.finals:
            func(*params)

    def add_final_callback(self, func, *params) -> None:
        self.finals.append((func, params))
