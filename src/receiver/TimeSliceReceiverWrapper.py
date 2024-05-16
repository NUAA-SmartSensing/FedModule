import time

from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class TimeSliceReceiverWrapper:
    def __init__(self, config):
        self.receiver = ModuleFindTool.find_class_by_path(config['params']['path'])(config['params'])
        self.server_finished_event = GlobalVarGetter.get()['server_finished_event']
        self.server_start_event = GlobalVarGetter.get()['server_start_event']

    def receive(self, *args, **kwargs):
        time.sleep(0.1)
        self.server_finished_event.set()
        print(args, kwargs)
        res = self.receiver.receive(*args, **kwargs)
        self.server_start_event.wait()
        self.server_start_event.clear()
        return res
