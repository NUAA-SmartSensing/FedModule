import time

from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class TimeSliceReceiverWrapper:
    def __init__(self, config):
        self.receiver = ModuleFindTool.find_class_by_path(config['params']['path'])(config['params'])
        self.server_finished_event = GlobalVarGetter.get()['server_finished_event']
        self.server_start_permit_event = GlobalVarGetter.get()['server_start_permit_event']
        self.server_start_request_event = GlobalVarGetter.get()['server_start_request_event']

    def receive(self, *args, **kwargs):
        self.server_finished_event.set()
        res = self.receiver.receive(*args, **kwargs)
        self.server_start_request_event.set()
        self.server_start_permit_event.wait()
        self.server_start_permit_event.clear()
        return res
