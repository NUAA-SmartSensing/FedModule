from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class ReceiverWrapper:
    def __init__(self, config):
        self.receiver = ModuleFindTool.generate_object_by_path(config['path'], config['params'])
        self.server_finished_event = GlobalVarGetter.get()['server_finished_event']
        self.client_finished_event = GlobalVarGetter.get()['client_finished_event']

    def receive(self, *args, **kwargs):
        self.server_finished_event.set()
        res = self.receiver.receive(*args, **kwargs)
        self.client_finished_event.wait()
        self.server_finished_event.clear()
        return res
