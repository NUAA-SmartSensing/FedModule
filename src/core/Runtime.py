from abc import abstractmethod
from multiprocessing import Process
from threading import Thread

from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter

CLIENT_STATUS = {'created': 0, 'active': 1, 'stopped': 2}


class Mode:
    """
    [experimental]
    Mode is a class that simulates the running mode of the client.
    """

    @abstractmethod
    def run(self):
        pass

    def start(self):
        pass

    def join(self):
        pass


def running_mode():
    """
    running_mode is a function that returns the running mode of the client.

    Returns:
        The running mode of the client.
    """
    if "mode" not in GlobalVarGetter.get()['global_config']:
        print("Running Mode: Thread")
        return Thread
    config = GlobalVarGetter.get()['global_config']['mode']
    if isinstance(config, dict):
        print(f"Running Mode: {config['path']}")
        mode = ModuleFindTool.find_class_by_path(config['path'])(config['params'])
        mode.run()
        return mode.get_manage_mode()
    elif isinstance(config, str):
        if config == 'thread':
            print("Running Mode: Thread")
            return Thread
        elif config == 'process':
            print("Running Mode: Process")
            return Process
        else:
            raise ValueError('if mode isinstance str, mode must be "thread" or "process"')
    else:
        raise ValueError('mode config must be a dict or a str')
