import os
import sys
from abc import abstractmethod
from multiprocessing import Process
from threading import Thread

from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter

CLIENT_STATUS = {'created': 0, 'running': 1, 'stale': 2, 'exited': 3, 'active': 4}
SERVER_STATUS = {'stale': 0, 'schedule': 1, 'aggregate': 2, 'exited': 3}


class Mode:
    """
    [experimental]
    Mode is a class that simulates the running mode of the client.
    """

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def join(self):
        pass


def running_mode(config):
    """
    running_mode is a function that returns the running mode of the client.

    Returns:
        The running mode of the client.
    """
    # this condition is used to a new python environment, i.e., process
    if not config:
        print("Running Mode: None")
        return Process
    if "mode" not in config['global']:
        print("Running Mode: Default(Thread)")
        return Thread
    mode_config = config['global']['mode']
    if isinstance(mode_config, dict):
        mode = ModuleFindTool.find_class_by_path(mode_config['path'])
        print(f"Running Mode: {mode_config['path']}")
        return mode
    elif isinstance(mode_config, str):
        if mode_config == 'thread':
            print("Running Mode: Thread")
            return Thread
        elif mode_config == 'process':
            print("Running Mode: Process")
            return Process
        else:
            raise ValueError('if mode isinstance str, mode must be "thread" or "process"')
    else:
        raise ValueError('mode config must be a dict or a str')


def running_mode_for_client():
    global_var = GlobalVarGetter.get()
    if global_var == "--multiprocessing-fork":
        # this condition only exists in multiprocessing-fork process
        # and client is the process
        return Process
    else:
        config = global_var['config']
    sys.stdout = open(os.devnull, 'w')
    mode = running_mode(config)
    sys.stdout = sys.__stdout__
    return mode
