import contextlib
import os
from abc import abstractmethod
from multiprocessing import Process
from threading import Thread

from utils import ModuleFindTool


class Mode:
    """
    [experimental]
    Mode is a class that simulates the running mode of the client.
    """
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def join(self):
        pass


class ClientProcessMode(Process):
    def __init__(self, client):
        Process.__init__(self)
        self.client = client

    def run(self):
        self.client.run()


class ClientThreadMode(Thread):
    def __init__(self, client):
        Thread.__init__(self)
        self.client = client

    def run(self):
        self.client.run()


class ModeFactory:
    @staticmethod
    def create_mode_instance(client, mode='thread', params=None):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            if mode == "process":
                return ClientProcessMode(client)
            elif mode == "thread":
                return ClientThreadMode(client)
            else:
                if params is None:
                    params = {}
                return ModuleFindTool.find_class_by_path(mode)(client, **params)


def running_mode(config, output=False):
    """
    running_mode is a function that returns the running mode of the client.

    Returns:
        The running mode of the client.
    """
    if not output:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            return _running_mode(config)
    else:
        return _running_mode(config)


def _running_mode(config):
    if not config:
        print("Running Mode: None, Default(Thread)")
        return "thread", {}
    if "mode" not in config['global']:
        print("Running Mode: Default(Thread)")
        return "thread", {}
    mode_config = config['global']['mode']
    if isinstance(mode_config, dict):
        print(f"Running Mode: {mode_config['path']}")
        if 'params' in mode_config:
            params = mode_config['params']
        else:
            params = {}
        return mode_config['path'], params
    elif isinstance(mode_config, str):
        if mode_config == 'thread':
            print("Running Mode: Thread")
            return "thread", {}
        elif mode_config == 'process':
            print("Running Mode: Process")
            return "process", {}
        else:
            raise ValueError('if mode isinstance str, mode must be "thread" or "process"')
    else:
        raise ValueError('mode config must be a dict or a str')

