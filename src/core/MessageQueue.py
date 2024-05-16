import copy
import multiprocessing as mp
import threading
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread
from time import sleep

from utils.GlobalVarGetter import GlobalVarGetter


def running_mode_for_mq():
    """
    running_mode is a function that returns the running mode of the message queue.

    Returns:
        The running mode of the message queue.
    """
    config = GlobalVarGetter.get()
    if not config or "global_config" not in config:
        return 'process'
    if "mode" not in config['global_config']:
        return 'thread'
    mode_config = config['global_config']['mode']
    if isinstance(mode_config, dict):
        if 'mq' not in mode_config:
            return 'process'
        elif mode_config['mq'] == 'thread':
            return 'thread'
        return 'process'
    elif isinstance(mode_config, str):
        if mode_config == 'thread':
            return 'thread'
        elif mode_config == 'process':
            return 'process'
        else:
            raise ValueError('if mode isinstance str, mode must be "thread" or "process"')
    else:
        raise ValueError('mode config must be a dict or a str')


class ManagerWrapper:
    _manager = None

    @staticmethod
    def get_manager(main_process=False):
        if main_process and ManagerWrapper._manager is None:
            ManagerWrapper.__register()
            ManagerWrapper._manager = SyncManager(address=('', 50000))
            ManagerWrapper._manager.start()
        elif not main_process and ManagerWrapper._manager is None:
            ManagerWrapper.__register()
            ManagerWrapper._manager = SyncManager(address=('', 50000))
            ManagerWrapper._manager.connect()
        return ManagerWrapper._manager

    @staticmethod
    def del_manager():
        if ManagerWrapper._manager:
            ManagerWrapper._manager.shutdown()
            ManagerWrapper._manager = None

    @staticmethod
    def __register():
        SyncManager.register('MessageQueue', MessageQueue)

    @staticmethod
    def register(cls_name, cls):
        SyncManager.register(cls_name, cls)


# this thread works in main process
class DataGetter(Thread):
    def __init__(self):
        super().__init__()
        self.is_end = False
        self.queue_manager = None
        self.message_queue = None

    def run(self) -> None:
        self.queue_manager = GlobalVarGetter.get()['queue_manager']
        self.message_queue = MessageQueueFactory.create_message_queue()
        while not self.is_end:
            while not self.message_queue.uplink_empty():
                update = self.message_queue.get_from_uplink()
                self.queue_manager.put(update)
            # Give up cpu to other threads
            sleep(0.01)

    def kill(self):
        self.is_end = True


# make sure this class is no about server or client
class MessageQueue:
    train_dataset = None
    test_dataset = None
    uplink = {'update': Queue()}
    downlink = {'received_weights': {}, 'received_time_stamp': {}, 'time_stamp_buffer': {}, 'weights_buffer': {},
                'schedule_time_stamp_buffer': {}, 'group_id': {}}
    training_status = {}
    training_params = {}
    config = {}
    latest_model = None
    current_t = None

    @staticmethod
    def get_from_uplink(key='update'):
        return copy.deepcopy(MessageQueue.uplink[key].get())

    @staticmethod
    def put_into_uplink(item, key='update'):
        if key != 'update' and key not in MessageQueue.uplink.keys():
            MessageQueue.uplink[key] = Queue()
        MessageQueue.uplink[key].put(item)

    @staticmethod
    def get_from_downlink(client_id, key):
        if client_id in MessageQueue.downlink[key]:
            return copy.deepcopy(MessageQueue.downlink[key][client_id])
        return None

    @staticmethod
    def put_into_downlink(client_id, key, item):
        if key not in MessageQueue.downlink.keys():
            MessageQueue.downlink[key] = {}
        MessageQueue.downlink[key][client_id] = item

    @staticmethod
    def uplink_empty(key='update'):
        return not MessageQueue.uplink[key].qsize()

    @staticmethod
    def downlink_empty(client_id, key):
        return MessageQueue.downlink[key][client_id].empty()

    @staticmethod
    def set_training_status(client_id, value):
        MessageQueue.training_status[client_id] = value

    @staticmethod
    def get_training_status():
        return MessageQueue.training_status

    @staticmethod
    def set_training_params(value):
        MessageQueue.training_params = value

    @staticmethod
    def get_training_params():
        return MessageQueue.training_params

    @staticmethod
    def get_training_client_num():
        total = 0
        for _, v in MessageQueue.training_status.items():
            if v:
                total = total + 1
        return total

    @staticmethod
    def set_config(config):
        MessageQueue.config = config

    @staticmethod
    def set_config_by_key(k, v):
        MessageQueue.config[k] = v

    @staticmethod
    def get_config():
        return MessageQueue.config

    @staticmethod
    def get_config_by_key(key):
        return MessageQueue.config[key]

    @staticmethod
    def set_latest_model(model, current_t):
        MessageQueue.latest_model = model
        MessageQueue.current_t = current_t

    @staticmethod
    def get_latest_model():
        return copy.deepcopy(MessageQueue.latest_model), MessageQueue.current_t

    @staticmethod
    def set_train_dataset(train_dataset):
        MessageQueue.train_dataset = train_dataset

    @staticmethod
    def get_train_dataset():
        return copy.deepcopy(MessageQueue.train_dataset)

    @staticmethod
    def set_test_dataset(test_dataset):
        MessageQueue.test_dataset = test_dataset

    @staticmethod
    def get_test_dataset():
        return copy.deepcopy(MessageQueue.test_dataset)


class MessageQueueFactory:
    @staticmethod
    def create_message_queue(main_process=False):
        mode = running_mode_for_mq()
        if mode == 'thread':
            return MessageQueue()
        elif mode == 'process':
            return ManagerWrapper.get_manager(main_process).MessageQueue()
        else:
            return ManagerWrapper.get_manager(main_process).MessageQueue()


class EventFactory:
    @staticmethod
    def create_Event():
        mode = running_mode_for_mq()
        if mode == 'thread':
            return threading.Event()
        elif mode == 'process':
            return mp.Event()
        else:
            return mp.Event()
