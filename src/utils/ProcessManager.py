import multiprocessing
import threading
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread
from time import sleep

from utils.GlobalVarGetter import GlobalVarGetter


class ManagerWrapper:
    _manager = None

    @staticmethod
    def get_manager():
        if not ManagerWrapper._manager:
            ManagerWrapper.__register()
            ManagerWrapper._manager = SyncManager()
            ManagerWrapper._manager.start()
            config = GlobalVarGetter().get()
            keys = ['config', 'global_config', 'server_config', 'client_config',
                    'client_manager_config', 'queue_manager_config']
            new_config = {k: config[k] for k in keys}
            ManagerWrapper._manager.MessageQueue().set_config(new_config)
        return ManagerWrapper._manager

    @staticmethod
    def del_manager():
        if ManagerWrapper._manager:
            ManagerWrapper._manager.shutdown()
            ManagerWrapper._manager = None

    @staticmethod
    def __register():
        SyncManager.register('MessageQueue', MessageQueue)


# this thread works in main process
class DataGetter(Thread):
    def __init__(self):
        super().__init__()
        self.is_end = False
        self.queue_manager = None
        self.message_queue = None

    def run(self) -> None:
        self.queue_manager = GlobalVarGetter().get()['queue_manager']
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
    uplink = {'update': Queue()}
    downlink = {'received_weights': {}, 'received_time_stamp': {}, 'time_stamp_buffer': {}, 'weights_buffer': {},
                'schedule_time_stamp_buffer': {}}
    training_status = {}
    config = None
    latest_model = None
    current_t = None

    @staticmethod
    def get_from_uplink(key='update'):
        return MessageQueue.uplink[key].get()

    @staticmethod
    def put_into_uplink(item, key='update'):
        if key != 'update' and key not in MessageQueue.uplink.keys():
            MessageQueue.uplink[key] = Queue()
        MessageQueue.uplink[key].put(item)

    @staticmethod
    def get_from_downlink(client_id, key):
        if client_id in MessageQueue.downlink[key]:
            return MessageQueue.downlink[key][client_id]
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
    def set_config(config):
        MessageQueue.config = config

    @staticmethod
    def get_config(key):
        return MessageQueue.config[key]

    @staticmethod
    def set_latest_model(model, current_t):
        MessageQueue.latest_model = model
        MessageQueue.current_t = current_t

    @staticmethod
    def get_latest_model():
        return MessageQueue.latest_model, MessageQueue.current_t


class EventFactory:
    @staticmethod
    def create_Event():
        if 'mode' in GlobalVarGetter().get()['global_config'] and GlobalVarGetter().get()['global_config']['mode'] == 'process':
            return multiprocessing.Event()
        else:
            return threading.Event()


class MessageQueueFactory:
    @staticmethod
    def create_message_queue():
        if 'mode' in GlobalVarGetter().get()['global_config'] and GlobalVarGetter().get()['global_config']['mode'] == 'process':
            return ManagerWrapper.get_manager().MessageQueue()
        else:
            return MessageQueue()
