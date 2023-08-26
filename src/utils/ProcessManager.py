from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread

from utils.GlobalVarGetter import GlobalVarGetter


class ManagerWrapper:
    _manager = None

    @staticmethod
    def get_manager():
        if not ManagerWrapper._manager:
            ManagerWrapper.__register()
            ManagerWrapper._manager = SyncManager()
            ManagerWrapper._manager.start()
        return ManagerWrapper._manager

    @staticmethod
    def del_manager():
        if ManagerWrapper._manager:
            ManagerWrapper._manager.shutdown()
            ManagerWrapper._manager = None

    @staticmethod
    def __register():
        SyncManager.register('MessageQueue', MessageQueue)


# nothing about server or client
class MessageQueue:
    uplink = {'update': Queue()}
    downlink = {'received_weights': {}, 'received_time_stamp': {}, 'time_stamp_buffer': {}, 'weights_buffer': {}}

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
        return MessageQueue.downlink[key][client_id]

    @staticmethod
    def put_into_downlink(client_id, key, item):
        if key not in MessageQueue.downlink.keys():
            MessageQueue.downlink[key] = {}
        MessageQueue.downlink[key][client_id] = item

    @staticmethod
    def uplink_empty(key='update'):
        return MessageQueue.uplink[key].empty()

    @staticmethod
    def downlink_empty(client_id, key):
        return MessageQueue.downlink[key][client_id].empty()
