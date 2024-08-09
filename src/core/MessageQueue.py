import copy
import multiprocessing as mp
import pickle
import threading
from multiprocessing.managers import SyncManager
from queue import Queue
from threading import Thread
from time import sleep

from utils.GlobalVarGetter import GlobalVarGetter
from utils.MQTT import MQTTClientSingleton


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
    _address_port = 50000
    _address_host = ''
    _address_init = False

    @staticmethod
    def get_manager(main_process=False):
        if not ManagerWrapper._address_init:
            ManagerWrapper._address_init = True
            config = GlobalVarGetter.get()['config']['global']
            if 'message_queue' in config:
                if 'address' in config:
                    ManagerWrapper._address_host = config['message_queue']['address']
                if 'port' in config:
                    ManagerWrapper._address_port = config['message_queue']['port']
        if main_process and ManagerWrapper._manager is None:
            ManagerWrapper.__register()
            ManagerWrapper._manager = SyncManager(address=(ManagerWrapper._address_host, ManagerWrapper._address_port))
            ManagerWrapper._manager.start()
        elif not main_process and ManagerWrapper._manager is None:
            ManagerWrapper.__register()
            ManagerWrapper._manager = SyncManager(address=(ManagerWrapper._address_host, ManagerWrapper._address_port))
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
    def create_uplink(key, dtype=dict):
        MessageQueue.uplink[key] = dtype()

    @staticmethod
    def create_downlink(key, dtype=dict):
        MessageQueue.downlink[key] = dtype()

    @staticmethod
    def get_from_downlink(client_id, key):
        if key not in MessageQueue.downlink:
            return None
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
    def get_registered_client_num():
        return len(MessageQueue.training_status)

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


class MessageQueueWrapperForMQTT:
    message_queue = None
    client = None
    uid = None
    mask_list = []
    __id_init = False

    def __new__(cls, *args, **kwargs):
        global_var = GlobalVarGetter.get()
        if MessageQueueWrapperForMQTT.message_queue is None:
            MessageQueueWrapperForMQTT.message_queue = MessageQueue()
        if cls.client is None:
            cls.client = MQTTClientSingleton.get_client()
            cls.uid = MQTTClientSingleton.get_uid()
            config = global_var['config']['global']['message_queue']
            if 'mask_list' in config:
                cls.mask_list = config['mask_list']

            def on_message(client, userdata, message):
                topic = message.topic
                msg = pickle.loads(message.payload)
                if topic == f'{cls.uid}/mq/uplink':
                    item, key = msg
                    cls.message_queue.put_into_uplink(item, key)
                elif topic == f'{cls.uid}/mq/training_status':
                    client_id, status = msg
                    cls.message_queue.set_training_status(client_id, status)
                elif topic == f'{cls.uid}/mq/training_params':
                    cls.message_queue.set_training_params(msg)
                elif topic == f'{cls.uid}/mq/config':
                    value = msg
                    if isinstance(value, tuple):
                        k, v = value
                        cls.message_queue.set_config_by_key(k, v)
                    else:
                        cls.message_queue.set_config(value)
                elif topic == f'{cls.uid}/mq/latest_model':
                    model, current_t = msg
                    cls.message_queue.set_latest_model(model, current_t)
                elif topic == f'{cls.uid}/mq/train_dataset':
                    cls.message_queue.set_train_dataset(msg)
                elif topic == f'{cls.uid}/mq/test_dataset':
                    cls.message_queue.set_test_dataset(msg)
                else:
                    topic, client_id = topic.split('/')[-2:]
                    key, item = msg
                    cls.message_queue.put_into_downlink(int(client_id), key, item)

            # sub_manager
            if "manager_id" in global_var["config"]["client_manager"]:
                cls.client.subscribe(f'{cls.uid}/mq/train_dataset') if 'train_dataset' not in cls.mask_list else None
                cls.client.subscribe(f'{cls.uid}/mq/test_dataset') if 'test_dataset' not in cls.mask_list else None
                cls.client.subscribe(f'{cls.uid}/mq/latest_model') if 'latest_model' not in cls.mask_list else None
                cls.client.subscribe(
                    f'{cls.uid}/mq/training_params') if 'training_params' not in cls.mask_list else None
                cls.client.subscribe(f'{cls.uid}/mq/config') if 'config' not in cls.mask_list else None
            # server
            else:
                cls.client.subscribe(f'{cls.uid}/mq/uplink') if 'uplink' not in cls.mask_list else None
                cls.client.subscribe(
                    f'{cls.uid}/mq/training_status') if 'training_status' not in cls.mask_list else None
            cls.client.message_callback_add(f'{cls.uid}/mq/#', on_message)
        elif not cls.__id_init and "manager_id" in global_var["config"]["client_manager"]:
            if 'downlink' not in cls.mask_list and len(global_var['client_id_list']) != 0:
                cls.__id_init = True
                for cid in global_var['client_id_list']:
                    cls.client.subscribe(f'{cls.uid}/mq/downlink/{cid}')
        return super().__new__(cls)

    @staticmethod
    def get_from_uplink(key='update'):
        return MessageQueueWrapperForMQTT.message_queue.get_from_uplink(key)

    @staticmethod
    def put_into_uplink(item, key='update'):
        if 'uplink' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}/mq/uplink',
                                                      pickle.dumps((item, key)))
        return MessageQueueWrapperForMQTT.message_queue.put_into_uplink(item, key)

    @staticmethod
    def create_uplink(key, dtype=dict):
        raise NotImplementedError

    @staticmethod
    def create_downlink(key, dtype=dict):
        raise NotImplementedError

    @staticmethod
    def get_from_downlink(client_id, key):
        return MessageQueueWrapperForMQTT.message_queue.get_from_downlink(client_id, key)

    @staticmethod
    def put_into_downlink(client_id, key, item):
        if 'downlink' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}/mq/downlink/{client_id}',
                                                      pickle.dumps((key, item)))
        return MessageQueueWrapperForMQTT.message_queue.put_into_downlink(client_id, key, item)

    @staticmethod
    def uplink_empty(key='update'):
        return not MessageQueue.uplink[key].qsize()

    @staticmethod
    def downlink_empty(client_id, key):
        return MessageQueue.downlink[key][client_id].empty()

    @staticmethod
    def set_training_status(client_id, value):
        if 'training_status' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}/mq/training_status',
                                                      pickle.dumps((client_id, value)))
        return MessageQueueWrapperForMQTT.message_queue.set_training_status(client_id, value)

    @staticmethod
    def get_training_status():
        return MessageQueue.training_status

    @staticmethod
    def get_registered_client_num():
        return len(MessageQueue.training_status)

    @staticmethod
    def set_training_params(value):
        if 'training_params' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}/mq/training_params',
                                                      pickle.dumps(value))
        return MessageQueueWrapperForMQTT.message_queue.set_training_params(value)

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
        if 'config' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}-config', pickle.dumps(config),
                                                      retain=True)
        return MessageQueueWrapperForMQTT.message_queue.set_config(config)

    @staticmethod
    def set_config_by_key(k, v):
        if 'config' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}-config', pickle.dumps({k: v}))
        return MessageQueueWrapperForMQTT.message_queue.set_config_by_key(k, v)

    @staticmethod
    def get_config():
        return MessageQueue.config

    @staticmethod
    def get_config_by_key(key):
        return MessageQueue.config[key]

    @staticmethod
    def set_latest_model(model, current_t):
        if 'latest_model' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}-latest_model',
                                                      pickle.dumps((model, current_t)), retain=True)
        return MessageQueueWrapperForMQTT.message_queue.set_latest_model(model, current_t)

    @staticmethod
    def get_latest_model():
        return copy.deepcopy(MessageQueue.latest_model), MessageQueue.current_t

    @staticmethod
    def set_train_dataset(train_dataset):
        if 'train_dataset' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}-train_dataset',
                                                      pickle.dumps(train_dataset), retain=True)
        return MessageQueueWrapperForMQTT.message_queue.set_train_dataset(train_dataset)

    @staticmethod
    def get_train_dataset():
        return copy.deepcopy(MessageQueue.train_dataset)

    @staticmethod
    def set_test_dataset(test_dataset):
        if 'test_dataset' not in MessageQueueWrapperForMQTT.mask_list:
            MessageQueueWrapperForMQTT.client.publish(f'{MessageQueueWrapperForMQTT.uid}/test_dataset',
                                                      pickle.dumps(test_dataset), retain=True)
        return MessageQueueWrapperForMQTT.message_queue.set_test_dataset(test_dataset)

    @staticmethod
    def get_test_dataset():
        return copy.deepcopy(MessageQueue.test_dataset)


class MessageQueueFactory:
    @staticmethod
    def create_message_queue(main_process=False):
        config = GlobalVarGetter.get()['config']['global']
        if 'message_queue' in config:
            if 'type' in config['message_queue']:
                if config['message_queue']['type'] == 'mqtt':
                    return MessageQueueWrapperForMQTT()
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
