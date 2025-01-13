import pickle
import warnings
from abc import abstractmethod
from multiprocessing import Event
from time import sleep

from clientmanager.BaseClientManager import BaseClientManager
from clientmanager.ClientFactroy import ClientFactory
from clientmanager.NormalClientManager import NormalClientManager
from core.MessageQueue import MessageQueueFactory
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from utils.MQTT import MQTTClientSingleton


def create_event(config, client_id, event_name="select"):
    if "type" in config:
        if config["type"] == "mqtt":
            return MQTTDistributedEvent(config, client_id, event_name)
        else:
            raise Exception("Unsupported event type")
    else:
        event_class = ModuleFindTool.find_class_by_path(config["path"])
        return event_class(config["params"], client_id, event_name)


def create_event_list(config, client_id_list, event_name="select"):
    event_list = []
    for client_id in client_id_list:
        event_list.append(create_event(config, client_id, event_name))
    return event_list


class DistributedClientManager(BaseClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.config = whole_config["client_manager"]
        self.global_var = GlobalVarGetter.get()
        self.total_client_num = whole_config["global"]["client_num"]
        self.sub_manger_num = len(self.config["client_num"])
        self.message_queue = MessageQueueFactory.create_message_queue()
        self.communication_proxy = create_communication_proxy(
            self.config["communication_proxy"] if "communication_proxy" in self.config else None)
        self.client_num = sum(whole_config["client_manager"][
                                  "init_client_num"]) if "init_client_num" in self.config else self.total_client_num
        id_list, prefix_client_num = self.generate_id_list()
        self.client_list = id_list
        self.client_id_list = id_list
        self.client_staleness_list = whole_config["client_manager"]["stale_list"]
        self.index_list = whole_config["client_manager"]["index_list"]  # each client's index list
        self.stop_event_list = create_event_list(self.config["event"], self.client_id_list, event_name="stop")
        self.selected_event_list = create_event_list(self.config["event"], self.client_id_list)
        self.client_status = [0 for _ in self.client_id_list]
        self.global_var['selected_event_list'] = self.selected_event_list
        print("main-client_manager has initialized, waiting for sub-client_manager joining")
        self.wait_for_sub_client_manager()

    def generate_id_list(self):
        id_list = []
        prefix_client_num = [0]
        for i in range(self.sub_manger_num):
            prefix_client_num.append(prefix_client_num[-1] + self.config["client_num"][i])
        for i in range(self.sub_manger_num):
            if "init_client_num" in self.config:
                id_list += [prefix_client_num[i] + j for j in range(self.config["init_client_num"][i])]
            else:
                id_list += [prefix_client_num[i] + j for j in range(self.config["client_num"][i])]
        return id_list, prefix_client_num

    def wait_for_sub_client_manager(self):
        joined_machine = {}

        def callback(client, userdata, message):
            msg = pickle.loads(message.payload)
            nonlocal joined_machine
            if msg not in joined_machine:
                print(f"sub-client_manager {msg} has joined")
                joined_machine[msg] = 1

        self.communication_proxy.get(-1, "manager_affair_join", callback)
        while len(joined_machine) != self.sub_manger_num:
            sleep(0.1)
        print("All sub-client_manager has joined")

        joined_machine = {}

        def callback2(client, userdata, message):
            msg = pickle.loads(message.payload)
            print(f"sub-client_manager {msg} has received data")
            nonlocal joined_machine
            joined_machine[msg] = 1

        self.communication_proxy.get(-1, "manager_affair_data", callback2)
        print("Start transferring data to sub-client_manager")
        self.communication_proxy.send(-2, "manager_affair",
                                      [self.client_staleness_list, self.index_list])
        print("Waiting for data transferring to complete")
        while len(joined_machine) != self.sub_manger_num:
            sleep(0.1)
        print("Data transferring has completed")

    def start_all_clients(self):
        # need to notify all sub-manager to start clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        joined_machine = 0

        def callback(client, userdata, message):
            msg = pickle.loads(message.payload)
            print(f"sub-client_manager {msg} has started all clients")
            nonlocal joined_machine
            joined_machine += 1

        self.communication_proxy.get(-1, "manager_affair_start", callback)
        print("Sending start signal to all sub-client_manager")
        self.communication_proxy.send(-2, "manager_affair", "start")
        print("Waiting for all clients to start")
        ready_client = 0
        while ready_client < self.client_num:
            sleep(1)
            ready_client = self.message_queue.get_registered_client_num()
        print("All clients has started")

    def get_client_list(self):
        warnings.warn("distributed client manager can not get client instance")
        return self.client_list

    def get_client_id_list(self):
        return self.client_id_list

    def stop_all_clients(self):
        # stop all clients
        print("Sending stop signal to all sub-client_manager")
        self.communication_proxy.send(-2, "manager_affair", "stop")

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()

    def create_and_start_new_client(self, dev='cpu'):
        # notify sub client-manager
        pass

    def client_join(self):
        pass


class SubNormalClientManager(NormalClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.config = whole_config["client_manager"]
        self.manager_id = self.config["manager_id"]
        self.total_client_num = self.config["client_num"][self.manager_id]

        self.multi_gpu = whole_config["global"]["multi_gpu"]
        self.global_client_num = whole_config["global"]["client_num"]
        if "init_client_num" in self.config:
            self.client_num = self.config["init_client_num"][self.manager_id]
        else:
            self.client_num = self.total_client_num

        self.start_id = 0
        for i in range(self.manager_id):
            self.start_id += self.config["client_num"][i]

        self.client_config = whole_config["client"]

        self.client_dev = self.get_client_dev_list(self.total_client_num, self.multi_gpu)
        self.client_class = ModuleFindTool.find_class_by_path(whole_config["client"]["path"])
        self.global_var = GlobalVarGetter.get()
        self.client_id_list = [i for i in range(self.start_id, self.start_id + self.client_num)]
        # to activate the message queue to subscribe the client topic
        MessageQueueFactory.create_message_queue()
        self.client_status = {k: 0 for k in self.client_id_list}
        self.global_var['client_id_list'] = self.client_id_list
        self.stop_event_list = create_event_list(self.config["event"], self.client_id_list, event_name="stop")
        self.selected_event_list = create_event_list(self.config["event"], self.client_id_list)

        self.communication_proxy = create_communication_proxy(
            self.config.get("communication_proxy", None))
        self.client_factory = ModuleFindTool.find_class_by_path(
            self.config['client_factory']['path']) if 'client_factory' in self.config else ClientFactory
        self.init_event = Event()
        self.connect_to_server()

    def connect_to_server(self):
        def on_message(client, userdata, message):
            payload = message.payload
            msg = pickle.loads(payload)
            if isinstance(msg, str):
                if msg == "start":
                    # release the on_message thread, let the main thread continue
                    print("Received start signal, start all clients")
                    self.start_all_clients()
                    sleep(0.1)
                    self.init_event.set()
                elif msg == "stop":
                    print("Received stop signal, stop all clients")
                    self.stop_all_clients()
            else:
                print("Received data from server")
                self.client_staleness_list, self.index_list = msg[0], msg[1]
                self.communication_proxy.send(-1, "manager_affair_data", self.manager_id)

        print("Start connecting to server")
        self.communication_proxy.get(-2, "manager_affair", on_message)
        while not self.init_event.is_set():
            self.communication_proxy.send(-1, "manager_affair_join", self.manager_id)
            sleep(1)

    def start_all_clients(self):
        self.__init_clients()
        # start clients
        self.global_var['client_list'] = self.client_list
        print("Starting clients")
        for client in self.client_list:
            client.start()
        self.communication_proxy.send(-1, "manager_affair_start", self.manager_id)

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id - self.start_id].set()
        self.selected_event_list[client_id - self.start_id].set()

    def __init_clients(self):
        self.client_id_list = list(range(self.start_id, self.start_id + self.client_num))
        self.client_list = self.client_factory.create_clients(self.client_id_list, self.stop_event_list,
                                                              self.selected_event_list,
                                                              self.client_staleness_list, self.index_list,
                                                              self.client_config,
                                                              self.client_dev, self.global_var['config'])

    def create_and_start_new_client(self, dev='cpu'):
        # client_id 如何生成
        client_id = self.start_id + self.client_num
        self.selected_event_list.append(create_event(self.config, client_id))
        self.stop_event_list.append(create_event(self.config, client_id, event_name="stop"))
        self.client_list.append(self.client_factory.create_client(client_id, self.stop_event_list[client_id],
                                                                  self.selected_event_list[client_id],
                                                                  self.client_staleness_list[client_id],
                                                                  self.index_list[client_id],
                                                                  self.client_config, self.client_dev[client_id],
                                                                  self.global_var['config']))  # instance
        self.client_id_list.append(client_id)
        self.client_list[client_id].start()
        self.client_num += 1


class DistributedEvent:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def set(self):
        pass

    @abstractmethod
    def is_set(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def wait(self):
        pass


class MQTTDistributedEvent(DistributedEvent):
    client = None
    uid = None

    def __new__(cls, *args, **kwargs):
        if cls.client is None:
            cls.client = MQTTClientSingleton.get_client()
            cls.uid = MQTTClientSingleton.get_uid()
        return super().__new__(cls)

    def __init__(self, config, id, event_name):
        super().__init__(config)
        self.id = id
        self.event_name = event_name
        self.topic = f"{MQTTDistributedEvent.uid}/{self.event_name}/{id}"
        self.event = Event()
        MQTTDistributedEvent.client.subscribe(self.topic)

        def on_message(client, userdata, message):
            msg = pickle.loads(message.payload)
            if msg:
                self.event.set()
            else:
                self.event.clear()

        MQTTDistributedEvent.client.message_callback_add(self.topic, on_message)

    def set(self):
        MQTTDistributedEvent.client.publish(self.topic, pickle.dumps(True))
        self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self):
        self.event.wait()

    def clear(self):
        self.event.clear()
        MQTTDistributedEvent.client.publish(self.topic, pickle.dumps(False))


def create_communication_proxy(config):
    if config is None:
        return MQTTCommunicationProxy()
    if "type" in config:
        if config["type"] == "mqtt":
            return MQTTCommunicationProxy(config)
    elif "path" in config:
        return ModuleFindTool.find_class_by_path(config["path"])(config["params"])
    else:
        raise Exception("Unsupported communication proxy type")


class Proxy:
    def send(self, uid, topic, msg):
        raise NotImplementedError

    def get(self, uid, topic, callback):
        raise NotImplementedError


class BlockProxy(Proxy):
    pass


class NoBlockProxy(Proxy):
    pass


class MQTTCommunicationProxy(NoBlockProxy):
    client = None
    uid = None

    def __new__(cls, *args, **kwargs):
        if cls.client is None:
            cls.client = MQTTClientSingleton.get_client()
            cls.uid = MQTTClientSingleton.get_uid()
        return super().__new__(cls)

    def send(self, uid, topic, msg):
        # uid: -1 means server, -2 means broadcast
        if uid == -2:
            topic = f"{MQTTCommunicationProxy.uid}/{topic}"
        else:
            topic = f"{MQTTCommunicationProxy.uid}/{topic}/{uid}"
        MQTTCommunicationProxy.client.publish(topic, pickle.dumps(msg))

    def get(self, uid, topic, callback=None):
        if uid == -2:
            topic = f"{MQTTCommunicationProxy.uid}/{topic}"
        else:
            topic = f"{MQTTCommunicationProxy.uid}/{topic}/{uid}"
        MQTTCommunicationProxy.client.subscribe(topic)
        MQTTCommunicationProxy.client.message_callback_add(topic, callback)
