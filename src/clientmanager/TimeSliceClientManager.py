import warnings
from multiprocessing import Event

from clientmanager.BaseClientManager import BaseClientManager
from core.TimeSlice import TimeSliceRunner
from utils.GlobalVarGetter import GlobalVarGetter
from core.MessageQueue import MessageQueueFactory


class TimeSliceClientManager(BaseClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.global_var = GlobalVarGetter.get()
        self.stop_event = Event()
        self.stop_event_list = [Event() for _ in range(whole_config["global"]["client_num"])]
        self.selected_event_list = [Event() for _ in range(whole_config["global"]["client_num"])]
        self.global_var['selected_event_list'] = self.selected_event_list
        self.create_client_event = Event()
        self.init_client_event = Event()
        self.server_finished_event = Event()
        self.server_start_event = Event()
        self.global_var['server_finished_event'] = self.server_finished_event
        self.global_var['server_start_event'] = self.server_start_event
        self.join_event = Event()
        self.client_id_list = list(range(whole_config["global"]["client_num"]))
        client_dev = self.get_client_dev_list(whole_config["global"]["client_num"], whole_config["global"]["multi_gpu"])
        self.runner = TimeSliceRunner(self.init_client_event, self.create_client_event, self.join_event,
                                      self.stop_event_list, self.stop_event, self.selected_event_list,
                                      self.server_finished_event, self.server_start_event,
                                      {"index_list": whole_config["client_manager"]["index_list"],
                                       "client_config": whole_config["client"],
                                       "multi_gpu": whole_config["global"]["multi_gpu"],
                                       "client_staleness_list": whole_config["client_manager"]["stale_list"],
                                       "client_num": whole_config["global"]["client_num"], "client_dev": client_dev,
                                       "client_manager": whole_config["client_manager"]})
        self.runner.start()

    def start_all_clients(self):
        self.global_var['client_id_list'] = self.client_id_list
        self.init_client_event.set()

    def stop_all_clients(self):
        self.stop_event.set()
        for e in self.stop_event_list:
            e.set()

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()

    def create_and_start_new_client(self, client_delay, dev='cpu'):
        mq = MessageQueueFactory.create_message_queue()
        mq.put_into_downlink(-1, "client_delay", client_delay)
        mq.put_into_downlink(-1, "dev", dev)
        self.create_client_event.set()
        self.client_id_list.append(len(self.client_id_list))
        self.client_list.append(len(self.client_list))

    def client_join(self):
        self.join_event.wait()
        print("All clients joined.")
        self.runner.join()
        print("client runner joined.")
        print("-----------------Client Manager joined.-----------------")

    def get_client_list(self):
        warnings.warn("This function is not implemented.")
        return self.client_list
